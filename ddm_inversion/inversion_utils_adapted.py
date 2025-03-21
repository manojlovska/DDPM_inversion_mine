import torch
from tqdm import tqdm

def get_uncond_embedding(model):
    # Create a dummy unconditional embedding from an empty prompt.
    empty_token = model.tokenizer([""], padding="max_length", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        uncond_embedding = model.text_encoder(empty_token)[0]
    return uncond_embedding

def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Creates a forward noisy trajectory from x0:
      x_t = sqrt(alpha_bar[t])*x0 + sqrt(1 - alpha_bar[t])*noise,
    where noise ~ N(0,I).
    """
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros((num_inference_steps+1,
                       model.unet.config.in_channels,
                       model.unet.sample_size,
                       model.unet.sample_size), device=x0.device)
    xts[0] = x0
    for t in reversed(timesteps):
        idx = num_inference_steps - t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
    return xts

def forward_step(model, model_output, timestep, sample, uncond_embedding):
    """
    A simplified forward step (DDIM-style) that computes the next noisy latent.
    """
    next_timestep = min(model.scheduler.config.num_train_timesteps - 2,
                        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t
    # Compute predicted original sample (x0 estimate)
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / (alpha_prod_t ** 0.5)
    next_sample = model.scheduler.add_noise(pred_original_sample,
                                            model_output,
                                            torch.LongTensor([next_timestep]).to(sample.device))
    return next_sample

def get_variance(model, timestep):
    """
    Compute variance at a given timestep according to the scheduler.
    """
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    variance = ((1 - alpha_prod_t / alpha_prod_t_prev) * (1 - alpha_prod_t_prev)) / beta_prod_t
    return variance

def inversion_forward_process_uncond(model, x0, etas=1.0, num_inference_steps=50, prog_bar=False):
    """
    Unconditional inversion forward process.
    Given an input latent x0, this function constructs a noisy path (xts)
    and recovers the noise maps (zs) that would perfectly reconstruct x0.
    """
    uncond_embedding = get_uncond_embedding(model)
    
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.config.in_channels, 
        model.unet.sample_size,
        model.unet.sample_size
    )
    if isinstance(etas, (int, float)) and etas == 0:
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if isinstance(etas, (int, float)):
            etas = [etas] * model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        zs = torch.zeros(size=variance_noise_shape, device=model.device)
        alpha_bar = model.scheduler.alphas_cumprod

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    op = tqdm(timesteps) if prog_bar else timesteps

    for t in op:
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        if not eta_is_zero:
            xt = xts[idx+1][None]
        with torch.no_grad():
            out = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)
        noise_pred = out.sample
        if eta_is_zero:
            # DDIM-style inversion when eta == 0.
            xt = forward_step(model, noise_pred, t, xt, uncond_embedding)
        else:
            # DDPM-style inversion: compute predicted x0 and extract noise.
            xtm1 = xts[idx][None]
            pred_original_sample = (xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred) / (alpha_bar[t] ** 0.5)
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
            variance = get_variance(model, t)
            pred_sample_direction = ((1 - alpha_prod_t_prev - etas[idx] * variance) ** 0.5) * noise_pred
            mu_xt = (alpha_prod_t_prev ** 0.5) * pred_original_sample + pred_sample_direction
            z = (xtm1 - mu_xt) / (etas[idx] * variance ** 0.5)
            zs[idx] = z
            # Correction to avoid numerical drift.
            xtm1 = mu_xt + (etas[idx] * variance ** 0.5) * z
            xts[idx] = xtm1

    if zs is not None:
        zs[0] = torch.zeros_like(zs[0])
    return xt, zs, xts

def reverse_step(model, model_output, timestep, sample, eta=0, variance_noise=None, uncond_embedding=None):
    """
    Performs one reverse (denoising) step.
    """
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / (alpha_prod_t ** 0.5)
    variance = get_variance(model, timestep)
    std_dev_t = eta * variance ** 0.5
    pred_sample_direction = ((1 - alpha_prod_t_prev - eta * variance) ** 0.5) * model_output
    prev_sample = (alpha_prod_t_prev ** 0.5) * pred_original_sample + pred_sample_direction
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        prev_sample = prev_sample + eta * variance ** 0.5 * variance_noise
    return prev_sample

def inversion_reverse_process_uncond(model, xT, etas=0.0, zs=None, num_inference_steps=50, prog_bar=False):
    """
    Unconditional inversion reverse process.
    Starting from latent xT and given stored noise maps zs,
    this function reconstructs x0 by reversing the forward process.
    """
    uncond_embedding = get_uncond_embedding(model)
    
    if isinstance(etas, (int, float)):
        etas = [etas] * model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)
    xt = xT.expand(1, -1, -1, -1)  # Ensure batch dimension.
    op = tqdm(timesteps[-zs.shape[0]:]) if prog_bar and zs is not None else timesteps[-zs.shape[0]:]
    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0]:])}

    for t in op:
        idx = model.scheduler.num_inference_steps - t_to_idx[int(t)] - (model.scheduler.num_inference_steps - zs.shape[0] + 1)
        with torch.no_grad():
            out = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)
        noise_pred = out.sample
        xt = reverse_step(model, noise_pred, t, xt, eta=etas[idx], variance_noise=zs[idx] if zs is not None else None, uncond_embedding=uncond_embedding)
    return xt, zs
