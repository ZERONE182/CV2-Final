import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt

def dev():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def logsnr_schedule_cosine(t, *, logsnr_min=-20., logsnr_max=20.):
    b = np.arctan(np.exp(-.5 * logsnr_max))
    a = np.arctan(np.exp(-.5 * logsnr_min)) - b

    return -2. * torch.log(torch.tan(a * t + b))


def xt2batch(x, logsnr, z, R, T, K):
    return {
        'x': x,
        'z': z,
        'logsnr': torch.stack([logsnr_schedule_cosine(torch.zeros_like(logsnr)), logsnr], dim=1),
        'R': R,
        't': T,
        'K': K,
    }


def q_sample(z, logsnr, noise):
    # lambdas = logsnr_schedule_cosine(t)

    alpha = logsnr.sigmoid().sqrt()
    sigma = (-logsnr).sigmoid().sqrt()

    alpha = alpha[:, None, None, None]
    sigma = sigma[:, None, None, None]

    return alpha * z + sigma * noise


def p_losses(denoise_model, img, R, T, K, logsnr, noise=None, loss_type="l2", cond_prob=0.1):
    B, N, C, H, W = img.shape
    x = img[:, 0]
    z = img[:, 1]
    if noise is None:
        noise = torch.randn_like(x)

    z_noisy = q_sample(z=z, logsnr=logsnr, noise=noise)

    cond_mask = (torch.rand((B,)) > cond_prob)

    x_condition = torch.where(cond_mask[:, None, None, None], x, torch.randn_like(x))

    batch = xt2batch(x=x_condition, logsnr=logsnr, z=z_noisy, R=R, T=T, K=K)

    predicted_noise = denoise_model(batch, cond_mask=cond_mask)

    if loss_type == 'l1':
        loss = F.l1_loss(noise.to(dev()), predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise.to(dev()), predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise.to(dev()), predicted_noise)
    else:
        raise NotImplementedError()

    rec_img = reconstruct_z_start(z_noisy.to(dev()), predicted_noise, logsnr.to(dev()))
    img_color_mean = torch.mean(z, dim=(2, 3))
    rec_color_mean = torch.mean(rec_img, dim=(2, 3)).to(dev())

    color_loss = torch.nn.MSELoss()
    color_loss = (logsnr.to(dev()) + 20) / 20 * color_loss(img_color_mean, rec_color_mean)

    return loss + color_loss


@torch.no_grad()
def sample(model, img, R, T, K, w, timesteps=256):
    x = img[:, 0]
    img = torch.randn_like(x)
    imgs = []

    logsnrs = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps + 1)[:-1])
    logsnr_nexts = logsnr_schedule_cosine(torch.linspace(1., 0., timesteps + 1)[1:])

    for logsnr, logsnr_next in tqdm(zip(logsnrs, logsnr_nexts)):  # [1, ..., 0] = size is 257
        img = p_sample(model, x=x, z=img, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next, w=w)  # [B, C, H, W]
        imgs.append(img.cpu().numpy())
    return imgs


def reconstruct_z_start(z_noisy, pred_noise, logsnr):
    B = z_noisy.shape[0]
    logsnr_next = torch.tensor([20.0] * B).to(dev())
    c = - torch.special.expm1(logsnr - logsnr_next)[:, None, None, None]
    squared_alpha, squared_alpha_next = logsnr.sigmoid(), logsnr_next.sigmoid()
    squared_sigma, squared_sigma_next = (-logsnr).sigmoid(), (-logsnr_next).sigmoid()
    alpha, sigma, alpha_next = map(lambda a: a.sqrt(), (squared_alpha, squared_sigma, squared_alpha_next))
    alpha = alpha[:, None, None, None]
    sigma = sigma[:, None, None, None]
    alpha_next = alpha_next[:, None, None, None]

    z_start = (z_noisy - sigma * pred_noise) / alpha
    z_start.clamp_(-1., 1.)

    z_start = alpha_next * (z_noisy * (1 - c) / alpha + c * z_start)
    return z_start


@torch.no_grad()
def p_sample(model, x, z, R, T, K, logsnr, logsnr_next, w):
    model_mean, model_variance = p_mean_variance(model, x=x, z=z, R=R, T=T, K=K, logsnr=logsnr, logsnr_next=logsnr_next,
                                                 w=w)

    if logsnr_next == 0:
        return model_mean

    return model_mean + model_variance.sqrt() * torch.randn_like(x).cpu()


@torch.no_grad()
def p_mean_variance(model, x, z, R, T, K, logsnr, logsnr_next, w=2.0):
    b = x.shape[0]
    w = w[:, None, None, None]

    c = - torch.special.expm1(logsnr - logsnr_next)
    # c = 1 - e^{\lambda_t - \lambda_s}

    squared_alpha, squared_alpha_next = logsnr.sigmoid(), logsnr_next.sigmoid()
    squared_sigma, squared_sigma_next = (-logsnr).sigmoid(), (-logsnr_next).sigmoid()

    alpha, sigma, alpha_next = map(lambda a: a.sqrt(), (squared_alpha, squared_sigma, squared_alpha_next))

    batch = xt2batch(x, logsnr.repeat(b), z, R, T, K)

    pred_noise = model(batch, cond_mask=torch.tensor([True] * b)).detach().cpu()
    batch['x'] = torch.randn_like(x)
    pred_noise_unconditioned = model(batch, cond_mask=torch.tensor([False] * b)).detach().cpu()

    pred_noise_final = (1 + w) * pred_noise - w * pred_noise_unconditioned

    z = z.detach().cpu()

    z_start = (z - sigma * pred_noise_final) / alpha
    z_start.clamp_(-1., 1.)

    model_mean = alpha_next * (z * (1 - c) / alpha + c * z_start)

    posterior_variance = squared_sigma_next * c

    return model_mean, posterior_variance
