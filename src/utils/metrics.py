import torch
import torch.nn.functional as F
import math

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def ssim(pred, target, C1=0.01**2, C2=0.03**2):
    mu_x = pred.mean()
    mu_y = target.mean()

    sigma_x = pred.var()
    sigma_y = target.var()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()

    return ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) *
        (sigma_x + sigma_y + C2)
    )
