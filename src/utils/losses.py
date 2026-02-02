import torch
import torch.nn as nn
import torch.nn.functional as F

def ssim(pred, target, C1=0.01**2, C2=0.03**2):
    """
    Structural Similarity Index (SSIM)
    Window-based implementation for image denoising.
    """
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)

    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) *
        (sigma_x + sigma_y + C2)
    )

    return ssim_map.mean()


class DenoisingLoss(nn.Module):
    """
    Combined L1 + SSIM loss for Monte Carlo denoising.
    """

    def __init__(self, ssim_weight=0.1):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1.0 - ssim(pred, target)

        return l1_loss + self.ssim_weight * ssim_loss

#to test    
if __name__ == "__main__":
    loss_fn = DenoisingLoss()
    x = torch.rand(1, 3, 200, 200)
    y = torch.rand(1, 3, 200, 200)
    print(loss_fn(x, y))
