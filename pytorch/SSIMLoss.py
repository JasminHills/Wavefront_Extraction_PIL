import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss for PyTorch.
    Returns 1 - SSIM(pred, target), so higher SSIM -> lower loss.
    """
    def __init__(self, data_range=1.0, kernel_size=11, sigma=1.5, k1=0.01, k2=0.03, reduction='mean'):
        super().__init__()
        self.data_range = data_range
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.reduction = reduction

        # Create gaussian kernel
        self.channel = 1
        self.gaussian_kernel = self.create_gaussian_kernel(kernel_size, sigma)

    def create_gaussian_kernel(self, kernel_size, sigma):
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        g = torch.exp(-(coords**2)/(2*sigma**2))
        g = g / g.sum()
        kernel = g[:, None] * g[None, :]
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape [1,1,K,K]
        return kernel

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (N, C, H, W) in [0, data_range]
        Returns: scalar differentiable SSIM loss
        """
        device = pred.device
        C = pred.size(1)
        kernel = self.gaussian_kernel.to(device).repeat(C, 1, 1, 1)

        mu_x = F.conv2d(pred, kernel, padding=self.kernel_size//2, groups=C)
        mu_y = F.conv2d(target, kernel, padding=self.kernel_size//2, groups=C)

        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(pred * pred, kernel, padding=self.kernel_size//2, groups=C) - mu_x2
        sigma_y2 = F.conv2d(target * target, kernel, padding=self.kernel_size//2, groups=C) - mu_y2
        sigma_xy = F.conv2d(pred * target, kernel, padding=self.kernel_size//2, groups=C) - mu_xy

        C1 = (self.k1 * self.data_range) ** 2
        C2 = (self.k2 * self.data_range) ** 2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

        if self.reduction == 'mean':
            return 1 - ssim_map.mean()
        elif self.reduction == 'sum':
            return 1 - ssim_map.sum()
        else:  # no reduction
            return 1 - ssim_map
