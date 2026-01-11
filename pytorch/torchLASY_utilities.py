import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from lasy.laser import Laser, Grid
from lasy import profiles
from lasy.profiles import Profile

def getAmp(laser_obj, device='cpu'):
    """
    Convert a LASY Laser object into an amplitude tensor.
    Returns initial_amp : torch.Tensor [1, 1, H, W] as torch tensor on `device`.
    """
    temporal_field = laser_obj.grid.get_temporal_field()
    i_slice = temporal_field.shape[-1] // 2
    E0 = temporal_field[:, :, i_slice]  # [H, W]

    initial_amp = torch.tensor(
        abs(E0), dtype=torch.float32, device=device
    ).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    return initial_amp
    
def showxy(laser, **kw):
    """
    Show a 2D image of the laser amplitude.

    Parameters
    ----------
    **kw: additional arguments to be passed to matplotlib's imshow command
    """
    temporal_field = laser.grid.get_temporal_field()
    i_slice = int(temporal_field.shape[-1] // 2)
    E = temporal_field[:, :, i_slice]
    extent = [
        laser.grid.lo[1],
        laser.grid.hi[1],
        laser.grid.lo[0],
        laser.grid.hi[0],
    ]

    plt.imshow(abs(E), extent=extent, aspect="auto", origin="lower", **kw)
    cb = plt.colorbar()
    cb.set_label("$|E_{envelope}|$ (V/m) ", fontsize=12)
    plt.xlabel("y (m)", fontsize=12)
    plt.ylabel("x (m)", fontsize=12)
    plt.show()
    return abs(E)


def pad_image_stack_to_extent_torch(images, img_extent, target_extent, pad_value=0.0):
    """
    Pad a stack of images (in real-world coordinates) so each fits within the target extent.
    Works with PyTorch tensors.

    Parameters
    ----------
    images : torch.Tensor
        Input images, shape [B, H, W] or [B, C, H, W].
    img_extent : tuple of float
        (x_min, x_max, y_min, y_max) of the input image in real-world units.
    target_extent : tuple of float
        (x_min, x_max, y_min, y_max) of the desired output area.
    pad_value : float
        Value to fill outside the image.

    Returns
    -------
    padded : torch.Tensor
        Padded images, shape [B, target_ny, target_nx] or [B, C, target_ny, target_nx].
    """
    device = images.device
    dtype = images.dtype

    if images.ndim == 3:
        B, H, W = images.shape
        C = None
    elif images.ndim == 4:
        B, C, H, W = images.shape
    else:
        raise ValueError("Input must be [B,H,W] or [B,C,H,W]")

    # Pixel size
    dx = (img_extent[1] - img_extent[0]) / W
    dy = (img_extent[3] - img_extent[2]) / H

    # Target shape
    target_nx = int(round((target_extent[1] - target_extent[0]) / dx))
    target_ny = int(round((target_extent[3] - target_extent[2]) / dy))

    # Offsets
    x_offset = int(round((img_extent[0] - target_extent[0]) / dx))
    y_offset = int(round((img_extent[2] - target_extent[2]) / dy))

    # Create padded tensor
    if C is None:
        padded = torch.full((B, target_ny, target_nx), pad_value, dtype=dtype, device=device)
    else:
        padded = torch.full((B, C, target_ny, target_nx), pad_value, dtype=dtype, device=device)

    # Compute overlapping indices
    y1_t = max(0, y_offset)
    y2_t = min(target_ny, y_offset + H)
    x1_t = max(0, x_offset)
    x2_t = min(target_nx, x_offset + W)

    y1_i = max(0, -y_offset)
    y2_i = y1_i + (y2_t - y1_t)
    x1_i = max(0, -x_offset)
    x2_i = x1_i + (x2_t - x1_t)

    # Insert images
    if C is None:
        padded[:, y1_t:y2_t, x1_t:x2_t] = images[:, y1_i:y2_i, x1_i:x2_i]
    else:
        # images shape [B, C, H, W]
        padded[:, :, y1_t:y2_t, x1_t:x2_t] = images[:, :, y1_i:y2_i, x1_i:x2_i]

    return padded


def pad_image_to_extent_torch(image, img_extent, target_extent, pad_value=0.0):
    """
    Pad an image (in real-world coordinates) so it fits exactly within the target extent.
    Works with PyTorch tensors.

    Parameters
    ----------
    image : torch.Tensor
        Input image [H, W] or [H, W, C].
    img_extent : tuple of float
        (x_min, x_max, y_min, y_max) of the input image in real-world units.
    target_extent : tuple of float
        (x_min, x_max, y_min, y_max) of the desired output area.
    pad_value : float
        Value to fill outside the image.

    Returns
    -------
    padded : torch.Tensor
        Padded image [target_ny, target_nx] or [target_ny, target_nx, C].
    """
    device = image.device
    dtype = image.dtype
    ypix, xpix = image.shape[:2]

    # Pixel size
    dx = (img_extent[1] - img_extent[0]) / xpix
    dy = (img_extent[3] - img_extent[2]) / ypix

    # Target shape
    target_nx = int(round((target_extent[1] - target_extent[0]) / dx))
    target_ny = int(round((target_extent[3] - target_extent[2]) / dy))

    # Offsets in pixels
    x_offset = int(round((img_extent[0] - target_extent[0]) / dx))
    y_offset = int(round((img_extent[2] - target_extent[2]) / dy))

    # Create padded tensor
    if image.ndim == 3:
        C = image.shape[2]
        padded = torch.full((target_ny, target_nx, C), pad_value, dtype=dtype, device=device)
    else:
        padded = torch.full((target_ny, target_nx), pad_value, dtype=dtype, device=device)

    # Compute overlapping indices
    y1_t = max(0, y_offset)
    y2_t = min(target_ny, y_offset + ypix)
    x1_t = max(0, x_offset)
    x2_t = min(target_nx, x_offset + xpix)

    y1_i = max(0, -y_offset)
    y2_i = y1_i + (y2_t - y1_t)
    x1_i = max(0, -x_offset)
    x2_i = x1_i + (x2_t - x1_t)

    # Insert original image
    padded[y1_t:y2_t, x1_t:x2_t] = image[y1_i:y2_i, x1_i:x2_i]

    return padded


