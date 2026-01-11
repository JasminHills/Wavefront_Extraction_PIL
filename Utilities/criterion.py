import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import lasy 
import scipy
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt    
import numpy as np
import torch
import torch.nn.functional as F
from Utilities.datasetNw import psf_dataset, splitDataLoader, ToTensor, Normalize
    
# # Usage
# phase_map = np.load("phase_map.npy")
# laser.apply_optical_element(CustomPhaseMap(phase_map))

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class resize(object):
    def __call__(self, sample):
        phase, image = sample['phase'], sample['image']
        
        # image[0] = minmax(np.sqrt(image[0]))
        # image[1] = minmax(np.sqrt(image[1]))
#         print(np.shape(phase), np.shape(image))        
        phase = scipy.ndimage.zoom(phase, 256/np.shape(phase)[0], order=0)
#         inputs = torch.nan_to_num(inputs, nan=1e-3)
#         phase_0 = torch.nan_to_num(sample, nan=1e-3)
#         image = scipy.ndimage.zoom(image, 128/np.shape(image)[0], order=0)
        # image[1] = scipy.ndimage.zoom(image[1], 128/np.shape(image[1])[0], order=0)
       
        return {'phase': phase, 'image': image}
    
import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np


from lasy.laser import Laser
from lasy import profiles
from lasy.utils.zernike import zernike
from lasy.optical_elements import ParabolicMirror, ZernikeAberrations


def torch_minmax(x, axis=None, feature_range=(0, 1)):
    """
    Scales tensor `x` to a specified range [min, max], similar to sklearn or numpy.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    axis : int or tuple of ints, optional
        Axis or axes along which to compute min and max. If None, use the entire tensor.
    feature_range : tuple
        (min, max) desired range after scaling.

    Returns
    -------
    x_scaled : torch.Tensor
        Tensor scaled to `feature_range`.
    """
    min_val = x.min(dim=axis, keepdim=True).values if axis is not None else x.min()
    max_val = x.max(dim=axis, keepdim=True).values if axis is not None else x.max()

    scale = (feature_range[1] - feature_range[0]) / (max_val - min_val + 1e-8)
    x_scaled = (x - min_val) * scale + feature_range[0]
    return x_scaled

def pad_image_to_extent(image, img_extent, target_extent, pad_value=0):
    """
    Pad an image (in real-world coordinates) so it fits exactly within the target extent.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (H×W or H×W×C).
    img_extent : tuple of float
        (x_min, x_max, y_min, y_max) of the input image in real-world units.
    target_extent : tuple of float
        (x_min, x_max, y_min, y_max) of the desired output area in the same units.
    pad_value : int or float
        Value to fill outside the image (default = 0).
    
    Returns
    -------
    padded : np.ndarray
        Image padded to the target extent, preserving pixel scale.
    """
    ypix, xpix = image.shape[:2]

    # Compute pixel size from image extent (assumed uniform)
    dx = (img_extent[1] - img_extent[0]) / xpix
    dy = (img_extent[3] - img_extent[2]) / ypix

    # Compute target image shape in pixels
    target_nx = int(round((target_extent[1] - target_extent[0]) / dx))
    target_ny = int(round((target_extent[3] - target_extent[2]) / dy))

    # Compute offset of input image within target in pixels
    x_offset = int(round((img_extent[0] - target_extent[0]) / dx))
    y_offset = int(round((img_extent[2] - target_extent[2]) / dy))

    # Create padded array
    if image.ndim == 3:
        padded = np.full((target_ny, target_nx, image.shape[2]), pad_value, dtype=image.dtype)
    else:
        padded = np.full((target_ny, target_nx), pad_value, dtype=image.dtype)

    # Determine overlap regions (clip edges safely)
    y1_t = max(0, y_offset)
    y2_t = min(target_ny, y_offset + ypix)
    x1_t = max(0, x_offset)
    x2_t = min(target_nx, x_offset + xpix)

    y1_i = max(0, -y_offset)
    y2_i = y1_i + (y2_t - y1_t)
    x1_i = max(0, -x_offset)
    x2_i = x1_i + (x2_t - x1_t)

    # Insert image into padded array
    padded[y1_t:y2_t, x1_t:x2_t] = image[y1_i:y2_i, x1_i:x2_i]

    return padded

import torch

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

