
import scipy 
import torch
import torchLASY_V4

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
# import phase aplly

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.transform import resize

def pad_and_resize(self, image, target_size):

        return F.interpolate(image, size=target_size, mode='nearest')

class PhaseMapOptic:
    """
    LASY optical element applying a user-defined spatial phase φ(x,y).
    Works with LASY's spectral-domain optic interface.
    """

    def __init__(self, x_phase, y_phase, phase_xy):
        # Build interpolator for arbitrary x,y requested by LASY
        self.phase_interp = RegularGridInterpolator(
            (x_phase, y_phase),
            phase_xy,
            bounds_error=False,
            fill_value=0.0
        )
        self.phase_xy=phase_xy

    # LASY requires this
    def amplitude_multiplier(self, x, y, omega):
        """No amplitude change: return 1 everywhere"""
         # LASY provides x,y as 1D arrays of equal shape
        pts = np.stack([x, y], axis=-1)
        phase_vals = self.phase_interp(pts)
        return  np.exp(1j * phase_vals)#np.ones_like(x, dtype=float)

    # LASY requires this
    def phase_multiplier(self, x, y, omega):
        """
        Return exp(i * φ(x,y)) evaluated at the (x,y) points
        where LASY wants the phase mask applied.
        """
        # LASY provides x,y as 1D arrays of equal shape
        pts = np.stack([x, y], axis=-1)
        phase_vals = self.phase_interp(pts)
        plt.imshow(phase_vals)
        # print(phase_vals)
        plt.show()
        return np.exp(1j * phase_vals)

class PhaseMapElement:
    """
    Optical element applying a user-defined spatial phase φ(x,y).
    """

    def __init__(self, x_phase, y_phase, phase_xy, apply_to="all"):
        from scipy.interpolate import RegularGridInterpolator
        self.apply_to = apply_to
        self.interpolator = RegularGridInterpolator(
            (x_phase, y_phase),
            phase_xy,
            bounds_error=False,
            fill_value=0.0
        )

    def apply(self, laser):
        """
        Apply the phase to a LASY Laser object
        """

        # ✔ Correct LASY grid access
        x = laser.grid.axes[0]    # 1D x-axis
        y = laser.grid.axes[1]    # 1D y-axis

        # Meshgrid for interpolation
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Interpolate phase
        pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
        phase_interp = self.interpolator(pts).reshape(X.shape)

        # Phase factor
        phase_factor = np.exp(1j * phase_interp)

        # ✔ Apply to actual LASY field components
        if self.apply_to in ("all", "Ex"):
            laser.Ex *= phase_factor
        if self.apply_to in ("all", "Ey"):
            laser.Ey *= phase_factor
        if self.apply_to in ("all", "Ez"):
            laser.Ez *= phase_factor
        if self.apply_to == "transverse":
            laser.Ex *= phase_factor
            laser.Ey *= phase_factor


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


def pad_and_resize(image, target_size):
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 3:
            image = image.unsqueeze(1)
        return F.interpolate(image, size=target_size, mode='nearest')

def propagator(phase, f0=9):
    
    criterion=torchLASY_V4.PhysicsInformedLoss(f0, plot=False, device="cpu")
    I_pred=criterion.forward_no_loss(torch.from_numpy(phase))
    laser_obj=criterion.laser
    return I_pred, laser_obj

class resize(object):
    def __call__(self, sample):
        phase, image = sample['phase'], sample['image']

        phase = scipy.ndimage.zoom(phase, 256/np.shape(phase)[0], order=0)

        return {'phase': phase, 'image': image}

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    
def showxy(laser, **kw):
    """
    Show a 2D image of the laser amplitude.

    Parameters
    ----------
    **kw: additional arguments to be passed to matplotlib's imshow command
    """
    temporal_field = laser.grid.get_temporal_field()
    i_slice = int(temporal_field.shape[-1] // 2)
    E = temporal_field[:,:,  i_slice]
    extent = [
            laser.grid.lo[1],
            laser.grid.hi[1],
            laser.grid.lo[0],
            laser.grid.hi[0],
        ]
#     extent = [
#             -400,
#             400,
#             -400,
#             400,
#         ]

    plt.imshow(np.abs(E)**2, extent=extent, aspect="auto", origin="lower", **kw)
    cb = plt.colorbar()
    cb.set_label("$|E_{envelope}|$ (V/m) ", fontsize=12)
    plt.xlabel("y ($\mu m$)", fontsize=12)
    plt.ylabel("x ($\mu m$)", fontsize=12)
    
    

def normalize_to(batch, min_batch, max_batch, min_val=0.0, max_val=1.0, val=1.0, eps=1e-8):
#         if batch.ndim == 3:
#             batch = batch.unsqueeze(1)  # [B, 1, H, W]

#         # Compute batch-level min and max across all pixels in all images
#         min_batch = batch.min()
#         max_batch = batch.max()

        # Normalize the entire batch to [0, 1]
        normed = (batch - min_batch) / (max_batch - min_batch + eps)

        # Scale to [min_val, max_val]
        normed = normed * (max_val - min_val) + min_val

        return normed
    
def plotter(img, vmin=0, vmax=0):
        
            f, axarr = plt.subplots(1, 1, figsize=(7, 7))
            if vmin!=0 and vmax!=0:
                im2 = axarr.imshow(img , cmap=plt.cm.viridis, extent=[-600, 600, -600, 600], vmin=vmin, vmax=vmax)
            else:
                im2 = axarr.imshow(img , cmap=plt.cm.viridis, extent=[-600, 600, -600, 600])
            plt.xlim(-400, 400)
            plt.ylim(-400, 400)

            c=plt.colorbar(im2, ax= axarr, fraction=0.04)
            c.ax.tick_params(labelsize=15)
            c.ax.set_ylabel('Normalised Intensity (Arb.)', fontsize=15)
            plt.xlabel('x ($\mu m$)', fontsize=15)
            plt.ylabel('y ($\mu m$)', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.show()