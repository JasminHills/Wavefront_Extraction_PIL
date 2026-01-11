import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from lasy.laser import Laser, Grid
from lasy import profiles
from lasy.profiles import Profile

import matplotlib.pyplot as plt
from Utilities.datasetNw import psf_dataset, splitDataLoader, ToTensor, Normalize
import numpy as np
from pytorch.torchLASY_utilities import *

class TorchProfile(Profile):
    def __init__(self, field_np_xy, x_axis, y_axis, wavelength, pol=(1,0)):
        """
        field_np_xy: 2D complex numpy array of envelope at t = 0
        x_axis, y_axis: 1D arrays of x and y coordinates
        """
        self.field = field_np_xy
        self.x = x_axis
        self.y = y_axis
        self.wavelength = wavelength
        self.pol = pol

        # Lasy expects some dtype, set this
        self.dtype = self.field.dtype

    def evaluate(self, x, y, t):
        """
        Return the complex envelope at given x, y, t.
        For simplicity, assume *no time-dependence* (flat in t).
        """
        # You need to interpolate your `self.field` (2D) to the requested (x, y)
        # Here use simple bilinear / nearest interpolation
        # For t, we ignore it and just return the same slice always.

        # Example using numpy.interp or scipy (simplest hack below):
        xi = np.interp(x, self.x, np.arange(self.x.size))
        yi = np.interp(y, self.y, np.arange(self.y.size))
        xi = np.round(xi).astype(int)
        yi = np.round(yi).astype(int)
        val = self.field[yi, xi]   # careful with ordering
        return val * np.ones_like(t, dtype=self.dtype)


class TorchLaser:
    def __init__(self, field, wavelength, extent, device="cpu"):
        """
        Parameters
        ----------
        field : torch.complex64 tensor [H, W]
            Complex electric field at the current plane.
        wavelength : float
            Laser wavelength in meters.
        extent : tuple (x_min, x_max, y_min, y_max)
            Real-world spatial extent of the field.
        device : str
            torch device ('cpu' or 'cuda').
        """
        # store an initial reference field; methods below will not mutate this in-place
        self.field = field.to(torch.complex64).to(device)
        self.wavelength = wavelength
        self.extent = extent
        self.device = device
        

        H, W = field.shape
        self.H = H
        self.W = W
        self.dx = (extent[1] - extent[0]) / W
        self.dy = (extent[3] - extent[2]) / H

    @classmethod
    def from_lasy(cls, lasy_obj, device="cpu", wavelength=800e-9):
        field_np = lasy_obj.grid.get_temporal_field()
        i_slice = field_np.shape[-1] // 2
        E0 = field_np[:, :, i_slice]  # complex numpy array
        # build complex torch tensor
        field_torch = torch.tensor(np.real(E0), dtype=torch.float32) + 1j * torch.tensor(
            np.imag(E0), dtype=torch.float32
        )

        extent = (lasy_obj.grid.lo[0], lasy_obj.grid.hi[0],
                  lasy_obj.grid.lo[1], lasy_obj.grid.hi[1])

        return cls(field_torch, wavelength=wavelength, extent=extent, device=device)
    

    def to_lasy(self):
        # convert torch â†’ numpy
        field_np = self.field.detach().cpu().numpy()
        # build coordinate axes from extent
        x_min, x_max, y_min, y_max = self.extent
        x = np.linspace(x_min, x_max, self.W)
        y = np.linspace(y_min, y_max, self.H)

        profile = TorchProfile(field_np, x, y, self.wavelength)

        # define grid parameters for Laser
        dim = "xyt"
        lo = (x_min, y_min, -1e-15)
        hi = (x_max, y_max, 1e-15)
        npoints = (self.W, self.H, 1)

        lasy_laser = Laser(dim, lo, hi, npoints, profile)

        # convert torch â†’ nump
        return lasy_laser



    def add_phase(self, base_field, phase_map):
        """
        Add a phase (in radians) to a given field (pure function).
        base_field : tensor [H,W] or [B,H,W] or [B,1,H,W] (complex)
        phase_map  : tensor broadcastable to base_field real dims (float)
        Returns the complex field with phase applied (tensor, not stored).
        """
        return base_field * torch.exp(1j * phase_map.to(self.device))

    def propagate(self, field, distance_m: float):
        """
        Fresnel propagation using angular spectrum method (torch-native).
        field: complex tensor [H,W] or [B,H,W] or [B,1,H,W]
        Returns propagated field tensor (does NOT mutate self.field).
        """
        field = field.to(self.device).to(torch.complex64)

        # normalize shapes
        if field.ndim == 2:
            field = field.unsqueeze(0)  # [1, H, W]

        if field.ndim == 3:  # [B,H,W]
            B, H, W = field.shape
        elif field.ndim == 4:  # [B,C,H,W] -> assume C==1
            B, C, H, W = field.shape
            if C == 1:
                field = field.squeeze(1)  # to [B,H,W]
            else:
                raise ValueError("propagate: unsupported channel dim >1")
        else:
            raise ValueError("propagate: unsupported field ndim")

        x_min, x_max, y_min, y_max = self.extent
        dx = (x_max - x_min) / W
        dy = (y_max - y_min) / H

        fx = torch.fft.fftfreq(W, dx).to(self.device)
        fy = torch.fft.fftfreq(H, dy).to(self.device)
        FX, FY = torch.meshgrid(fx, fy, indexing="xy")  # FX, FY shaped [H, W]
        FX2FY2 = FX**2 + FY**2

        # avoid negative inside sqrt for evanescent components:
        arg = (1.0 / (self.wavelength**2)) - FX2FY2
        arg = torch.clamp(arg, min=0.0)

        H_prop = torch.exp(1j * 2 * torch.pi * distance_m * torch.sqrt(arg).to(self.device))  # [H,W]

        # FFT
        field_fft = torch.fft.fft2(field)  # works per-batch
        field_fft_prop = field_fft * H_prop  # broadcasting over batch
        field_prop = torch.fft.ifft2(field_fft_prop)
        del field_fft, field_fft_prop
        torch.cuda.empty_cache()
        # return with same shape as input: if input was [H,W] return [H,W], else [B,H,W]
        if field_prop.shape[0] == 1:
            return field_prop.squeeze(0)
        return field_prop
    
    def propagate_in_chunks(self, field_batch, distance_m, chunk_size=1):
        # field_batch shape: [B, H, W] or [B, C, H, W] depending on your code
        results = []
        B = field_batch.shape[0]
        for i in range(0, B, chunk_size):
            chunk = field_batch[i:i+chunk_size]
            results.append(self.propagate(chunk, distance_m))
            # free GPU memory that chunk used
            del chunk
            torch.cuda.empty_cache()
        return torch.cat(results, dim=0)
    def propagate_in_chunks_new(self, field_batch, distance_m, chunk_size=1):
        """
        Propagate a batch of fields in chunks to save memory.
        Accepts field_batch with shape either [H, W] (single sample) or [B, H, W].
        Always concatenates chunk outputs along batch dim and returns:
          - [B, H, W] if input was batched
          - [H, W] if input was a single sample
        """
        single_input = (field_batch.dim() == 2)  # [H, W]
        if single_input:
            field_batch = field_batch.unsqueeze(0)  # make [1, H, W]

        B = field_batch.shape[0]
        results = []
        device = field_batch.device
        dtype = field_batch.dtype

        for i in range(0, B, chunk_size):
            chunk = field_batch[i:i+chunk_size]            # shape: [C, H, W] or [chunk, H, W]
            # ensure chunk on correct device/dtype (usually already is)
            if chunk.device != device or chunk.dtype != dtype:
                chunk = chunk.to(device=device, dtype=dtype)

            out_chunk = self.propagate(chunk, distance_m)  # CALL existing propagate
            # normalize output to have batch dim
            if out_chunk.dim() == 2:
                out_chunk = out_chunk.unsqueeze(0)   # [1, H, W]
            elif out_chunk.dim() == 3:
                # if returned shape [H,W] accidentally (just in case) handle above; else assume [k,H,W]
                pass
            else:
                raise RuntimeError(f"Unexpected output dim from propagate: {out_chunk.dim()}")

            # sanity check: batch size matches chunk length or is 1 (propagate may squeeze)
            # If propagate returned a single sample but the chunk had >1 samples, that's likely a bug in propagate.
            if out_chunk.shape[0] not in (1, chunk.shape[0]):
                # best effort handling: try to expand/squeeze if shapes align
                if out_chunk.shape == chunk.shape[1:]:
                    out_chunk = out_chunk.unsqueeze(0)
                else:
                    raise RuntimeError(
                        f"propagate returned batch size {out_chunk.shape[0]} but input chunk size {chunk.shape[0]}"
                    )

            results.append(out_chunk)
            # free memory that chunk used
            del chunk, out_chunk
            torch.cuda.empty_cache()

        # concat results along batch dim
        result = torch.cat(results, dim=0)  # shape: [B, H, W]
        if single_input:
            return result[0]   # return [H, W] to preserve original API
        return result


    def intensity(self, field=None):
        """Return intensity |E|^2. If field is None, use stored reference field (detached).
        This does not mutate internal state.
        """
        if field is None:
            f = self.field.detach()
        else:
            f = field
        return torch.abs(f) ** 2

    def phase(self, field=None):
        """Return phase (angle)."""
        if field is None:
            f = self.field
        else:
            f = field
        return torch.angle(f)