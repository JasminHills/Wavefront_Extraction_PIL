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
from pytorch.PyTorchLaser import TorchLaser, TorchProfile


class PhysicsInformedLoss(nn.Module): # changed phse_ext to 200
    def __init__(self, f0, laser_params=None, device='cpu', plot=True, phse_ext=200, int_max=1e23, int_min=0, phase_max=50, phase_min=-50):
        super().__init__()
        self.device = device
        self.plot=plot
        self.f0 = f0
        self.mse =  nn.MSELoss() # defining MSE loss function
        lasy_laser = self.laserGen(phse_ext=200) # Generate a lasy laser object
        self.initial_amp = getAmp(lasy_laser) #Get initial intensity amplitude from lasy laser object
        
        # create TorchLaser from LASY object (stores reference field but methods below are pure)
        self.laser = TorchLaser.from_lasy(lasy_obj=lasy_laser, device=device)

        # extent of laser grid defined and special normalisation (if changing)
        self.int_max=int_max
        self.int_min=int_min
        self.phase_max=phase_max
        self.phase_min=phase_min

        self.i=0

    def laserGen(self,phse_ext=200 ):
        wavelength = 800e-9  # Laser wavelength in meters
        polarization = (1, 0)  # Linearly polarized in the x direction
        energy = 0.4  # Energy of the laser pulse in joules
        spot_size = 30e-6  # Waist of the laser pulse in meters
        WndowSz = 500e-6
        tau = 42e-15  # Pulse duration of the laser in seconds
        t_peak = 0.0  # Location of the peak of the laser pulse in time

        self.ls_extent = [-WndowSz, WndowSz, -WndowSz, WndowSz]
        self.phse_extent = [-phse_ext * 1e-6, phse_ext * 1e-6, -phse_ext * 1e-6, phse_ext * 1e-6]

        ls_profile = profiles.GaussianProfile(w0=spot_size, wavelength=wavelength, tau=tau, t_peak=t_peak, laser_energy=0.4, pol=polarization)
        lo = (-WndowSz, -WndowSz, -2 * tau)  # Lower bounds of the simulation box
        hi = (WndowSz, WndowSz, 2 * tau)  # Upper bounds of the simulation box

        num_points = (2000, 2000, 2)  # Number of points in each dimension

        self.ls = Laser('xyt', lo, hi, num_points, ls_profile) #Create lasy object with profile and grid parameters
        self.ls.propagate(-self.f0 * 1e-3) # Propagate spot from focus
        return self.ls

    def pad_and_resize(self, image, target_size):
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 3:
            image = image.unsqueeze(1)
        return F.interpolate(image, size=target_size, mode='nearest')

    def normalize_to(self, batch, min_batch, max_batch, min_val=0.0, max_val=1.0, val=1.0, eps=1e-8):
        if batch.ndim == 3:
            batch = batch.unsqueeze(1)  # [B, 1, H, W]

        normed = (batch - min_batch) / (max_batch - min_batch + eps)

        # Scale to [min_val, max_val]
        normed = normed * (max_val - min_val) + min_val

        return normed
    
    def normalize(self, batch, min_val=0.0,max_val=1.0, val=1.0, eps=1e-8):
            if batch.ndim == 3:
                batch = batch.unsqueeze(1)  # [B, 1, H, W]

            B = batch.size(0)

            # Compute per-image min/max
            min_per_image = batch.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
            max_per_image = batch.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)

            # Normalize to [0, 1]
            normed = (batch - min_per_image) / (max_per_image - min_per_image + eps)


            normed = normed * (max_val - min_val) + min_val

            return normed

    def add_phase_ext(self, phase_pred):
        """
        Compute physics-informed intensity loss + optional phase reconstruction loss.

        Args:
            phase_pred: [B, 1, H, W] predicted phase in radians
            I_measured: [B, 1, H, W] measured intensity
            phase_true: [B, 1, H, W] (optional) true phase, if available
            lambda_phase: float in [0,1], weight for phase loss term
        """
        self.i+=1
        # Resize amplitude and phase_pred to laser grid
        H, W = self.laser.H, self.laser.W
        
        phase_padded=pad_image_to_extent_torch(phase_pred, self.phse_extent, self.ls_extent, pad_value=0.0)
        phase_resized = self.pad_and_resize(phase_padded, (H, W))
        if self.plot==True:
            # print('_____________________')    
            # print('phase_resized')
    #         print(self.normalize(torch.log(I_pred[0].unsqueeze(0))))
    #         print(I_pred[0].shape)
            plt.imshow(phase_resized[0].squeeze(0).detach().cpu().numpy())
            plt.show()
            # print('_____________________') 

        # Apply predicted phase
        base_field = self.laser.field.detach()
        new_field = self.laser.add_phase(base_field, phase_resized)

        # Propagate through the optical system
        field_prop = self.laser.propagate(new_field, self.f0 * 1e-3)

        # Compute predicted intensity
        I_pred = torch.abs(field_prop)** 2
        return I_pred
    
    def forward_no_loss(self, phase_pred):
        """
        Propagate laser object of a given predicted phase without computing loss.

        Args:
            phase_pred: [B, 1, H, W] predicted phase in radians
            
        """
        self.i+=1
        # Resize amplitude and phase_pred to laser grid
        H, W = self.laser.H, self.laser.W
        phase_resized = self.pad_and_resize(phase_pred, (H, W))

        # Apply predicted phase
        base_field = self.laser.field.detach()
        new_field = self.laser.add_phase(base_field, phase_resized)

        # Propagate through the optical system
        field_prop = self.laser.propagate_in_chunks_new(new_field, self.f0 * 1e-3, chunk_size=1)

        # Compute predicted intensity
        I_pred = torch.abs(field_prop)
#         I_pred = F.interpolate(I_pred.unsqueeze(1), size=I_measured.shape[-2:], mode="nearest")
            
        if self.plot==True:
            norm = (I_pred - torch.min(I_pred)) / (torch.max(I_pred) - torch.min(I_pred))
            f, axarr = plt.subplots(1, 1, figsize=(7, 7))
            im2=axarr.imshow(I_pred.squeeze(0).squeeze(0).detach().cpu().numpy()**2, cmap=plt.cm.viridis, extent=[-600, 600, -600, 600])
            c=plt.colorbar(im2, ax= axarr, fraction=0.04)
            c.ax.tick_params(labelsize=15)
            c.ax.set_ylabel('Normalised Intensity (Arb.)', fontsize=15)
            plt.xlabel('x ($\mu m$)', fontsize=15)
            plt.ylabel('y ($\mu m$)', fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
#             plt.colorbar()
            plt.show()

        return I_pred
    
    
    def forward(self, phase_pred, I_measured, phase_true=None, lambda_phase=0.1):
        """
        Compute physics-informed intensity loss + optional phase reconstruction loss.

        Args:
            phase_pred: [B, 1, H, W] predicted phase in radians
            I_measured: [B, 1, H, W] measured intensity
            phase_true: [B, 1, H, W] (optional) true phase, if available
            lambda_phase: float in [0,1], weight for phase loss term
        """
        self.i+=1
        # Resize amplitude and phase_pred to laser grid
        H, W = self.laser.H, self.laser.W
        phase_resized = self.pad_and_resize(phase_pred, (H, W))

        # Apply predicted phase
        base_field = self.laser.field.detach()
        new_field = self.laser.add_phase(base_field, phase_resized)

        # Propagate through the optical system
        field_prop = self.laser.propagate_in_chunks_new(new_field, self.f0 * 1e-3, chunk_size=8)

        # Compute predicted intensity
        I_pred = torch.abs(field_prop) #** 2 #i uncommented the square here
        I_pred = F.interpolate(I_pred.unsqueeze(1), size=I_measured.shape[-2:], mode="nearest")
        
        # --- Physics-informed loss ---
        loss_intensity = self.mse(I_pred, torch.sqrt(I_measured.to(self.device)))/1e20  #Compare the two values - scale down loss to prevent blow up 
        loss_normalised_intensity = self.mse(self.normalize(I_pred), self.normalize(torch.sqrt(I_measured.to(self.device)))) #Compare the two values - scale down loss to prevent blow up 
    
        # --- Optional phase loss ---
        if phase_true is not None: # and lambda_phase > 0.0:
            # Resize target phase to match prediction
            phase_true_resized = self.pad_and_resize(phase_true, (H, W))

            # Normalize phases to [-pi, pi] â†’ [0,1] for stability
            phase_pred_norm = phase_resized #(torch.atan2(torch.sin(phase_resized), torch.cos(phase_resized)) + np.pi) / (2 * np.pi)
            phase_true_norm = phase_true_resized

            loss_phase = self.mse(phase_pred_norm, phase_true_norm) 
            if phase_pred_norm.max()>phase_true_norm.max():
                true_max=phase_pred_norm.max()
            else:
                true_max=phase_true_norm.max()
            
            if phase_pred_norm.min()>phase_true_norm.min():
                true_min=phase_true_norm.min()
            else:
                true_min=phase_pred_norm.min()
            
            loss_phase = self.mse(phase_pred_norm, phase_true_norm)
            # print('loss_phase', loss_phase)
            #loss_phase_v2=1-self.lossfnc_int(self.normalize_to(phase_pred_norm, self.phase_min, self.phase_max), self.normalize_to(phase_true_norm,  self.phase_min, self.phase_max))
            
        else:
            loss_phase = torch.tensor(0.0, device=self.device)
            
        # --- Combine losses ---
        total_loss =(1 - lambda_phase) * loss_intensity +lambda_phase*loss_phase # Set lambda_phase to 0 to nullify
        
        self.phse_loss=loss_phase
        self.int_loss=loss_intensity
        self.int_normalized_intensity=loss_normalised_intensity

        return total_loss