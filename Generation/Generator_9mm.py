# utilities
import os
import sys

# plotting and maths 
import numpy as np
import matplotlib.pyplot as plt
import scipy 

sys.path.insert(0, '../pytorch/')

from pytorch.PIL import *

#Useful definitions - image manipulation
import scipy
import aotools # for defining phase
from astropy.io import fits # for saving HDU outputs 
import warnings

global wavelength
wavelength =800*1e-9 

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

    plt.imshow(abs(E), extent=extent, aspect="auto", origin="lower", **kw)
    cb = plt.colorbar()
    cb.set_label("$|E_{envelope}|$ (V/m) ", fontsize=12)
    plt.xlabel("y (m)", fontsize=12)
    plt.ylabel("x (m)", fontsize=12)
    plt.show()
    
def recenter(arr, am):
    image=arr
    Nmax, Mmax = np.unravel_index(scipy.ndimage.median_filter(image,3).argmax(), image.shape)
    # print(Nmax, Mmax)
    # print(np.shape(image))
    return arr[Nmax-am:Nmax+am, Mmax-am:Mmax+am] 

def ZernikeGen(n_psfs, n_zernike, seed):
    np.random.seed(seed=seed)
    i_zernike = np.arange(2, n_zernike + 2)      # Zernike polynomial indices (piston excluded)
    o_zernike= []             # Zernike polynomial radial Order, see J. Noll paper :
    
    for i in range(1,n_zernike):      # "Zernike polynomials and atmospheric turbulence", 1975
        for j in range(i+1):
            if len(o_zernike) < n_zernike:
                o_zernike.append(i)

    # Generate randomly Zernike coefficient. By dividing the value

    # by its radial order we produce a distribution following
    # the expected 1/f^-2 law.
    scales=[]
    c_zernikeFinal=np.zeros((n_psfs, n_zernike))
    for j in range(n_psfs):
        c_zernike = np.random.random(n_zernike)-0.5
        terms=int(np.random.randint(30, 500))

        
        for i in range(terms):#n_zernike):
            c_zernikeFinal[j, i] = c_zernike[i] / o_zernike[i]
            
    c_zernikeFinal= np.array([c_zernikeFinal[k, :] / np.abs(c_zernikeFinal[k, :]).sum()* wavelength*(10**9)for k in range(1*n_psfs)])
    return c_zernikeFinal
    

def LaserProp(i, c, f0, outFile):
    outFIl3=outFile+'256'
    os.system('mkdir -p '+outFIl3)
    
    pil=PhysicsInformedLoss(int(f0),phse_ext=200, plot=False, device='cpu') # initialising laser using PIL module, defined diameter and focus distance

    WFE=np.random.randint(1, 500) #Randomised wavefront amplitudes
    cc=c*WFE*1e-3 
    pup=200
    pupil_coords=(0, 0, pup*1e-6) #defining pupil

    zernike_amplitudes = {index: value for index, value in enumerate(cc)} #Generate amplitudes for different coefficients
    phs=aotools.functions.zernike.phaseFromZernikes(cc, int(pup), norm='noll') #Generating phase from the coefficients
    norm=pil.add_phase_ext(torch.from_numpy(phs)).cpu().detach().numpy() # add phase to laser defined previously and propagate to focus 

    outfile = outFIl3+"/psf_" + str(i) + ".fits" # Define output folder 
    aberrations_in=scipy.ndimage.zoom(phs, 256/np.shape(phs)[0], order=0) # resize phase input

    # Save all the inputs, i.e. the phase coefficient values, the phase map, and the intensity of the spot at focus 
    hdu_primary = fits.PrimaryHDU(cc.astype(np.float32))
    hdu_phase = fits.ImageHDU(aberrations_in.astype(np.float32), name='PHASE')
    norm=scipy.ndimage.zoom(norm, 256/np.shape(norm)[0], order=0)
    hdu_In = fits.ImageHDU(norm.astype(np.float32), name='INFOCUS')
    hdu_Out = fits.ImageHDU(norm.astype(np.float32), name='OUTFOCUS')
    hdu = fits.HDUList([hdu_primary, hdu_phase, hdu_In, hdu_Out])
    hdu.writeto(outfile, overwrite=True)
    

if __name__ == "__main__":
    # total arguments
    # print(sys.argv)
    nm, filnm, n_psfs, n_zernike, f0, seed = sys.argv
    c_z=ZernikeGen(int(n_psfs), int(n_zernike), int(seed))
    for i, c in enumerate(c_z):
        LaserProp(i, c, f0, filnm)
        
    
    

def runner(nm, filnm, n_psfs, n_zernike, f0, seed ): # name of file, name of output folder, number of data points, number of coefficients, focal distance, random seed
    c_z=ZernikeGen(int(n_psfs), int(n_zernike), int(seed))
    for i, c in enumerate(c_z):
        LaserProp(i, c, f0, filnm)
        
    
    