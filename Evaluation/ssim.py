import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_ssim_numpy(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute SSIM between two 2D NumPy arrays.

    Parameters:
        img1, img2 (np.ndarray): Input images of shape [H, W]

    Returns:
        float: SSIM value between -1 and 1 (1 = perfect match)
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape")
    
    if img1.shape[0] != 2000 or img1.shape[1] != 2000:
        # print("Warning: images are not 2000x2000, function still works")
    
    # Ensure float type
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Compute SSIM
    ssim_value = ssim(img1, img2, data_range=img2.max() - img2.min())
    return ssim_value