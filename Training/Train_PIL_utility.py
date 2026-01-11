# Maths -ey imports
import scipy 
import math
from scipy.ndimage import *
import numpy as np

def normalize_to(batch, min_batch, max_batch, min_val=0.0, max_val=1.0, val=1.0, eps=1e-8):

        # Normalize the entire batch to [0, 1]
        normed = (batch - min_batch) / (max_batch - min_batch + eps)

        # Scale to [min_val, max_val]
        normed = normed * (max_val - min_val) + min_val

        return normed

class resize(object): # Resize class to be passed into dataset loader
    
    def __call__(self, sample):
        phase, image = sample['phase'], sample['image']
        phase = scipy.ndimage.zoom(phase, 256/np.shape(phase)[0], order=0)
        return {'phase': phase, 'image': image}
    


def unstack(a, axis=0):
    return np.moveaxis(a, axis, 0)

def cosine_decay(epoch, total_epochs, start=0.9, end=0.1): #1-0.3
    """Cosine decay from start to end over total_epochs."""
    cos_inner = (math.pi * epoch) / total_epochs
    if epoch>total_epochs:
        return end
    else:
        return end + (start - end) * (1 + math.cos(cos_inner)) / 2
    
def min_max_ext(dataset, dataset_size):
    p_min=[]
    p_max=[]
    i_min=[]
    i_max=[]
    for di in range(dataset_size):
        d=dataset.__getitem__(di)
        p_max.append(np.max(d['phase']))
        p_min.append(np.min(d['phase']))
        i_max.append(float((d['image'][0]).max().numpy()))
        i_min.append(float((d['image'][0]).min().numpy()))
    return np.max(i_max), np.min(i_min), np.min(p_min), np.max(p_max)