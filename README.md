# Wavefront_Extraction_PIL
Trained Unet model to reconstruct wavefront abberations from focal spot images for a known geometry. Propagation dynamics were incorporated in the loss function through the introduction of a PyTorch laser propagation module (based on the lasy angular spectrum propagator).

**Includes:**
Generation of initial training set. Pairs of wavefront abberations and resultant focal spot images. 
PIL loss implementation, i.e. Takes model predicted wavefronts propagates these to focus and compares them to the original intensity profile.
Training of the model + evaluation. 
NAS for different Unet architectures + amount of dropout + (optional hyperparameters)
Evaluation of image similarity using SSIM and RMSE

## How to run
Jupyter notebook for testing different parts. Also submission scripts for HPC included. 

## Structure
**Bash_Scripts:** Submission scripts
**Generation:** Generates training data over 9mm of propagation
**NAS:** Optimising model/hyperparameters
**pytorch:** Utilities for loss function and pytorch laser propagation. In addition to different model architectures
**Training:** Training functions 
**Evaluation:** Evaluates model + image similarity

**Dependencies required:**
PIL, lasy, aotools, scipy, astropy, numpy, matplotlib, torch

## What I’d improve next
Training with SSIM in the loss function. Add in additional noise imperfections in immediate wavefront. Train the final model on experimental data - i.e. try to recreate experimental data during fine tuning of the model. 
