# -*- coding: utf-8 -*-
"""
Library of utilities for cFos analysis

@author: Dreosti Lab
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageSequence

# Utilities for analyzing cFos experiments

# Load NII stack
def load_nii(path):
 
    image = nib.load(path)
    image_size = image.shape
    image_type = image.get_data_dtype()
    image_data = image.get_data() + 32768 # offset from 16-bit signed
    image_data = np.transpose(image_data, (1, 0, 2)) # swap x/y
    
#    z_stack_avg = np.mean(image_data,axis=2)
#    z_stack_max = np.max(image_data,axis=2)
#    z_stack_std = np.std(image_data,axis=2)
    
#    plt.figure()
#    plt.imshow(z_stack_max)
    
#    plt.figure()
#    plt.imshow(z_stack_std)
    
 #   plt.figure()
 #   plt.imshow(z_stack_avg)
        
    return image_data

# Save NII stack
def save_nii(path, data, affine, header):
 
    new_img = nib.Nifti1Image(data, affine, header)
    nib.save(new_img, path)

# Load Mask Labels (TIFF)
def load_mask(path):
    
    tiff = Image.open(path)
    tiff.load()
    num_frames = tiff.n_frames
    images = np.zeros((tiff.height, tiff.width, num_frames), dtype=np.uint8) 
    for i, page in enumerate(ImageSequence.Iterator(tiff)):
        images[:,:,i] = np.reshape(page, (tiff.height, tiff.width))

    return np.array(images)

# FIN
