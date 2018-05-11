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
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageSequence
import BONSAI_ARK
from openpyxl import load_workbook

# Utilities for analyzing cFos experiments

# Load NII stack
def load_nii(path, normalized=False): 
    image = nib.load(path)
    image_affine = image.affine
    image_header = image.header
    if(normalized):
        image_data = image.get_data()
    else:
        image_data = image.get_data() + 32768 # offset from 16-bit signed integer
    return image_data, image_affine, image_header

# Save NII stack
def save_nii(path, data, affine, header):
 
    new_img = nib.Nifti1Image(data, affine, header)
    nib.save(new_img, path)

# Load Mask Labels (TIFF)
def load_mask(path, transpose=True):
    # Mask should be transposed to align with .nii loaded images
    tiff = Image.open(path)
    tiff.load()
    num_frames = tiff.n_frames
    if(transpose):
        images = np.zeros((tiff.width, tiff.height, num_frames), dtype=np.uint8) 
        for i, page in enumerate(ImageSequence.Iterator(tiff)):
            images[:,:,i] = np.transpose(np.reshape(page, (tiff.height, tiff.width)))
    else:
        images = np.zeros((tiff.height, tiff.width, num_frames), dtype=np.uint8) 
        for i, page in enumerate(ImageSequence.Iterator(tiff)):
            images[:,:,i] = np.reshape(page, (tiff.height, tiff.width))
 
    return np.array(images)

# Read cFos experiment summary list
def read_summarylist(path, normalized=False):
    
    # Load worksbook/sheet
    summaryWb = load_workbook(filename = path)
    summaryWs = summaryWb.active

    # Extract cell data
    all_cells =  list(summaryWs.values)
    data_cells = all_cells[1:]
    metric_labels = all_cells[0][2:]
    num_rows = len(data_cells)
    
    # Empty lists to fill   
    cfos_paths = []
    behaviour_metrics = []
    for i in range(0,num_rows):

        # Find correct cfos image file path
        current_cell = data_cells[i]
        base_cfos_path = current_cell[0] + current_cell[1]
        if(normalized):
            cfos_image_name = glob.glob(base_cfos_path + '\*warped_red_normalized.nii.gz')[0]
        else:
            cfos_image_name = glob.glob(base_cfos_path + '\*warped_red.nii.gz')[0]            
        cfos_paths.append(cfos_image_name)
        
        # Find behaviour metrics
        behaviour = np.zeros(15);
        for i in range(15): 
            behaviour[i] = current_cell[2 + i]                 
        behaviour_metrics.append(behaviour)
                
    return cfos_paths, behaviour_metrics, metric_labels

# FIN
