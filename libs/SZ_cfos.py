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

# Read cFos experiment folder list
def read_folderlist(base_path, path):

    folderFile = open(path, "r")  #"r" means read the file
    folderList = folderFile.readlines()     # returns a list containing the lines
    folder_names = []
    test_ROIs = []
    stim_ROIs = []
    cfos_names = []
    NS_names = []
    S_names = []
    fish_numbers = []
    for i, f in enumerate(folderList):  #enumerate tells you what folder is 'i'

        # Find folder name (path)
        folder_name = base_path+f[:-1]
        folder_names.append(folder_name)
        print('Checking: ' + folder_name)
        
        # Find ROIs from bonsai workflow - Test Fish
        bonsai_test_name = glob.glob(folder_name + r'\Behaviour\*Social_1*.bonsai')[0]
        test_ROIs.append(BONSAI_ARK.read_bonsai_crop_rois(bonsai_test_name))

        # Find ROIs from bonsai workflow - Stimuli Fish
        bonsai_stim_name = glob.glob(folder_name + r'\Behaviour\Social_Fish.bonsai')[0]
        stim_ROIs.append(BONSAI_ARK.read_bonsai_crop_rois(bonsai_stim_name))
                
        # Find cFos nii paths: "warped_red"
        cfos_name = glob.glob(folder_name + r'\*warped_red.nii.gz')[0]
        cfos_names.append(cfos_name)
        
        # Find behaviour (tracking) npz (NonSocial)
        NS_name = glob.glob(folder_name + r'\Behaviour\*_NS.npz')[0]
        NS_names.append(NS_name)

        # Find behaviour (tracking) npz (Social)
        S_name = glob.glob(folder_name + r'\Behaviour\*_S.npz')[0]
        S_names.append(S_name)

        # Find fish number (in experiment folder)
        fish_number = int(S_name[-7])
        fish_numbers.append(fish_number)

    return folder_names, test_ROIs, stim_ROIs, cfos_names, NS_names, S_names, fish_numbers

# FIN
