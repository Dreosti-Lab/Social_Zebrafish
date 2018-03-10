# -*- coding: utf-8 -*-
"""
This script loads and processes an NII folder list: .nii images and behaviour

@author: Dreosti Lab
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'
# -----------------------------------------------------------------------------
# Set "Base Path" for this analysis session
base_path = r'C:\Users\adamk\Desktop\cFos Experiments'

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SZ_cfos as SZCFOS

# Set Folder List
folderListFile = base_path + r'\test_list.txt'

# Read Folder List
folderFile = open(folderListFile, "r")  #"r" means read the file
folderList = folderFile.readlines()     # returns a list containing the lines
num_folders = len(folderList) 
folder_names = [] # We use this becasue we do not know the exact lenght
nii_names = []
npz_NS_names = []
npz_S_names = []
for i, f in enumerate(folderList):  #enumerate tells you what folder is 'i'
    folder_name = base_path+f[:-1]
    folder_names.append(folder_name)
    
    # Find cFos nii paths: "warped_red"
    nii_name = glob.glob(folder_name + r'\*warped_red.nii.gz')[0]
    nii_names.append(nii_name)
    
    # Find behaviour npz (NS and S)
    npz_name_NS = glob.glob(folder_name + r'\Behaviour\*_NS.npz')[0]
    npz_name_S = glob.glob(folder_name + r'\Behaviour\*_S.npz')[0]
    npz_NS_names.append(npz_name_NS)
    npz_S_names.append(npz_name_S)

# Set Mask Path
mask_path = base_path + r'\Masks\Caudal_Hypothalamus.labels.tif'

# Load mask(s)
mask_stack = SZCFOS.load_mask(mask_path)
num_mask_voxels = np.sum(np.sum(np.sum(mask_stack)))
# ------------------------------------------------------------------
# Start Analysis

# Analyze Behaviour (SPI for now)
motion_values = np.zeros(num_folders)
for i in range(num_folders):
    data = np.load(npz_S_names[i])
    tracking = data['tracking']
    
    fx = tracking[:,0] 
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort = tracking[:,7]
    motion = tracking[:,8]
    motion_values[i] = np.mean(motion)

# Measure cFOS in Mask (normalize to "background"...eventually)
cFos_values = np.zeros(num_folders)
for i in range(num_folders):
    data = SZCFOS.load_nii(nii_names[i])
    cFos_values[i] = np.sum(np.sum(np.sum(mask_stack * data)))/num_mask_voxels
    print(str(i) + ", cFos = " + str(cFos_values[i]) + ", motion = " + str(motion_values[i]))
    
    # Display
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(data[:,:,54])
    plt.subplot(1,2,2)
    plt.imshow(mask_stack[:,:,54] * data[:,:,54])

# FIN
