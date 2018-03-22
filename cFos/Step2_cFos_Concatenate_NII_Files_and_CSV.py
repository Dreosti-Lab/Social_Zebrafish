# -*- coding: utf-8 -*-
"""
This script loads a cFos folder list and builds a 4D .nii file and summary CSV

@author: Dreosti Lab
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'
# -----------------------------------------------------------------------------
# Set "Base Path" for this analysis session'
base_path = r'C:\Users\adamk\Desktop\cFos Experiments'
#base_path = r'D:\Registration'

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import SZ_cfos as SZCFOS
import SZ_summary as SZS
import SZ_analysis as SZA

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Set Folder List
#folderListFile = base_path + r'\Folder_list\wt_list.txt'
folderListFile = base_path + r'\Folder_list\isolation.txt'
#folderListFile = base_path + r'\Folder_list\No_SC.txt'

# Set Background Path
#background_path = base_path + r'\Masks\Diencephalon_Area_10_DIL.tif'
#background_path = base_path + r'\Masks\Diencephalon_Area_8.tif'
background_path = base_path + r'\Masks\Tectum.labels.tif'

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read folder list
folder_names, test_ROIs, stim_ROIs, cfos_names, NS_names, S_names, fish_numbers = SZCFOS.read_folderlist(base_path, folderListFile)
num_folders = len(folder_names)

# Load background mask
background_stack = SZCFOS.load_mask(background_path)
num_background_voxels = np.sum(np.sum(np.sum(background_stack)))

# ------------------------------------------------------------------
# Start Analysis

# Analyze Behaviour for each folder (BPS and SPI for now)
group_values = np.zeros(num_folders)
bps_values = np.zeros(num_folders)
spi_values = np.zeros(num_folders)
vpi_values = np.zeros(num_folders)
dist_values = np.zeros(num_folders)
for i in range(num_folders):
    
    # Retrieve group ID
    # - From folder list name
    
    # Set fish number 
    fish_number = fish_numbers[i]

    # Load tracking data (S)
    behaviour_data = np.load(S_names[i])
    tracking = behaviour_data['tracking']
    fx = tracking[:,0] 
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort = tracking[:,7]
    motion = tracking[:,8]
    
    # Compute VPI (S)
    VPI_s, AllVisibleFrames, AllNonVisibleFrames = SZA.computeVPI(bx, by, test_ROIs[i][fish_number-1], stim_ROIs[i][fish_number-1])
    vpi_values[i] = VPI_s

    # Compute SPI (S)
    SPI_s, AllSocialFrames_TF, AllNONSocialFrames_TF = SZA.computeSPI(bx, by, test_ROIs[i][fish_number-1], stim_ROIs[i][fish_number-1])
    spi_values[i] = SPI_s
    
    # Compute BPS (S)
    BPS_s, avgBout_s = SZS.measure_BPS(motion)
    bps_values[i] = BPS_s
    
    # Load tracking data (NS)
    behaviour_data = np.load(NS_names[i])
    tracking = behaviour_data['tracking']
    fx = tracking[:,0] 
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort = tracking[:,7]
    motion = tracking[:,8]

    # Compute Distance Traveled (NS)
    dist_values[i] = SZA.distance_traveled(bx, by)

    # Save to line in Summary CSV


# Concatenate NII files into one BIG 4-D file
all_stacks = []
for i in range(num_folders):
    image = nib.load(cfos_names[i])
    raw_data = image.get_data() + 32768 # offset from 16-bit signed
    background_value = np.sum(np.sum(np.sum(background_stack * cfos_data)))/num_background_voxels
    norm_data = raw_data/background_value

    norm_image = nib.Nifti1Image(norm_data, image.affine, image.header)
    all_stacks.append(norm_image)
    print(str(i))

big_stack = nib.concat_images(all_stacks)

# Save BIG stack
output_path = base_path + r'\output.nii'
nib.save(big_stack, output_path)

# FIN
