# -*- coding: utf-8 -*-
"""
This script loads and processes an NII folder list: .nii images and behaviour

@author: Dreosti Lab
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'
# -----------------------------------------------------------------------------
# Set "Base Path" for this analysis session'
#base_path = r'C:\Users\adamk\Desktop\cFos Experiments'
base_path = r'D:\Registration'

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import SZ_cfos as SZCFOS
import SZ_summary as SZS
import SZ_analysis as SZA

#---------------------------------------------------------------------------

# Set Folder List
folderListFile = base_path + r'\wt_list.txt'
#folderListFile = base_path + r'\isolation_list.txt'

# Set Mask Path
mask_path = base_path + r'\Masks\Caudal_Hypothalamus.labels.tif'

# Set Background Path
background_path = base_path + r'\Masks\Tectum.labels.tif'

#---------------------------------------------------------------------------
# Read folder list
folder_names, test_ROIs, stim_ROIs, cfos_names, NS_names, S_names, fish_numbers = SZCFOS.read_folderlist(base_path, folderListFile)
num_folders = len(folder_names)

# Load mask(s)
mask_stack = SZCFOS.load_mask(mask_path)
num_mask_voxels = np.sum(np.sum(np.sum(mask_stack)))

background_stack = SZCFOS.load_mask(background_path)
num_background_voxels = np.sum(np.sum(np.sum(background_stack)))

# ------------------------------------------------------------------
# Start Analysis

# Analyze Behaviour for each folder (BPS and SPI for now)
bps_values = np.zeros(num_folders)
spi_values = np.zeros(num_folders)
for i in range(num_folders):
    
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
    
    # Compute SPI (S)
    SPI_s, AllSocialFrames_TF, AllNONSocialFrames_TF = SZA.computeSPI(bx, by, test_ROIs[i][fish_number-1], stim_ROIs[i][fish_number-1])
    spi_values[i] = SPI_s
    
    # Compute BPS (S)
    BPS_s, avgBout_s = SZS.measure_BPS(motion)
    bps_values[i] = BPS_s

# Measure cFOS in Mask (normalize to "background"...eventually)
cFos_values = np.zeros(num_folders)
for i in range(num_folders):
    cfos_data = SZCFOS.load_nii(cfos_names[i])
    signal_value = np.sum(np.sum(np.sum(mask_stack * cfos_data)))/num_mask_voxels
    background_value = np.sum(np.sum(np.sum(background_stack * cfos_data)))/num_background_voxels
    
    cFos_values[i] = signal_value/background_value
    print(str(i) + ", cFos = " + str(cFos_values[i]) + ", SPI = " + str(spi_values[i]))

# Make plots
plt.figure()
plt.title("BPS vs cFos - Normalized by Tectum")
plt.plot(bps_values, cFos_values, '.')

plt.figure()
plt.title("SPI vs cFos - Normalized by Tectum")
plt.plot(spi_values, cFos_values, '.')

# FIN


# Huh?
#    x_min = min(test_ROIs[i][fish_number-1,0], stim_ROIs[i][fish_number-1,0])
#    y_min = min(test_ROIs[i][fish_number-1,1], stim_ROIs[i][fish_number-1,1])
#    x_max = max(test_ROIs[i][fish_number-1,0] + test_ROIs[i][fish_number-1,2], stim_ROIs[i][fish_number-1,0] + stim_ROIs[i][fish_number-1,2])
#    y_max = max(test_ROIs[i][fish_number-1,1] + test_ROIs[i][fish_number-1,3], stim_ROIs[i][fish_number-1,1] + stim_ROIs[i][fish_number-1,3])

