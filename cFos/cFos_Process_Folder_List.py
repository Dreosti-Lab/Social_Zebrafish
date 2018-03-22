# -*- coding: utf-8 -*-
"""
This script loads and processes a cFos folder list: .nii images and behaviour

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
#---------------------------------------------------------------------------

# Set Folder List
folderListFile = base_path + r'\Folder_list\wt_list.txt'
#folderListFile = base_path + r'\Folder_list\isolation.txt'
#folderListFile = base_path + r'\Folder_list\No_SC.txt'

# Set Mask Path
#mask_path = base_path + r'\Masks\Telencephalon_Area_1.labels.tif'
#mask_path = base_path + r'\Masks\Telencephalon_Area_2.labels.tif'
#mask_path = base_path + r'\Masks\Telencephalon_Area_8.tif'

#mask_path = base_path + r'\Masks\Diencephalon_Area_1_Caudal_Hypothalamus.tif'
#mask_path = base_path + r'\Masks\Diencephalon_Area_2_Habenula.labels.tif'
#mask_path = base_path + r'\Masks\Diencephalon_Area_3.labels.tif'
mask_path = base_path + r'\Masks\Diencephalon_Area_4.tif'
#mask_path = base_path + r'\Masks\Diencephalon_Area_7.labels.tif'

# Set Background Path
#background_path = base_path + r'\Masks\Diencephalon_Area_10_DIL.tif'
background_path = base_path + r'\Masks\Diencephalon_Area_8.tif'

#---------------------------------------------------------------------------
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
vpi_values = np.zeros(num_folders)
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
    
    # Compute VPI (S)
    VPI_s, AllVisibleFrames, AllNonVisibleFrames = SZA.computeVPI(bx, by, test_ROIs[i][fish_number-1], stim_ROIs[i][fish_number-1])
    vpi_values[i] = VPI_s

    # Compute SPI (S)
    SPI_s, AllSocialFrames_TF, AllNONSocialFrames_TF = SZA.computeSPI(bx, by, test_ROIs[i][fish_number-1], stim_ROIs[i][fish_number-1])
    spi_values[i] = SPI_s
    
    # Compute BPS (S)
    BPS_s, avgBout_s = SZS.measure_BPS(motion)
    bps_values[i] = BPS_s

# Measure cFOS in Mask (normalize to "background")
signal_values = np.zeros(num_folders)
background_values = np.zeros(num_folders)
normalized_cFos_values = np.zeros(num_folders)
for i in range(num_folders):
    cfos_data = SZCFOS.load_nii(cfos_names[i])
    signal_value = np.sum(np.sum(np.sum(mask_stack * cfos_data)))/num_mask_voxels
    background_value = np.sum(np.sum(np.sum(background_stack * cfos_data)))/num_background_voxels
                             
    # Append to list
    signal_values[i] = signal_value
    background_values[i] = background_value
    normalized_cFos_values[i] = signal_value/background_value
    print(str(i) + ", cFos = " + format(normalized_cFos_values[i], '.3f') + ", SPI = " + format(spi_values[i], '.3f') + ", VPI = " + format(vpi_values[i], '.3f'))

# Make plots
plt.figure()

# Plot unnormalized data
plt.subplot(2,3,1)
plt.title("BPS vs cFos - UnNormalized")
plt.plot(bps_values, signal_values, 'b.')
plt.plot(bps_values, background_values, 'r.')

plt.subplot(2,3,2)
plt.title("SPI vs cFos - UnNormalized")
plt.plot(spi_values, signal_values, 'b.')
plt.plot(spi_values, background_values, 'r.')

plt.subplot(2,3,3)
plt.title("VPI vs cFos - UnNormalized")
plt.plot(vpi_values, signal_values, 'b.')
plt.plot(vpi_values, background_values, 'r.')

# Plot normalized data
plt.subplot(2,3,4)
plt.title("BPS vs cFos - Normalized by DA9")
plt.plot(bps_values, normalized_cFos_values, 'k.')

plt.subplot(2,3,5)
plt.title("SPI vs cFos - Normalized by DA9")
plt.plot(spi_values, normalized_cFos_values, 'k.')

plt.subplot(2,3,6)
plt.title("VPI vs cFos - Normalized by DA9")
plt.plot(vpi_values, normalized_cFos_values, 'k.')

# FIN


# Huh?
#    x_min = min(test_ROIs[i][fish_number-1,0], stim_ROIs[i][fish_number-1,0])
#    y_min = min(test_ROIs[i][fish_number-1,1], stim_ROIs[i][fish_number-1,1])
#    x_max = max(test_ROIs[i][fish_number-1,0] + test_ROIs[i][fish_number-1,2], stim_ROIs[i][fish_number-1,0] + stim_ROIs[i][fish_number-1,2])
#    y_max = max(test_ROIs[i][fish_number-1,1] + test_ROIs[i][fish_number-1,3], stim_ROIs[i][fish_number-1,1] + stim_ROIs[i][fish_number-1,3])

