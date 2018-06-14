# -*- coding: utf-8 -*-
"""
This script computes the voxel histograms for each stack in a cFos folder list

@author: Dreosti Lab
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
import os
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import SZ_cfos as SZCFOS
import SZ_summary as SZS
import SZ_analysis as SZA

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Set Summary List
#summaryListFile = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Excel_Sheets\Test_Comparison_2.xlsx'
summaryListFile = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\Test_all_ARK.xlsx'

# Set Mask Path
#mask_path = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Anatomical_Masks\BRAIN_DAPI_MASK_FINAL2.nii'
mask_path = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\C-Fos_Groups\BRAIN_DAPI_MASK_FINAL_ARK.nii'
mask_slice_range_start = 70
mask_slice_range_stop = 120

# Use the normalized stacks?
normalized = True

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
cfos_paths, behaviour_metrics, metric_labels = SZCFOS.read_summarylist(summaryListFile, normalized, change_base_path=True)
num_files = len(cfos_paths)

# Load mask
mask_data, mask_affine, mask_header = SZCFOS.load_nii(mask_path, normalized=True)
mask_data = mask_data[:,:,:,0]
mask_data[:,:,:mask_slice_range_start] = 0
mask_data[:,:,mask_slice_range_stop:] = 0
num_mask_voxels = np.sum(np.sum(np.sum(mask_data)))

# ------------------------------------------------------------------
# Histogram
# ------------------------------------------------------------------

# Measure cFOS in Mask (normalize to "background")
plt.figure()
start = 0
#stop = 60
stop = num_files
for i in range(start, stop, 1):
    
    # Load original (warped) cFos stack
    cfos_data, cfos_affine, cfos_header = SZCFOS.load_nii(cfos_paths[i], normalized)
    masked_values = cfos_data[mask_data == 1]
    
    # Remove reg artifacts (0 value or 32768)
    if(not(normalized)):
        masked_values = masked_values[masked_values != 32768]

    # Measure histogram values
    if(normalized):
        histogram, bin_edges  = np.histogram(masked_values, bins = 10000, range=[-10, 10]);
    else:
        histogram, bin_edges  = np.histogram(masked_values, bins = 10000, range=[-4000, 70000]);        
    bin_width = (bin_edges[1]-bin_edges[0])/2
    bin_centers = bin_edges[:-1] + bin_width

    # Find offset
    offset_bin = np.where(histogram > 100)[0][0].astype(np.uint)
    offset = bin_centers[offset_bin]
    
    # Find median bin
    half_count = np.sum(histogram) / 2
    median_bin = np.round(np.argmin(np.abs(np.cumsum(histogram) - half_count))).astype(np.uint)
    median = bin_centers[median_bin]

    # Find lower quartile bin
    bot_decile_count = np.sum(histogram) / 10
    bot_decile_bin = np.round(np.argmin(np.abs(np.cumsum(histogram) - bot_decile_count))).astype(np.uint)
    bot_decile = bin_centers[bot_decile_bin]

    # Find lower quartile bin
    top_decile_count = 9 * np.sum(histogram) / 10
    top_decile_bin = np.round(np.argmin(np.abs(np.cumsum(histogram) - top_decile_count))).astype(np.uint)
    top_decile = bin_centers[top_decile_bin]

    # Find mode bin
    mode_bin = np.argmax(histogram)
    mode = bin_centers[mode_bin]
    
    # Check whether this is a "false" mode resulting from low dynamic range
    if (np.abs(mode_bin - median_bin) > 70):
        # Find "real" mode bin
        mode_bin = np.argmax(histogram[(mode_bin+70):]) + (mode_bin+70)
        mode = bin_centers[mode_bin]
        
    # Plot histogram
    plt.plot(bin_centers, histogram)
    plt.plot(median, histogram[median_bin], 'ko')
    plt.plot(offset, histogram[offset_bin], 'go')
    plt.plot(mode, histogram[mode_bin], 'k+')
    plt.plot(bot_decile, histogram[bot_decile_bin], 'bo')
    plt.plot(top_decile, histogram[top_decile_bin], 'ro')

    # Save histogram
    if(normalized):
        histogram_file = os.path.dirname(cfos_paths[i]) + r'\voxel_histogram_normalized.npz'
    else:
        histogram_file = os.path.dirname(cfos_paths[i]) + r'\voxel_histogram.npz'        
    np.savez(histogram_file, 
             histogram=histogram, 
             bin_centers=bin_centers, 
             offset = offset,
             median=median,
             bot_decile = bot_decile,
             top_decile = top_decile,
             mode=mode)
 
    print("Saved Histogram " + str(i+1) + ' of ' + str(num_files) + ':\n' + cfos_paths[i] + '\n')
                              
# FIN
