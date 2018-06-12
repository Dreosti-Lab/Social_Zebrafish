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
summaryListFile = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\Test_all.xlsx'

# Set Mask Path
#mask_path = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Anatomical_Masks\BRAIN_DAPI_MASK_FINAL2.nii'
mask_path = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\BRAIN_DAPI_MASK_FINAL2.nii'
mask_slice_range_start = 70
mask_slice_range_stop = 120

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
#cfos_paths, behaviour_metrics, metric_labels = SZCFOS.read_summarylist(summaryListFile, normalized=False)
cfos_paths, behaviour_metrics, metric_labels = SZCFOS.read_summarylist(summaryListFile, normalized=True)
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
for i in range(num_files):
    
    # Load original (warped) cFos stack
    cfos_data, cfos_affine, cfos_header = SZCFOS.load_nii(cfos_paths[i], normalized=True)
    masked_values = cfos_data[mask_data == 1]

    # Measure histogram values
    histogram, bin_edges  = np.histogram(masked_values, bins = 10000, range=[-20, 20]);
    bin_width = (bin_edges[1]-bin_edges[0])/2
    bin_centers = bin_edges[:-1] + bin_width
    
    
    # Plot histogram
    plt.plot(bin_centers, histogram)
                           
    print(str(i+1) + ' of ' + str(num_files) + ':\n' + cfos_paths[i] + '\n')
                              
# FIN
