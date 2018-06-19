# -*- coding: utf-8 -*-
"""
This script converts a stack to a display-scaled version.
- Black = min, 0  = 128/gray, White = max

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

# Set Input stack
#summaryListFile = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Excel_Sheets\Test_Comparison_1.xlsx'
stackFolder = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\Analysis\G1vG2spp'
stackFile = stackFolder + r'\Diff_Stack.nii.gz'

# Set Mask Path
#mask_path = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Anatomical_Masks\BRAIN_DAPI_MASK_FINAL2.nii'
mask_path = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\C-Fos_Groups\BRAIN_DAPI_MASK_FINAL_ARK.nii'
mask_slice_range_start = 0
mask_slice_range_stop = 251

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Load mask
mask_data, mask_affine, mask_header = SZCFOS.load_nii(mask_path, normalized=True)
mask_data = mask_data[:,:,:,0]
mask_data[:,:,:mask_slice_range_start] = 0
mask_data[:,:,mask_slice_range_stop:] = 0
num_mask_voxels = np.sum(np.sum(np.sum(mask_data)))

# Load stack
cfos_data, cfos_affine, cfos_header = SZCFOS.load_nii(stackFile, normalized=True)
masked_values = cfos_data[mask_data == 1]
n_stack_rows = np.size(cfos_data, 0)
n_stack_cols = np.size(cfos_data, 1)
n_stack_slices = np.size(cfos_data, 2)
display_data = np.zeros((n_stack_rows, n_stack_cols, n_stack_slices), dtype = np.float32)    

# Compute stats
min_val = np.min(cfos_data[:])
max_val = np.max(cfos_data[:])

# Measure histogram values
histogram, bin_edges  = np.histogram(masked_values, bins = 10000, range=[-10, 10]);
bin_width = (bin_edges[1]-bin_edges[0])/2
bin_centers = bin_edges[:-1] + bin_width

# Find lower 0.25% bin
bot_count = np.sum(histogram) / 250
bot_bin = np.round(np.argmin(np.abs(np.cumsum(histogram) - bot_count))).astype(np.uint)
bot_val = bin_centers[bot_bin]

# Find upper 0.25% bin
top_count = 249 * np.sum(histogram) / 250
top_bin = np.round(np.argmin(np.abs(np.cumsum(histogram) - top_count))).astype(np.uint)
top_val = bin_centers[top_bin]

# Adjust stack
neg_vals = cfos_data < 0.0
pos_vals = cfos_data > 0.0
display_data[neg_vals] = cfos_data[neg_vals] / np.abs(bot_val)
display_data[pos_vals] = cfos_data[pos_vals] / np.abs(top_val)

# ------------------------------------------------------------------
# Measure Descriptive Statistics
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Save Display stack
# ------------------------------------------------------------------

# Save NII stack of results
image_affine = np.eye(4)
displayFile = stackFolder + r'\Diff_Stack_DISPLAY_MIN' + format(bot_val, '.3f') + '_MAX' + format(top_val, '.3f') + '.nii.gz'
SZCFOS.save_nii(displayFile, display_data, image_affine)


# FIN
