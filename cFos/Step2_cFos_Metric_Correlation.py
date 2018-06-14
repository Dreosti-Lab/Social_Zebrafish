# -*- coding: utf-8 -*-
"""
This script processes a cFos folder list: .nii images and behaviour
- It computes the correlation between each each voxel in the stack and the selected
-  behaviour metric 

@author: Dreosti Lab
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'

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
#summaryListFile = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Excel_Sheets\Test_Comparison_1.xlsx'
summaryListFile = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\Test_all_ARK.xlsx'

# Set Metric to correlate columns. 0 = Group
selected_metric = 2;            
# 2 = VPI (S)

# Set Groups to include
group = 2;

# Spatial smoothing factor
smooth_factor = 4;

# Set analysis path
analysis_folder = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\Analysis\G2_VPI'
#analysis_folder = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Comparison'

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
cfos_paths, behaviour_metrics, metric_labels = SZCFOS.read_summarylist(summaryListFile, normalized=True, change_base_path=True)
cfos_paths = np.array(cfos_paths)
behaviour_metrics = np.array(behaviour_metrics)

# Assign metrics/paths for group
group_indices = np.where(behaviour_metrics[:,0] == group)[0].astype(np.uint)
cfos_paths = cfos_paths[group_indices]
n = len(cfos_paths)

# ------------------------------------------------------------------
# Measure voxel-wise correlations
# ------------------------------------------------------------------

# Extract relevant metric (explanatory variable)
metric = behaviour_metrics[:, selected_metric]

# Compute correlation between metric and voxel values
corr_stack = SZCFOS.voxel_correlation(cfos_paths, metric, smooth_factor, normalized=True)

# ------------------------------------------------------------------
# Display and Save Results
# ------------------------------------------------------------------

# Save NII stack of results
image_affine = np.eye(4)
save_path = analysis_folder + r'\Corr_Stack.nii.gz'
SZCFOS.save_nii(save_path, corr_stack, image_affine)

# FIN
