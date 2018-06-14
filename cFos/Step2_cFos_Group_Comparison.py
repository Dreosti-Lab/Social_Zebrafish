# -*- coding: utf-8 -*-
"""
This script loads and processes two groups in a cFos folder list.
- It computes the t-score differences between each group for each voxel in the stack 

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
#summaryListFile = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Excel_Sheets\Test_Comparison_1.xlsx'
summaryListFile = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\Test_all_ARK.xlsx'

# Set Groups to compare
group_A = 2;
group_B = 4;

# Set VPI thresholds
VPI_min_A = -1.1
VPI_max_A = 1.1
VPI_min_B = -1.1
VPI_max_B = 1.1

# Spatial smoothing factor
smooth_factor = 4;

# Set analysis path
analysis_folder = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\Analysis\G2vG4'
#analysis_folder = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Comparison'

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
cfos_paths, behaviour_metrics, metric_labels = SZCFOS.read_summarylist(summaryListFile, normalized=True, change_base_path=True)
cfos_paths = np.array(cfos_paths)
behaviour_metrics = np.array(behaviour_metrics)

# Assign metrics/paths for each group
group_A_correct_id = (behaviour_metrics[:,0] == group_A)
group_B_correct_id = (behaviour_metrics[:,0] == group_B)

group_A_metric_in_range = (behaviour_metrics[:,2] > VPI_min_A) * (behaviour_metrics[:,2] <= VPI_max_A)
group_B_metric_in_range = (behaviour_metrics[:,2] > VPI_min_B) * (behaviour_metrics[:,2] <= VPI_max_B)

group_A_indices = np.where(group_A_correct_id * group_A_metric_in_range)[0].astype(np.uint)
group_B_indices = np.where(group_B_correct_id * group_B_metric_in_range)[0].astype(np.uint)

cfos_paths_A = cfos_paths[group_A_indices]
cfos_paths_B = cfos_paths[group_B_indices]
n_A = len(group_A_indices)
n_B = len(group_B_indices)

# ------------------------------------------------------------------
# cFos Descriptive Statistics
# ------------------------------------------------------------------
	
# Measure mean and std stacks for group A and B
mean_stack_A, std_stack_A = SZCFOS.summary_stacks(cfos_paths_A, smooth_factor, normalized=True)
mean_stack_B, std_stack_B = SZCFOS.summary_stacks(cfos_paths_B, smooth_factor, normalized=True)

# Compute t-score stack for (B - A)
# - Subtract meanB - meanA = Diff_Mean
# - Estimate combined STD for A and B: sqrt(stdA^2/nA + stdB^2/nB)
diff_mean = mean_stack_B - mean_stack_A
both_std = np.sqrt( ((std_stack_A*std_stack_A)/n_A) + ((std_stack_B*std_stack_B)/n_B) )
t_stack = diff_mean/both_std

# ------------------------------------------------------------------
# Display and Save Results
# ------------------------------------------------------------------

# Make plots
plt.figure()
plt.subplot(1,2,1)
plt.imshow(diff_mean[:,:,50])
plt.subplot(1,2,2)
plt.imshow(t_stack[:,:,50])

# Save NII stack of results
image_affine = np.eye(4)
save_path = analysis_folder + r'\T_Stack.nii.gz'
SZCFOS.save_nii(save_path, t_stack, image_affine)

save_path = analysis_folder + r'\Diff_Stack.nii.gz'
SZCFOS.save_nii(save_path, diff_mean, image_affine)

save_path = analysis_folder + r'\Mean_Stack_A.nii.gz'
SZCFOS.save_nii(save_path, mean_stack_A, image_affine)

save_path = analysis_folder + r'\Mean_Stack_B.nii.gz'
SZCFOS.save_nii(save_path, mean_stack_B, image_affine)

save_path = analysis_folder + r'\STD_Stack_A.nii.gz'
SZCFOS.save_nii(save_path, std_stack_A, image_affine)

save_path = analysis_folder + r'\STD_Stack_B.nii.gz'
SZCFOS.save_nii(save_path, std_stack_B, image_affine)

save_path = analysis_folder + r'\STD_Stack.nii.gz'
SZCFOS.save_nii(save_path, both_std, image_affine)

# FIN
