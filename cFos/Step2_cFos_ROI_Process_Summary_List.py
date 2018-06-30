# -*- coding: utf-8 -*-
"""
This script loads and processes a cFos folder list: .nii images and behaviour

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
summaryListFile = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Excel_Sheets\Test.xlsx'

# Set Group
group = 2;
group_name = r'wt'

# Set VPI thresholds
VPI_min = -1.1
VPI_max = 1.1

# Set ROI Path
roi_path = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Anatomical_Masks\Diencephalon_Area_1_Caudal_Hypothalamus.tif'
roi_name = r'CH'

# Set analysis folder and filename
analysis_folder = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\cFos_Values'
analysis_path = analysis_folder + '\\' + group_name + '_' + roi_name + '_cFos.npz'

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
cfos_paths, behaviour_metrics, metric_labels = SZCFOS.read_summarylist(summaryListFile, normalized=True)
cfos_paths = np.array(cfos_paths)
behaviour_metrics = np.array(behaviour_metrics)

# Assign metrics/paths for each group
group_correct_id = (behaviour_metrics[:,0] == group)
group_metric_in_range = (behaviour_metrics[:,2] > VPI_min) * (behaviour_metrics[:,2] <= VPI_max)
group_indices = np.where(group_correct_id * group_metric_in_range)[0].astype(np.uint)
cfos_paths = cfos_paths[group_indices]
n = len(group_indices)

# Load ROI mask
roi_stack = SZCFOS.load_mask(roi_path, transpose=True)
num_roi_voxels = np.sum(np.sum(np.sum(roi_stack)))

# ------------------------------------------------------------------
# cFos Analysis
# ------------------------------------------------------------------

# Measure (normalized) cFOS in Mask ROI
cFos_values = np.zeros(n)
for i in range(n):
    
    # Load original (warped) cFos stack
    cfos_data, cfos_affine, cfos_header = SZCFOS.load_nii(cfos_paths[i], normalized=True)

    # Measure average signal level in mask ROI
    cFos_value = np.sum(np.sum(np.sum(roi_stack * cfos_data)))/num_roi_voxels
                             
    # Append to list
    cFos_values[i] = cFos_value
    
    print(str(i+1) + ' of ' + str(n) + ':\n' + cfos_paths[i] + '\n')

# ------------------------------------------------------------------
# Save cFos values
# ------------------------------------------------------------------
np.savez(analysis_path, cFos_values=cFos_values, group_name=group_name, roi_name=roi_name)
print("Saved cFos Values: Mean - " + np.mean(cFos_values) + ' +/- STD ' + np.std(cFos_values))

# FIN
