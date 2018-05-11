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

# Set Mask Path
mask_path = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Anatomical_Masks\Diencephalon_Area_1_Caudal_Hypothalamus.tif'

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
cfos_paths, behaviour_metrics, metric_labels = SZCFOS.read_summarylist(summaryListFile, normalized=True)
behaviour_metrics = np.array(behaviour_metrics)
num_files = len(cfos_paths)

# Load masks
mask_stack = SZCFOS.load_mask(mask_path, transpose=True)
num_mask_voxels = np.sum(np.sum(np.sum(mask_stack)))

# ------------------------------------------------------------------
# cFos Analysis
# ------------------------------------------------------------------

# Measure (normalized) cFOS in Mask ROI
cFos_values = np.zeros(num_files)
for i in range(num_files):
    
    # Load original (warped) cFos stack
    cfos_data, cfos_affine, cfos_header = SZCFOS.load_nii(cfos_paths[i], normalized=True)

    # Measure average signal level in mask ROI
    cFos_value = np.sum(np.sum(np.sum(mask_stack * cfos_data)))/num_mask_voxels
                             
    # Append to list
    cFos_values[i] = cFos_value
    
    print(str(i+1) + ' of ' + str(num_files) + ':\n' + cfos_paths[i] + '\n')

# ------------------------------------------------------------------
# Imaging Analysis
# ------------------------------------------------------------------

# Make plots
plt.figure()

# Plot comparisons
for i in range(14):
    plt.subplot(2,7,i+1)
    plt.title(metric_labels[i+1])
    plt.plot(behaviour_metrics[:, i+1], cFos_values, 'k.')
    
# FIN
