# -*- coding: utf-8 -*-
"""
This script loads and normalizes a cFos folder list: .nii images and behaviour

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
summaryListFile = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Excel_Sheets\Test_Comparison_2.xlsx'

# Set Background and Normalization Mask Paths
background_path = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Anatomical_Masks\Background_Mask\Bkg_No_Fish.tif'
normalizer_path = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Anatomical_Masks\Background_Mask\Background_C_Fos_Brain_Area.tif'

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
cfos_paths, behaviour_metrics, metric_labels = SZCFOS.read_summarylist(summaryListFile, normalized=False)
num_files = len(cfos_paths)

# Load masks
normalizer_stack = SZCFOS.load_mask(normalizer_path, transpose=True)
num_normalizer_voxels = np.sum(np.sum(np.sum(normalizer_stack)))

background_stack = SZCFOS.load_mask(background_path, transpose=True)
num_background_voxels = np.sum(np.sum(np.sum(background_stack)))


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------

# Measure cFOS in Mask (normalize to "background")
background_values = np.zeros(num_files)
normalizer_values = np.zeros(num_files)

for i in range(num_files):
    
    # Load original (warped) cFos stack
    cfos_data, cfos_affine, cfos_header = SZCFOS.load_nii(cfos_paths[i], normalized=False)

    # Apply background mask toCfos, sum, and divide by mask voxel count
    background_value = np.sum(np.sum(np.sum(background_stack * cfos_data)))/num_background_voxels
 
    # Subtract background
    backsub = cfos_data - background_value
    
    # Apply normalizer mask toCfos, sum, and divide by mask voxel count
    normalizer_value = np.sum(np.sum(np.sum(normalizer_stack * backsub)))/num_normalizer_voxels

    # Normalize
    normalized = backsub / normalizer_value
    
    # Save normlaized NII stack...
    save_path = cfos_paths[i][:-7] + '_normalized_new.nii.gz'      
    SZCFOS.save_nii(save_path, normalized, cfos_affine, cfos_header)
                            
    # Append to list
    background_values[i] = background_value
    normalizer_values[i] = normalizer_value
    
    print(str(i+1) + ' of ' + str(num_files) + ':\n' + cfos_paths[i] + '\n')
                              
# FIN
