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
#summaryListFile = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Excel_Sheets\Test_Comparison_2.xlsx'
summaryListFile = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\Test_all_ARK.xlsx'

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
cfos_paths, behaviour_metrics, metric_labels = SZCFOS.read_summarylist(summaryListFile, normalized=False, change_base_path=True)
num_files = len(cfos_paths)

# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------

# Subtract histogram offset and scale (divide) by mode
for i in range(num_files):
    
    # Read histogram npz
    histogram_file = os.path.dirname(cfos_paths[i]) + r'\voxel_histogram.npz'
    npzfile = np.load(histogram_file)
    histogram = npzfile['histogram']
    bin_centers = npzfile['bin_centers']
    offset = npzfile['offset']
    median = npzfile['median']
    bot_decile = npzfile['bot_decile']
    top_decile = npzfile['top_decile']
    mode = npzfile['mode']

    # Load original (warped) cFos stack
    cfos_data, cfos_affine, cfos_header = SZCFOS.load_nii(cfos_paths[i], normalized=False)
 
    # Subtract offset
    backsub = cfos_data - offset
    
    # Normalize to histogram mode
    normalized = backsub / (mode - offset)
    
    # Save normlaized NII stack...
    save_path = cfos_paths[i][:-7] + '_normalized.nii.gz'      
    SZCFOS.save_nii(save_path, normalized, cfos_affine, cfos_header)
                                
    print("Normalized " + str(i+1) + ' of ' + str(num_files) + ':\n' + cfos_paths[i] + '\n')
                              
# FIN
