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

# Use the normalized stacks?
normalized = True

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
cfos_paths, behaviour_metrics, metric_labels = SZCFOS.read_summarylist(summaryListFile, normalized=True, change_base_path=True)
num_files = len(cfos_paths)

# ------------------------------------------------------------------
# Load Histograms
# ------------------------------------------------------------------

plt.figure()
start = 90
stop = 120
#stop = num_files
for i in range(start, stop, 1):
    
    # Read histogram npz
    if(normalized):
        histogram_file = os.path.dirname(cfos_paths[i]) + r'\voxel_histogram_normalized.npz'
    else:
        histogram_file = os.path.dirname(cfos_paths[i]) + r'\voxel_histogram.npz'        
    npzfile = np.load(histogram_file)
    histogram = npzfile['histogram']
    bin_centers = npzfile['bin_centers']
    offset = npzfile['offset']
    median = npzfile['median']
    mode = npzfile['mode']
    
    # Find bin positions for histogram descriptors
    offset_bin = (np.abs(bin_centers - offset)).argmin()
    median_bin = (np.abs(bin_centers - median)).argmin()
    mode_bin = (np.abs(bin_centers - mode)).argmin()
            
    # Plot histogram
    plt.plot(bin_centers, histogram)
    plt.plot(median, histogram[median_bin], 'ko')
    plt.plot(offset, histogram[offset_bin], 'ro')
    plt.plot(mode, histogram[mode_bin], 'bo')
                            
    print("Plotting Histogram " + str(i+1) + ' of ' + str(num_files) + ':\n' + cfos_paths[i] + '\n')
                              
# FIN
