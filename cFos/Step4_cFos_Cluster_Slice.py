# -*- coding: utf-8 -*-
"""
This script loads a single z-plane from a Summary List, does PCA, and clusters

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
from scipy import signal
from sklearn.decomposition import PCA
from skimage.transform import resize

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------


# Set Summary List
#summaryListFile = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Excel_Sheets\Test_Comparison_1.xlsx'
summaryListFile = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\Test_all_ARK.xlsx'

# Set analysis slice
analysis_plane = 45;

# Spatial binning factor
bin_factor = 4;

# Set Mask Path
#mask_path = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Anatomical_Masks\BRAIN_DAPI_MASK_FINAL2.nii'
mask_path = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\C-Fos_Groups\BRAIN_DAPI_MASK_FINAL_ARK.nii'

# Use the normalized stacks?
normalized = True

# Set analysis path
analysis_folder = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope\Analysis\PCA'
#analysis_folder = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Comparison'

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Load mask
mask_data, mask_affine, mask_header = SZCFOS.load_nii(mask_path, normalized=True)
mask_data = mask_data[:,:,:,0]
mask_plane = mask_data[:,:,analysis_plane]
num_mask_voxels = np.sum(np.sum(mask_data))

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
cfos_paths, behaviour_metrics, metric_labels = SZCFOS.read_summarylist(summaryListFile, normalized=True, change_base_path=True)
cfos_paths = np.array(cfos_paths)
behaviour_metrics = np.array(behaviour_metrics)
n = len(cfos_paths)
#n = 10

# Extract z-plane from stacks and pre-process for PCA
stack_rows = np.int(np.floor(512 / bin_factor))
stack_cols = np.int(np.floor(512 / bin_factor))
z_planes = np.zeros([n, stack_rows*stack_cols])
for i in range(0, n, 1):
    
    # Load original (warped) cFos stack
    cfos_plane, cfos_affine, cfos_header = SZCFOS.load_nii_plane(cfos_paths[i], analysis_plane, normalized)
    cfos_plane[mask_plane == 0] = 0;
    
    # Bin image
    cfos_binning = cfos_plane.reshape(512 // bin_factor, bin_factor, 512 // bin_factor, bin_factor)
    cfos_binned = cfos_binning.mean(axis=3).mean(axis=1)
    z_plane = np.reshape(cfos_binned, [1, stack_rows*stack_cols])
        
    # Store z_plane
    z_planes[i,:] = z_plane;
    
    # Report
    print("Loaded " + str(i) + "\n")

# Zero-mean
mean_z_plane = np.mean(z_planes, axis=0)
for i in range(0, n, 1):
    z_planes[i,:] = z_planes[i,:] - mean_z_plane

# PCA
pca = PCA(n_components=2)
pca.fit(z_planes)

# Get first two PCs
PC1 = pca.components_[0]
PC2 = pca.components_[1]

# Project all
PC1s = np.zeros(n);
PC2s = np.zeros(n);
for i in range(0, n, 1):
    PC1s[i] = np.dot(z_planes[i,:], PC1)
    PC2s[i] = np.dot(z_planes[i,:], PC2)
           
# Plot Projections
plt.figure()
plt.plot(PC1s, PC2s, 'k.')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Plot PCs
plt.figure()
PC1_image =  np.reshape(PC1, (128,128))
PC2_image =  np.reshape(PC2, (128,128))
plt.subplot(1,2,1)
plt.imshow(PC1_image)
plt.xlabel('PC1')
plt.subplot(1,2,2)
plt.imshow(PC2_image)
plt.xlabel('PC2')

# FIN
