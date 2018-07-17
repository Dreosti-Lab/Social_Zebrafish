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

# Set analysis planes
analysis_start_plane = 30;
analysis_stop_plane = 120;
analysis_step = 1;
analysis_planes = np.arange(analysis_start_plane, analysis_stop_plane, analysis_step)
#analysis_planes = [48, 67, 112]

#analysis_plane = 48

# Spatial binning factor
bin_factor = 8;

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
mask_planes = mask_data[:,:,analysis_planes]
num_mask_voxels = np.sum(np.sum(mask_planes))

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
cfos_paths, behaviour_metrics, metric_labels = SZCFOS.read_summarylist(summaryListFile, normalized=True, change_base_path=True)
cfos_paths = np.array(cfos_paths)
behaviour_metrics = np.array(behaviour_metrics)
n = len(cfos_paths)

# Extract z-planes from stacks and pre-process for PCA
stack_rows = np.int(np.floor(512 / bin_factor))
stack_cols = np.int(np.floor(512 / bin_factor))
stack_planes = len(analysis_planes)
pix_per_plane = stack_rows*stack_cols
pix_per_strip = pix_per_plane*stack_planes
stack_strips = np.zeros([n, pix_per_strip])
for i in range(0, n, 1):
    
    # Load original (warped) cFos stack
    cfos_planes, cfos_affine, cfos_header = SZCFOS.load_nii_plane(cfos_paths[i], analysis_planes, normalized)
    cfos_planes[mask_planes == 0] = 0;
    
    # Bin each plane and append to strip
    stack_strip = np.zeros(pix_per_strip)
    for z in range(0, stack_planes, 1):
        cfos_binning = cfos_planes[:,:,z].reshape(512 // bin_factor, bin_factor, 512 // bin_factor, bin_factor)
        cfos_binned = cfos_binning.mean(axis=3).mean(axis=1)
        start = z*pix_per_plane
        stop = z*pix_per_plane + pix_per_plane
        stack_strip[start:stop] = np.reshape(cfos_binned, [1, pix_per_plane])
        
    # Store z_plane
    stack_strips[i,:] = stack_strip;
    
    # Report
    print("Loaded " + str(i) + "\n")

# Zero-mean
#mean_z_plane = np.mean(z_planes, axis=0)
#for i in range(0, n, 1):
#    z_planes[i,:] = z_planes[i,:] - mean_z_plane

# Select group subset? (Do PCA on WT only?)
group_correct_id = (behaviour_metrics[:,0] == 2)
group_indices = np.where(group_correct_id)[0].astype(np.uint)
selected_group_strips = stack_strips[group_indices, :]
selected_group_strips = stack_strips

# PCA
pca = PCA(n_components=3)
pca.fit(selected_group_strips)

# Get first two PCs
PC1 = pca.components_[0]
PC2 = pca.components_[1]
# Project all
PC1s = np.zeros(n);
PC2s = np.zeros(n);
for i in range(0, n, 1):
    PC1s[i] = np.dot(stack_strips[i,:], PC1)
    PC2s[i] = np.dot(stack_strips[i,:], PC2)   
#    plt.figure()
#    im = np.reshape(z_planes[i,:], (128, 128))
#    plt.imshow(im)
#    plt.title(str(behaviour_metrics[i,2]))
#    plt.xlabel(str(PC1s[i]))
#    plt.ylabel(str(PC2s[i]))
           
# Plot Projections
plt.figure()
for i in range(0, n, 1):
    ID = behaviour_metrics[i,0] 
    VPI = behaviour_metrics[i,2] 
    if(ID == 2):
        if(VPI > 0.5):
            marker_color = [1.0, 0.0, 0.0, 0.5]
        elif(VPI < -0.5):
            marker_color = [0.0, 0.0, 1.0, 0.5]
        else:
            marker_color = [0.5, 0.5, 0.5, 0.5]
        plt.plot(PC1s[i], PC2s[i], 'o', color = marker_color)
        print("Groups: " + str(ID) + "- VPI: " + str(VPI))
    elif (ID==4):
        marker_color = [0.0, 0.0, 0.0, 1.0]     
        plt.plot(PC1s[i], PC2s[i], '+', color = marker_color)
        print("Groups: " + str(ID) + " -- VPI: " + str(VPI))
plt.xlabel('PC1')
plt.ylabel('PC2')

# Plot PCs
plt.figure()
PC1_image = np.reshape(PC1, (64,64,stack_planes), order='F')
PC2_image = np.reshape(PC2, (64,64,stack_planes), order='F')
plt.subplot(1,2,1)
plt.imshow(PC1_image[:,:,18])
plt.xlabel('PC1')
plt.subplot(1,2,2)
plt.imshow(PC2_image[:,:,18])
plt.xlabel('PC2')

# FIN
