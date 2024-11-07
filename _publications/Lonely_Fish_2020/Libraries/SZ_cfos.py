    # -*- coding: utf-8 -*-
"""
Library of utilities for cFos analysis

@author: Dreosti Lab
"""
# -----------------------------------------------------------------------------
# Detect Platform
import platform
if(platform.system() == 'Linux'):
    # Set "Repo Library Path" - Social Zebrafish Repo
    lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Zebrafish/libs'
else:
    # Set "Repo Library Path" - Social Zebrafish Repo
    lib_path = r'C:/Repos/Dreosti-Lab/Social_Zebrafish/libs'

# Set Library Paths
import sys
sys.path.append(lib_path)
# -----------------------------------------------------------------------------

# Import useful libraries
import os
import glob
import nibabel as nib
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageSequence
import BONSAI_ARK
from openpyxl import load_workbook
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd

# Utilities for analyzing cFos experiments

# Load NII stack
def load_nii(path, normalized=False): 
    image = nib.load(path)
    image_affine = image.affine
    image_header = image.header
    if(normalized):
        image_data = image.get_data()
    else:
        image_data = image.get_data() + 32768 # offset from 16-bit signed integer
    return image_data, image_affine, image_header

# Load NII planes
def load_nii_planes(path, planes, normalized=False): 
    image = nib.load(path)
    image_affine = image.affine
    image_header = image.header
    if(normalized):
        image_data = image.get_data()
    else:
        image_data = image.get_data() + 32768 # offset from 16-bit signed integer
    planes_data = image_data[:,:,planes]
    return planes_data, image_affine, image_header

# Load NII plane
def load_nii_plane(path, plane, normalized=False): 
    image = nib.load(path)
    image_affine = image.affine
    image_header = image.header
    if(normalized):
        image_data = image.get_data()
    else:
        image_data = image.get_data() + 32768 # offset from 16-bit signed integer
    plane_data = image_data[:,:,plane]
    return plane_data, image_affine, image_header

# Save NII stack
def save_nii(path, data, affine, header=False):
    
    if(header):
        new_img = nib.Nifti1Image(data, affine, header)
    else:
        new_img = nib.Nifti1Image(data, affine)
    nib.save(new_img, path)

# Load Mask Labels (TIFF)
def load_mask(path, transpose=True):
    # Mask should be transposed to align with .nii loaded images
    tiff = Image.open(path)
    tiff.load()
    num_frames = tiff.n_frames
    if(transpose):
        images = np.zeros((tiff.width, tiff.height, num_frames), dtype=np.uint8) 
        for i, page in enumerate(ImageSequence.Iterator(tiff)):
            images[:,:,i] = np.transpose(np.reshape(page, (tiff.height, tiff.width)))
    else:
        images = np.zeros((tiff.height, tiff.width, num_frames), dtype=np.uint8) 
        for i, page in enumerate(ImageSequence.Iterator(tiff)):
            images[:,:,i] = np.reshape(page, (tiff.height, tiff.width))
 
    return np.array(images)

# Read cFos experiment summary list
def read_summarylist(path, normalized=False, change_base_path=False):
    
    # Load worksbook/sheet
    summaryWb = load_workbook(filename = path)
    summaryWs = summaryWb.active

    # Extract cell data
    all_cells =  list(summaryWs.values)
    data_cells = all_cells[1:]
    metric_labels = all_cells[0][2:]
    num_rows = len(data_cells)
    
    # Empty lists to fill   
    cfos_paths = []
    behaviour_metrics = []
    for i in range(0,num_rows):

        # Find correct cfos image file path
        current_cell = data_cells[i]
        if(change_base_path):
            alt_base_path = r'C:\Users\adamk\Dropbox\Adam_Ele\Last_Hope'
            try:
                base_cfos_path = alt_base_path + current_cell[0].split('VPI')[1] + current_cell[1]
            except IndexError:
                print("Bad path in summary list: Row " + str(i))
                sys.exit()            
        else:
            try:
                base_cfos_path = current_cell[0] + current_cell[1]
            except IndexError:
                print("Bad path in summary list: Row " + str(i))
                sys.exit()
        if(normalized):
            try:
                cfos_image_name = glob.glob(base_cfos_path + '\*warped_red_normalized.nii.gz')[0]
            except IndexError:
                print("No file found with name: " + base_cfos_path + '\*warped_red_normalized.nii.gz')
                sys.exit()
        else:
            try:
                cfos_image_name = glob.glob(base_cfos_path + '\*warped_red.nii.gz')[0]            
            except IndexError:
                print("No file found with name: " + base_cfos_path + '\*warped_red.nii.gz')
                sys.exit()
        # Append path to list
        cfos_paths.append(cfos_image_name)
        
        # Find behaviour metrics
        behaviour = np.zeros(15);
        for i in range(15): 
            behaviour[i] = current_cell[2 + i]                 
        behaviour_metrics.append(behaviour)
                
    return cfos_paths, behaviour_metrics, metric_labels

# Compute summary stacks from a cFos experiment summary list
def summary_stacks(paths, smooth_factor=1, normalized=False):

    # Load all data stacks into 4-D array
    n_stacks = len(paths)    
    for i in range(n_stacks):
        
        # Load stack
        cfos_data, cfos_affine, cfos_header = load_nii(paths[i], normalized=normalized)
        n_stack_rows = np.size(cfos_data, 0)
        n_stack_cols = np.size(cfos_data, 1)
        n_stack_slices = np.size(cfos_data, 2)
        
        # (Optional) Smooth_Factor
        if(smooth_factor > 1):
            smooth_kernel = np.ones((smooth_factor,smooth_factor,smooth_factor)) / np.power(smooth_factor,3)
            smoothed_data = signal.fftconvolve(cfos_data, smooth_kernel, mode='same')
            cfos_data = smoothed_data
        
        # Allocate if first iteration
        if (i == 0):
            group_data = np.zeros((n_stack_rows, n_stack_cols, n_stack_slices, n_stacks), dtype = np.float32)    

        # Fill data stack
        group_data[:,:,:,i] = cfos_data;

        # Report progress        
        summary_mean = np.mean(np.mean(np.mean(cfos_data, 2), 1), 0)
        summary_std = np.std(np.std(np.std(cfos_data, 2), 1), 0)
        print('Summarizing: ' + str(i+1) + ' of ' + str(n_stacks) + ':\n' + paths[i])
        print('Mean: ' + str(summary_mean) + ' - STD: ' + str(summary_std) + '\n')
    	
    # Measure mean and std stack for group 
    mean_stack = np.mean(group_data, 3)
    std_stack = np.std(group_data, 3)
    
    # Check for outliers
    all_means = np.mean(np.mean(np.mean(group_data, 2), 1), 0) 
    all_stds = np.std(np.std(np.std(group_data, 2), 1), 0) 
    plt.figure()
    plt.plot(all_means, 'k')
    plt.plot(all_stds, 'r')
    plt.title("Mean stack intensity of all stacks.")
    
    print

    return mean_stack, std_stack

# Compute correlation between voxel values and explanatory metric
def voxel_correlation(paths, metric, smooth_factor=1, normalized=False):

    # Load all data stacks into 4-D array
    n_stacks = len(paths)    
    for i in range(n_stacks):
        
        # Load stack
        cfos_data, cfos_affine, cfos_header = load_nii(paths[i], normalized=normalized)
        n_stack_rows = np.size(cfos_data, 0)
        n_stack_cols = np.size(cfos_data, 1)
        n_stack_slices = np.size(cfos_data, 2)
        
        # (Optional) Smooth_Factor
        if(smooth_factor > 1):
            smooth_kernel = np.ones((smooth_factor,smooth_factor,smooth_factor)) / np.power(smooth_factor,3)
            smoothed_data = signal.fftconvolve(cfos_data, smooth_kernel, mode='same')
            cfos_data = smoothed_data
        
        # Allocate if first iteration
        if (i == 0):
            group_data = np.zeros((n_stack_rows, n_stack_cols, n_stack_slices, n_stacks), dtype = np.float32)    

        # Fill data stack
        group_data[:,:,:,i] = cfos_data;

        # Report progress        
        print('Loading: ' + str(i+1) + ' of ' + str(n_stacks) + ':\n' + paths[i] + '\n')
    	
    # Compute correlation between metric vector and voxel values 
    corr_stack = np.zeros((n_stack_rows, n_stack_cols, n_stack_slices), dtype = np.float32)    
    
    # Zero-mean, unit variance for metric vector
    metric_vec = metric - np.mean(metric)
    metric_vec = metric_vec / np.std(metric_vec)
    metric_auto_corr = np.correlate(metric_vec, metric_vec)[0]

    # Correlate each voxel vector with the metric vector
    for r in range(n_stack_rows):
        for c in range(n_stack_cols):
            for z in range(n_stack_slices):
                vox_vec = group_data[r,c,z,:]
                vox_vec = vox_vec - np.mean(vox_vec)
                vox_vec = vox_vec / np.std(vox_vec)
                
                vox_auto_corr = np.correlate(vox_vec,vox_vec)[0]
                norm_value = (vox_auto_corr + metric_auto_corr) / 2
                corr_factor = np.correlate(vox_vec, metric_vec) / norm_value
                
                corr_stack[r,c,z] = corr_factor[0]
            
        # Report progress        
        print('Correlating: Row ' + str(r+1) + ' of ' + str(n_stack_rows))

    return corr_stack

# Plot mask groups in seaborn barplot
def bar_plot_mask_groups(bar_colours, group_files, control_files=[]):

    # Is there a control file?
    if (len(control_files) == 0):
        controls=False
    else:
        # Load control data
        num_controls = len(control_files)
        controls_s = np.empty(num_controls, dtype=object)
        controls_means = np.empty(num_controls, dtype=object)
        for i in range(num_controls):
            npzfile = np.load(control_files[i])
            control_data = npzfile['cFos_values']
            control_name = npzfile['group_name']
            control_roi = npzfile['roi_name']
            controls_s[i] = pd.Series(control_data, name=str(control_name))
            controls_means[i] = np.mean(controls_s[i])
            controls=True
    
    # Load group data
    num_groups = len(group_files)
    groups_s = np.empty(num_groups, dtype=object)
    groups_minus_controls_s = np.empty(num_groups, dtype=object)
    for i in range(num_groups):
        npzfile = np.load(group_files[i])
        group_data = npzfile['cFos_values']
        group_name = npzfile['group_name']
        group_roi = npzfile['roi_name']
        groups_s[i] = pd.Series(group_data, name=str(group_name))
        if controls:
            groups_minus_controls_s[i] = pd.Series((group_data-controls_means[i])/controls_means[i], name=str(group_name))    

    # Make plot
    plt.figure()
    if controls:
        df = pd.concat(groups_minus_controls_s, axis=1)
    else:
        df = pd.concat(groups_s, axis=1)
    sns.barplot(data=df, ci=68, capsize=.2, palette=sns.color_palette(bar_colours))

# Compare mask groups using ttest
def compare_mask_groups(group_files_A, group_files_B, report_path):
    
    # Load Group A data
    num_groups = len(group_files_A)
    groups_s_A = np.empty(num_groups, dtype=object)
    for i in range(num_groups):
        npzfile = np.load(group_files_A[i])
        group_data = npzfile['cFos_values']
        group_name = npzfile['group_name']
        group_roi = npzfile['roi_name']
        groups_s_A[i] = pd.Series(group_data, name=str(group_name))

    # Load Group A data
    num_groups = len(group_files_B)
    groups_s_B = np.empty(num_groups, dtype=object)
    for i in range(num_groups):
        npzfile = np.load(group_files_B[i])
        group_data = npzfile['cFos_values']
        group_name = npzfile['group_name']
        group_roi = npzfile['roi_name']
        groups_s_B[i] = pd.Series(group_data, name=str(group_name))

    # Stats
    from scipy.stats import ttest_ind, mannwhitneyu
    report_file = open(report_path, 'w')
    for i in range(num_groups):
        S1 = groups_s_A[i]
        S2 = groups_s_B[i]
        # Statistics: Compare S1 vs. S2 (relative TTEST)
        result = ttest_ind(S1, S2, equal_var = True)
        report = str(S1.name) + ' vs ' + str(S2.name) + ' (Un-paired T-Test)' + ' :: ' + str(result)
        report_file.write(report + '\n')
        print(report)

        result = mannwhitneyu(S1, S2, True)
        report = str(S1.name) + ' vs ' + str(S2.name) + ' (Mann-Whitney U-Test)' + ' :: ' + str(result)
        report_file.write(report + '\n')
        print(report)
        # Non-parametric version of independent TTest
    report_file.close()

# FIN