# -*- coding: utf-8 -*-
"""
This script makes plots and summary stats for ROI masks of cFOS activity across groups

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
import numpy as np
import matplotlib.pyplot as plt
import SZ_cfos as SZCFOS
import seaborn as sns
sns.set_style("whitegrid")
import pandas as pd

#---------------------------------------------------------------------------
#cfos_data_folder = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Final_Images\cfos_mask'
cfos_data_folder = r'/home/kampff/Dropbox/Adam_Ele/Movie_Clip/cfos_mask'
#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
# CH_Middle
#---------------------------------------------------------------------------
mask_folder = r'/CH_middle'
figure_path =  cfos_data_folder + mask_folder + r'/fig.png'
report_path =  cfos_data_folder + mask_folder + r'/report.txt'

# Set cFos files (controls and groups)
control_files = np.empty(4, dtype=object)
control_files[0] = cfos_data_folder + mask_folder + r'/Control_G1_CH_Middle_cFos.npz'
control_files[1] = cfos_data_folder + mask_folder + r'/Control_G1_CH_Middle_cFos.npz'
control_files[2] = cfos_data_folder + mask_folder + r'/Control_G3_CH_Middle_cFos.npz'
control_files[3] = cfos_data_folder + mask_folder + r'/Control_G5_CH_Middle_cFos.npz'

group_files = np.empty(4, dtype=object)
group_files[0] = cfos_data_folder + mask_folder + r'/Wt_PP_CH_Middle_cFos.npz'
group_files[1] = cfos_data_folder + mask_folder + r'/Wt_MM_CH_Middle_cFos.npz'
group_files[2] = cfos_data_folder + mask_folder + r'/Long_Iso_MM_CH_Middle_cFos.npz'
group_files[3] = cfos_data_folder + mask_folder + r'/48h_MM_CH_Middle_cFos.npz'

# Make Plot
bar_colours = [ "#ff0000", "#0000ff", "#8b00ef", "#cf79f7"]
SZCFOS.bar_plot_mask_groups(bar_colours, group_files, control_files)
plt.yticks([-0.6,-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8], fontsize=18)
plt.savefig(figure_path)

# Compare Masks
print('\nCH_Middle:')
group_files_A = np.empty(3, dtype=object)
group_files_A[0] = cfos_data_folder + mask_folder + r'/Wt_PP_CH_Middle_cFos.npz'
group_files_A[1] = cfos_data_folder + mask_folder + r'/Wt_PP_CH_Middle_cFos.npz'
group_files_A[2] = cfos_data_folder + mask_folder + r'/Long_Iso_MM_CH_Middle_cFos.npz'

group_files_B = np.empty(3, dtype=object)
group_files_B[0] = cfos_data_folder + mask_folder + r'/Wt_MM_CH_Middle_cFos.npz'
group_files_B[1] = cfos_data_folder + mask_folder + r'/Control_G1_CH_Middle_cFos.npz'
group_files_B[2] = cfos_data_folder + mask_folder + r'/Control_G3_CH_Middle_cFos.npz'
SZCFOS.compare_mask_groups(group_files_A, group_files_B, report_path)


#---------------------------------------------------------------------------
# CH_Lower
#---------------------------------------------------------------------------
mask_folder = r'/CH_lower'
figure_path =  cfos_data_folder + mask_folder + r'/fig.png'
report_path =  cfos_data_folder + mask_folder + r'/report.txt'

# Set cFos files (controls and groups)
control_files = np.empty(4, dtype=object)
control_files[0] = cfos_data_folder + mask_folder + r'/Control_G1_CH_Lower_cFos.npz'
control_files[1] = cfos_data_folder + mask_folder + r'/Control_G1_CH_Lower_cFos.npz'
control_files[2] = cfos_data_folder + mask_folder + r'/Control_G3_CH_Lower_cFos.npz'
control_files[3] = cfos_data_folder + mask_folder + r'/Control_G5_CH_Lower_cFos.npz'

group_files = np.empty(4, dtype=object)
group_files[0] = cfos_data_folder + mask_folder + r'/Wt_PP_CH_Lower_cFos.npz'
group_files[1] = cfos_data_folder + mask_folder + r'/Wt_MM_CH_Lower_cFos.npz'
group_files[2] = cfos_data_folder + mask_folder + r'/Long_Iso_MM_CH_Lower_cFos.npz'
group_files[3] = cfos_data_folder + mask_folder + r'/48h_MM_CH_Lower_cFos.npz'

# Make Plot
bar_colours = [ "#ff0000", "#0000ff", "#8b00ef", "#cf79f7"]
SZCFOS.bar_plot_mask_groups(bar_colours, group_files, control_files)
plt.yticks([-0.6,-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8], fontsize=18)
plt.savefig(figure_path)

# Compare Masks
print('\nCH_Lower:')
group_files_A = np.empty(3, dtype=object)
group_files_A[0] = cfos_data_folder + mask_folder + r'/Wt_PP_CH_Lower_cFos.npz'
group_files_A[1] = cfos_data_folder + mask_folder + r'/Wt_PP_CH_Lower_cFos.npz'
group_files_A[2] = cfos_data_folder + mask_folder + r'/Long_Iso_MM_CH_Lower_cFos.npz'

group_files_B = np.empty(3, dtype=object)
group_files_B[0] = cfos_data_folder + mask_folder + r'/Wt_MM_CH_Lower_cFos.npz'
group_files_B[1] = cfos_data_folder + mask_folder + r'/Control_G1_CH_Lower_cFos.npz'
group_files_B[2] = cfos_data_folder + mask_folder + r'/Control_G3_CH_Lower_cFos.npz'
SZCFOS.compare_mask_groups(group_files_A, group_files_B, report_path)


#---------------------------------------------------------------------------
# Pa
#---------------------------------------------------------------------------
mask_folder = r'/Pa'
figure_path =  cfos_data_folder + mask_folder + r'/fig.png'
report_path =  cfos_data_folder + mask_folder + r'/report.txt'

# Set cFos files (controls and groups)
control_files = np.empty(4, dtype=object)
control_files[0] = cfos_data_folder + mask_folder + r'/Control_G1_Pa_cFos.npz'
control_files[1] = cfos_data_folder + mask_folder + r'/Control_G1_Pa_cFos.npz'
control_files[2] = cfos_data_folder + mask_folder + r'/Control_G3_Pa_cFos.npz'
control_files[3] = cfos_data_folder + mask_folder + r'/Control_G5_Pa_cFos.npz'

group_files = np.empty(4, dtype=object)
group_files[0] = cfos_data_folder + mask_folder + r'/Wt_PP_Pa_cFos.npz'
group_files[1] = cfos_data_folder + mask_folder + r'/Wt_MM_Pa_cFos.npz'
group_files[2] = cfos_data_folder + mask_folder + r'/Long_Iso_MM_Pa_cFos.npz'
group_files[3] = cfos_data_folder + mask_folder + r'/48h_MM_Pa_cFos.npz'

# Make Plot
bar_colours = [ "#ff0000", "#0000ff", "#8b00ef", "#cf79f7"]
SZCFOS.bar_plot_mask_groups(bar_colours, group_files, control_files)
plt.yticks([-0.6, -0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2], fontsize=18)
plt.savefig(figure_path)

# Compare Masks
print('\nPa:')
group_files_A = np.empty(4, dtype=object)
group_files_A[0] = cfos_data_folder + mask_folder + r'/Wt_PP_Pa_cFos.npz'
group_files_A[1] = cfos_data_folder + mask_folder + r'/Wt_PP_Pa_cFos.npz'
group_files_A[2] = cfos_data_folder + mask_folder + r'/Long_Iso_MM_Pa_cFos.npz'
group_files_A[3] = cfos_data_folder + mask_folder + r'/48h_MM_Pa_cFos.npz'

group_files_B = np.empty(4, dtype=object)
group_files_B[0] = cfos_data_folder + mask_folder + r'/Wt_MM_Pa_cFos.npz'
group_files_B[1] = cfos_data_folder + mask_folder + r'/Control_G1_Pa_cFos.npz'
group_files_B[2] = cfos_data_folder + mask_folder + r'/Control_G3_Pa_cFos.npz'
group_files_B[3] = cfos_data_folder + mask_folder + r'/Control_G5_Pa_cFos.npz'
SZCFOS.compare_mask_groups(group_files_A, group_files_B, report_path)

#---------------------------------------------------------------------------
# PM
#---------------------------------------------------------------------------
mask_folder = r'/PM'
figure_path =  cfos_data_folder + mask_folder + r'/fig.png'
report_path =  cfos_data_folder + mask_folder + r'/report.txt'

# Set cFos files (controls and groups)
control_files = np.empty(4, dtype=object)
control_files[0] = cfos_data_folder + mask_folder + r'/Control_G1_PM_cFos.npz'
control_files[1] = cfos_data_folder + mask_folder + r'/Control_G1_PM_cFos.npz'
control_files[2] = cfos_data_folder + mask_folder + r'/Control_G3_PM_cFos.npz'
control_files[3] = cfos_data_folder + mask_folder + r'/Control_G5_PM_cFos.npz'

group_files = np.empty(4, dtype=object)
group_files[0] = cfos_data_folder + mask_folder + r'/Wt_PP_PM_cFos.npz'
group_files[1] = cfos_data_folder + mask_folder + r'/Wt_MM_PM_cFos.npz'
group_files[2] = cfos_data_folder + mask_folder + r'/Long_Iso_MM_PM_cFos.npz'
group_files[3] = cfos_data_folder + mask_folder + r'/48h_MM_PM_cFos.npz'

# Make Plot
bar_colours = [ "#ff0000", "#0000ff", "#8b00ef", "#cf79f7"]
SZCFOS.bar_plot_mask_groups(bar_colours, group_files, control_files)
plt.yticks([-0.6, -0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2], fontsize=18)
plt.savefig(figure_path)

# Compare Masks
print('\nPM:')
group_files_A = np.empty(4, dtype=object)
group_files_A[0] = cfos_data_folder + mask_folder + r'/Wt_PP_PM_cFos.npz'
group_files_A[1] = cfos_data_folder + mask_folder + r'/Wt_PP_PM_cFos.npz'
group_files_A[2] = cfos_data_folder + mask_folder + r'/Long_Iso_MM_PM_cFos.npz'
group_files_A[3] = cfos_data_folder + mask_folder + r'/48h_MM_PM_cFos.npz'

group_files_B = np.empty(4, dtype=object)
group_files_B[0] = cfos_data_folder + mask_folder + r'/Wt_MM_PM_cFos.npz'
group_files_B[1] = cfos_data_folder + mask_folder + r'/Control_G1_PM_cFos.npz'
group_files_B[2] = cfos_data_folder + mask_folder + r'/Control_G3_PM_cFos.npz'
group_files_B[3] = cfos_data_folder + mask_folder + r'/Control_G5_PM_cFos.npz'
SZCFOS.compare_mask_groups(group_files_A, group_files_B, report_path)


#---------------------------------------------------------------------------
# Tectum
#---------------------------------------------------------------------------
mask_folder = r'/Tectum'
figure_path =  cfos_data_folder + mask_folder + r'/fig.png'
report_path =  cfos_data_folder + mask_folder + r'/report.txt'

# Set cFos files (groups)
group_files = np.empty(7, dtype=object)
group_files[0] = cfos_data_folder + mask_folder + r'/Control_G1_Tectum_cFos.npz'
group_files[1] = cfos_data_folder + mask_folder + r'/Wt_PP_Tectum_cFos.npz'
group_files[2] = cfos_data_folder + mask_folder + r'/Wt_MM_Tectum_cFos.npz'
group_files[3] = cfos_data_folder + mask_folder + r'/Control_G3_Tectum_cFos.npz'
group_files[4] = cfos_data_folder + mask_folder + r'/Long_Iso_MM_Tectum_cFos.npz'
group_files[5] = cfos_data_folder + mask_folder + r'/Control_G5_Tectum_cFos.npz'
group_files[6] = cfos_data_folder + mask_folder + r'/48h_MM_Tectum_cFos.npz'

# Make Plot
bar_colours = ["#c0c0c0","#ff0000", "#0000ff","#c0c0c0", "#8b00ef","#c0c0c0", "#cf79f7"]
SZCFOS.bar_plot_mask_groups(bar_colours, group_files)
plt.yticks([0.6, 1.0, 1.4, 1.8], fontsize=18)
plt.ylim((0.6,1.8))
plt.savefig(figure_path)

# Compare Masks
print('\nTectum:')
group_files_A = np.empty(3, dtype=object)
group_files_A[0] = cfos_data_folder + mask_folder + r'/Control_G1_Tectum_cFos.npz'
group_files_A[1] = cfos_data_folder + mask_folder + r'/Control_G1_Tectum_cFos.npz'
group_files_A[2] = cfos_data_folder + mask_folder + r'/Control_G1_Tectum_cFos.npz'

group_files_B = np.empty(3, dtype=object)
group_files_B[0] = cfos_data_folder + mask_folder + r'/Control_G3_Tectum_cFos.npz'
group_files_B[1] = cfos_data_folder + mask_folder + r'/Control_G5_Tectum_cFos.npz'
group_files_B[2] = cfos_data_folder + mask_folder + r'/Wt_PP_Tectum_cFos.npz'

SZCFOS.compare_mask_groups(group_files_A, group_files_B, report_path)


#---------------------------------------------------------------------------
# PTN_Large
#---------------------------------------------------------------------------
mask_folder = r'/PTN_Large'
figure_path =  cfos_data_folder + mask_folder + r'/fig.png'
report_path =  cfos_data_folder + mask_folder + r'/report.txt'

# Set cFos files (groups)
group_files = np.empty(7, dtype=object)
group_files[0] = cfos_data_folder + mask_folder + r'/Control_G1_PTN_Large_cFos.npz'
group_files[1] = cfos_data_folder + mask_folder + r'/Wt_PP_PTN_Large_cFos.npz'
group_files[2] = cfos_data_folder + mask_folder + r'/Wt_MM_PTN_Large_cFos.npz'
group_files[3] = cfos_data_folder + mask_folder + r'/Control_G3_PTN_Large_cFos.npz'
group_files[4] = cfos_data_folder + mask_folder + r'/Long_Iso_MM_PTN_Large_cFos.npz'
group_files[5] = cfos_data_folder + mask_folder + r'/Control_G5_PTN_Large_cFos.npz'
group_files[6] = cfos_data_folder + mask_folder + r'/48h_MM_PTN_Large_cFos.npz'

# Make Plot
bar_colours = ["#c0c0c0","#ff0000", "#0000ff","#c0c0c0", "#8b00ef","#c0c0c0", "#cf79f7"]
SZCFOS.bar_plot_mask_groups(bar_colours, group_files)
plt.yticks([1.0, 1.5, 2.0, 2.5, 3.0], fontsize=18)
plt.ylim((1.0,3.0))
plt.savefig(figure_path)

# Compare Masks
print('\nPTN (Large):')
group_files_A = np.empty(3, dtype=object)
group_files_A[0] = cfos_data_folder + mask_folder + r'/Control_G1_PTN_Large_cFos.npz'
group_files_A[1] = cfos_data_folder + mask_folder + r'/Control_G1_PTN_Large_cFos.npz'
group_files_A[2] = cfos_data_folder + mask_folder + r'/Control_G1_PTN_Large_cFos.npz'

group_files_B = np.empty(3, dtype=object)
group_files_B[0] = cfos_data_folder + mask_folder + r'/Control_G3_PTN_Large_cFos.npz'
group_files_B[1] = cfos_data_folder + mask_folder + r'/Control_G5_PTN_Large_cFos.npz'
group_files_B[2] = cfos_data_folder + mask_folder + r'/Wt_PP_PTN_Large_cFos.npz'

SZCFOS.compare_mask_groups(group_files_A, group_files_B, report_path)

# FIN