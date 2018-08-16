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
# CH_lower
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

# Compare Masks (vs Controls)
SZCFOS.compare_mask_groups(control_files, group_files, report_path)

# Compare Masks (PP vs MM)
report_path =  cfos_data_folder + mask_folder + r'/report2.txt'
group_files_A = np.empty(1, dtype=object)
group_files_A[0] = cfos_data_folder + mask_folder + r'/Wt_PP_CH_Lower_cFos.npz'

group_files_B = np.empty(1, dtype=object)
group_files_B[0] = cfos_data_folder + mask_folder + r'/Wt_MM_CH_Lower_cFos.npz'
SZCFOS.compare_mask_groups(group_files_A, group_files_B, report_path)

#---------------------------------------------------------------------------
# CH_middle
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

# Compare Masks (vs Controls)
SZCFOS.compare_mask_groups(control_files, group_files, report_path)

# Compare Masks (PP vs MM)
report_path =  cfos_data_folder + mask_folder + r'/report2.txt'
group_files_A = np.empty(1, dtype=object)
group_files_A[0] = cfos_data_folder + mask_folder + r'/Wt_PP_CH_Middle_cFos.npz'

group_files_B = np.empty(1, dtype=object)
group_files_B[0] = cfos_data_folder + mask_folder + r'/Wt_MM_CH_Middle_cFos.npz'
SZCFOS.compare_mask_groups(group_files_A, group_files_B, report_path)

#---------------------------------------------------------------------------
# Tectum
#---------------------------------------------------------------------------
mask_folder = r'/Tectum'
figure_path =  cfos_data_folder + mask_folder + r'/fig.png'
report_path =  cfos_data_folder + mask_folder + r'/report.txt'

# Set cFos files (controls and groups)
control_files = np.empty(4, dtype=object)
control_files[0] = cfos_data_folder + mask_folder + r'/Control_G1_Tectum_cFos.npz'
control_files[1] = cfos_data_folder + mask_folder + r'/Control_G1_Tectum_cFos.npz'
control_files[2] = cfos_data_folder + mask_folder + r'/Control_G3_Tectum_cFos.npz'
control_files[3] = cfos_data_folder + mask_folder + r'/Control_G5_Tectum_cFos.npz'

group_files = np.empty(4, dtype=object)
group_files[0] = cfos_data_folder + mask_folder + r'/Wt_PP_Tectum_cFos.npz'
group_files[1] = cfos_data_folder + mask_folder + r'/Wt_MM_Tectum_cFos.npz'
group_files[2] = cfos_data_folder + mask_folder + r'/Long_Iso_MM_Tectum_cFos.npz'
group_files[3] = cfos_data_folder + mask_folder + r'/48h_MM_Tectum_cFos.npz'

# Make Plot
bar_colours = [ "#ff0000", "#0000ff", "#8b00ef", "#cf79f7"]
SZCFOS.bar_plot_mask_groups(bar_colours, group_files, control_files)
plt.yticks([-0.6,-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8], fontsize=18)
plt.savefig(figure_path)

# Compare Masks (vs Controls)
SZCFOS.compare_mask_groups(control_files, group_files, report_path)

# Compare Masks (PP vs MM)
report_path =  cfos_data_folder + mask_folder + r'/report2.txt'
group_files_A = np.empty(1, dtype=object)
group_files_A[0] = cfos_data_folder + mask_folder + r'/Wt_PP_Tectum_cFos.npz'

group_files_B = np.empty(1, dtype=object)
group_files_B[0] = cfos_data_folder + mask_folder + r'/Wt_MM_Tectum_cFos.npz'
SZCFOS.compare_mask_groups(group_files_A, group_files_B, report_path)

# FIN