# -*- coding: utf-8 -*-
"""
This script plots summary stats for ROI cFOS activity across groups

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

# Set cFos file (group A and B)
cFos_file_A = r'??'
cFos_file_B = r'??'

# Load data
npzfile = np.load(cFos_file_A)
cFos_A = npzfile['cFos_values']
group_name_A = npzfile['group_name']
roi_name_A = npzfile['roi_name']

npzfile = np.load(cFos_file_B)
cFos_B = npzfile['cFos_values']
group_name_B = npzfile['group_name']
roi_name_B = npzfile['roi_name']

# Analyze
mean_A = np.mean(cFos_A)
std_A = np.std(cFos_A)

mean_B = np.mean(cFos_B)
std_B = np.std(cFos_B)

# Plot
plt.figure()
plt.bar(0, mean_A)
plt.bar(1, mean_B)

# FIN