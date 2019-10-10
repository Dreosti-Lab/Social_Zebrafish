# -*- coding: utf-8 -*-
"""
Create summary (figures and report) for robustness experiment

@author: kampff
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Zebrafish/libs'
#-----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Set "Base Path" for this analysis session
base_path = r'/home/kampff/Data/Zebrafish'
#base_path = r'\\128.40.155.187\data\D R E O S T I   L A B'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats
import seaborn as sns
import pandas as pd

# Import local modules
import SZ_utilities as SZU
import SZ_macros as SZM
import SZ_video as SZV
import SZ_analysis as SZA
import SZ_summary as SZS
import BONSAI_ARK
import glob
import pylab as pl

# Specify Analysis folders
morning_Folder = base_path + '/Robustness/Combined_Morning'
afternoon_Folder = base_path + '/Robustness/Combined_Afternoon'

# Find all the npz files saved for each group and fish with all the information
morning_npzFiles = glob.glob(morning_Folder+'/*.npz')
morning_npzFiles.sort()
afternoon_npzFiles = glob.glob(afternoon_Folder+'/*.npz')
afternoon_npzFiles.sort()

# Calculate how many files
morning_numFiles = np.size(morning_npzFiles, 0)
afternoon_numFiles = np.size(afternoon_npzFiles, 0)

# Allocate space for summary data
morning_VPI_S_ALL = np.zeros(morning_numFiles)
afternoon_VPI_S_ALL = np.zeros(afternoon_numFiles)

# Go through all the files contained in the morning analysis folder
for f, filename in enumerate(morning_npzFiles):

    # Load each npz file
    dataobject = np.load(filename)
    
    # Extract from the npz file
    VPI_S = dataobject['VPI_S']   

    # Make an array with all summary stats
    morning_VPI_S_ALL[f] = VPI_S

# Go through all the files contained in the afternoon analysis folder
for f, filename in enumerate(afternoon_npzFiles):

    # Load each npz file
    dataobject = np.load(filename)
    
    # Extract from the npz file
    VPI_S = dataobject['VPI_S']   

    # Make an array with all summary stats
    afternoon_VPI_S_ALL[f] = VPI_S

# Measure stat: abs(afternoonVPI - morningVPI)
# - What is the magnitide of change in VPI from morning to afternoon?
# -- Should be "small" if VPI is robust, i.e. smaller than random chance pairnigs
morning_VPI = np.mean(morning_VPI_S_ALL)
afternoon_VPI = np.mean(afternoon_VPI_S_ALL)
real_dVPIs = afternoon_VPI_S_ALL - morning_VPI_S_ALL
real_dVPI = np.mean(real_dVPIs)
real_abs_dVPI = np.mean(np.abs(real_dVPIs))

# Build permuted distributions
num_permutes = 10000
test_dVPIs = np.zeros(num_permutes)
for i in range(num_permutes):
    A = np.copy(morning_VPI_S_ALL)
    B = np.copy(afternoon_VPI_S_ALL)
    A = np.random.permutation(A)
    B = np.random.permutation(B)
    test_dVPIs[i] = np.mean(np.abs(B - A))

# Plot morning v. afternoon VPI
plt.figure()
plt.plot([-1,1], [-1,1], 'r')
plt.plot(morning_VPI_S_ALL, afternoon_VPI_S_ALL, 'k.')
plt.xlabel('Morning VPI')
plt.ylabel('Afternoon VPI')
plt.show()

# Plot change in VPI from morning to afternoon
dVPI_histogram, edges = np.histogram(real_dVPIs, bins=20, range=[-2,2])
centres = edges[:-1] + 0.1
plt.figure()
plt.bar(centres, dVPI_histogram, width=0.19, align='center', color=[0,0,0,1])
plt.title('Change in VPI form morning to afternoon')
plt.show()
print('Change in VPI: %f +/- %f' % (np.mean(real_dVPIs), np.std(real_dVPIs)/np.sqrt(len(real_dVPIs))))

# Plot statistics
test_abs_dVPI_histogram, edges = np.histogram(test_dVPIs, bins=2000, range=[-2,2])
centres = edges[:-1] + 0.001
plt.figure()
plt.title('Permutations of |dVPI| vs. measured value')
plt.bar(centres, test_abs_dVPI_histogram, width=0.002, align='center', color=[0,0,0,1])
plt.vlines(real_abs_dVPI, 0, np.max(test_abs_dVPI_histogram), colors='r')
plt.xlim([0.35, 0.70])
plt.show()
count_null = np.sum(test_dVPIs < real_abs_dVPI)
print('P-value: %f' % (count_null/num_permutes))

# FIN
