# -*- coding: utf-8 -*-
"""
Create summary (figures and report) for all analyzed fish in a social preference experiment

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

# Specify save folder
figureFolder = base_path

# Specify Analysis folder
analysisFolder_ctl = base_path + r'/All_Controls/Analysis'
analysisFolder_ctl_drug = base_path + r'/Buspirone_30uM/Analysis/Control_Drugged'
analysisFolder_iso = base_path + r'/48hrs_Isolation/Analysis'
analysisFolder_drug_30 = base_path + r'/Buspirone_30uM/Analysis/Isolated_Drugged'
analysisFolder_drug_50 = base_path + r'/Buspirone_50uM/Analysis/Isolated_Drugged'

# Find all the npz files saved for each group and fish with all the information
npzFiles_ctl = glob.glob(analysisFolder_ctl + '/*.npz')
npzFiles_ctl_drug = glob.glob(analysisFolder_ctl_drug + '/*.npz')
npzFiles_iso = glob.glob(analysisFolder_iso + '/*.npz')
npzFiles_drug_30 = glob.glob(analysisFolder_drug_30 + '/*.npz')
npzFiles_drug_50 = glob.glob(analysisFolder_drug_50 + '/*.npz')
npzFiles_drug = npzFiles_drug_30 + npzFiles_drug_50

# LOAD CONTROLS

# Calculate how many files
numFiles = np.size(npzFiles_ctl, 0)

# Allocate space for summary data
VPI_NS_CTL = np.zeros(numFiles)
VPI_S_CTL = np.zeros(numFiles)
VPI_NS_BINS_CTL = np.zeros((numFiles, 15))
VPI_S_BINS_CTL = np.zeros((numFiles, 15))

# Go through all the files contained in the analysis folder
for f, filename in enumerate(npzFiles_ctl):

    # Load each npz file
    dataobject = np.load(filename)
    
    # Extract from the npz file
    VPI_NS = dataobject['VPI_NS']    
    VPI_S = dataobject['VPI_S']   
    VPI_NS_BINS = dataobject['VPI_NS_BINS']    
    VPI_S_BINS = dataobject['VPI_S_BINS']

    # Make an array with all summary stats
    VPI_NS_CTL[f] = VPI_NS
    VPI_S_CTL[f] = VPI_S
    VPI_NS_BINS_CTL[f,:] = VPI_NS_BINS
    VPI_S_BINS_CTL[f,:] = VPI_S_BINS

# LOAD CONTROLS DRUGGED

# Calculate how many files
numFiles = np.size(npzFiles_ctl_drug, 0)

# Allocate space for summary data
VPI_NS_CTL_DRUG = np.zeros(numFiles)
VPI_S_CTL_DRUG = np.zeros(numFiles)
VPI_NS_BINS_CTL_DRUG = np.zeros((numFiles, 15))
VPI_S_BINS_CTL_DRUG = np.zeros((numFiles, 15))

# Go through all the files contained in the analysis folder
for f, filename in enumerate(npzFiles_ctl_drug):

    # Load each npz file
    dataobject = np.load(filename)
    
    # Extract from the npz file
    VPI_NS = dataobject['VPI_NS']    
    VPI_S = dataobject['VPI_S']   
    VPI_NS_BINS = dataobject['VPI_NS_BINS']    
    VPI_S_BINS = dataobject['VPI_S_BINS']

    # Make an array with all summary stats
    VPI_NS_CTL_DRUG[f] = VPI_NS
    VPI_S_CTL_DRUG[f] = VPI_S
    VPI_NS_BINS_CTL_DRUG[f,:] = VPI_NS_BINS
    VPI_S_BINS_CTL_DRUG[f,:] = VPI_S_BINS

# LOAD ISO

# Calculate how many files
numFiles = np.size(npzFiles_iso, 0)

# Allocate space for summary data
VPI_NS_ISO = np.zeros(numFiles)
VPI_S_ISO = np.zeros(numFiles)
VPI_NS_BINS_ISO = np.zeros((numFiles, 15))
VPI_S_BINS_ISO = np.zeros((numFiles, 15))

# Go through all the files contained in the analysis folder
for f, filename in enumerate(npzFiles_iso):

    # Load each npz file
    dataobject = np.load(filename)
    
    # Extract from the npz file
    VPI_NS = dataobject['VPI_NS']    
    VPI_S = dataobject['VPI_S']   
    VPI_NS_BINS = dataobject['VPI_NS_BINS']    
    VPI_S_BINS = dataobject['VPI_S_BINS']

    # Make an array with all summary stats
    VPI_NS_ISO[f] = VPI_NS
    VPI_S_ISO[f] = VPI_S
    VPI_NS_BINS_ISO[f,:] = VPI_NS_BINS
    VPI_S_BINS_ISO[f,:] = VPI_S_BINS

# LOAD DRUGGED

# Calculate how many files
numFiles = np.size(npzFiles_drug, 0)

# Allocate space for summary data
VPI_NS_DRUG = np.zeros(numFiles)
VPI_S_DRUG = np.zeros(numFiles)
VPI_NS_BINS_DRUG = np.zeros((numFiles, 15))
VPI_S_BINS_DRUG = np.zeros((numFiles, 15))

# Go through all the files contained in the analysis folder
for f, filename in enumerate(npzFiles_drug):

    # Load each npz file
    dataobject = np.load(filename)
    
    # Extract from the npz file
    VPI_NS = dataobject['VPI_NS']    
    VPI_S = dataobject['VPI_S']   
    VPI_NS_BINS = dataobject['VPI_NS_BINS']    
    VPI_S_BINS = dataobject['VPI_S_BINS']

    # Make an array with all summary stats
    VPI_NS_DRUG[f] = VPI_NS
    VPI_S_DRUG[f] = VPI_S
    VPI_NS_BINS_DRUG[f,:] = VPI_NS_BINS
    VPI_S_BINS_DRUG[f,:] = VPI_S_BINS



# ----------------
# VPI "Binned" Summary Plot
 
plt.figure(figsize=(10.24,7.68))
plt.title("Temporal VPI (one minute bins)")
plt.hlines(0, 0, 15, colors='k', linestyles='dashed')

m = np.nanmean(VPI_S_BINS_CTL, 0)
std = np.nanstd(VPI_S_BINS_CTL, 0)
valid = (np.logical_not(np.isnan(VPI_S_BINS_CTL)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'k', LineWidth=4)
plt.plot(m, 'ko', MarkerSize=7)
plt.plot(m+se, 'k', LineWidth=1)
plt.plot(m-se, 'k', LineWidth=1)

m = np.nanmean(VPI_S_BINS_ISO, 0)
std = np.nanstd(VPI_S_BINS_ISO, 0)
valid = (np.logical_not(np.isnan(VPI_S_BINS_ISO)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'gray', LineWidth=4)
plt.plot(m, 'ko', MarkerSize=7)
plt.plot(m+se, 'gray', LineWidth=1)
plt.plot(m-se, 'gray', LineWidth=1)

m = np.nanmean(VPI_S_BINS_DRUG, 0)
std = np.nanstd(VPI_S_BINS_DRUG, 0)
valid = (np.logical_not(np.isnan(VPI_S_BINS_DRUG)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'g', LineWidth=4)
plt.plot(m, 'go', MarkerSize=7)
plt.plot(m+se, 'g', LineWidth=1)
plt.plot(m-se, 'g', LineWidth=1)

plt.axis([0, 14, -0.5, 0.75])
plt.xlabel('minutes')
plt.ylabel('VPI')


# Format plot layout as tight (applies to both subplots)
plt.tight_layout() 
filename = figureFolder +'/Figure_4B.png'
plt.savefig(filename, dpi=600)
filename = figureFolder +'/Figure_4B.eps'
plt.savefig(filename, dpi=600)
plt.close('all')


# ----------------
# VPI "Binned" Summary Plot (with drugged controls)
 
plt.figure(figsize=(10.24,7.68))
plt.title("Temporal VPI (one minute bins) with drug controls")
plt.hlines(0, 0, 15, colors='k', linestyles='dashed')

m = np.nanmean(VPI_S_BINS_CTL, 0)
std = np.nanstd(VPI_S_BINS_CTL, 0)
valid = (np.logical_not(np.isnan(VPI_S_BINS_CTL)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'k', LineWidth=4)
plt.plot(m, 'ko', MarkerSize=7)
plt.plot(m+se, 'k', LineWidth=1)
plt.plot(m-se, 'k', LineWidth=1)

m = np.nanmean(VPI_S_BINS_CTL_DRUG, 0)
std = np.nanstd(VPI_S_BINS_CTL_DRUG, 0)
valid = (np.logical_not(np.isnan(VPI_S_BINS_CTL)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'y', LineWidth=4)
plt.plot(m, 'yo', MarkerSize=7)
plt.plot(m+se, 'y', LineWidth=1)
plt.plot(m-se, 'y', LineWidth=1)

m = np.nanmean(VPI_S_BINS_ISO, 0)
std = np.nanstd(VPI_S_BINS_ISO, 0)
valid = (np.logical_not(np.isnan(VPI_S_BINS_ISO)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'gray', LineWidth=4)
plt.plot(m, 'ko', MarkerSize=7)
plt.plot(m+se, 'gray', LineWidth=1)
plt.plot(m-se, 'gray', LineWidth=1)

m = np.nanmean(VPI_S_BINS_DRUG, 0)
std = np.nanstd(VPI_S_BINS_DRUG, 0)
valid = (np.logical_not(np.isnan(VPI_S_BINS_DRUG)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'g', LineWidth=4)
plt.plot(m, 'go', MarkerSize=7)
plt.plot(m+se, 'g', LineWidth=1)
plt.plot(m-se, 'g', LineWidth=1)

plt.axis([0, 14, -0.5, 0.75])
plt.xlabel('minutes')
plt.ylabel('VPI')

# Format plot layout as tight (applies to both subplots)
plt.tight_layout() 
filename = figureFolder +'/Figure_4B_wCtlDrug.png'
plt.savefig(filename, dpi=600)
filename = figureFolder +'/Figure_4B_wCtlDrug.eps'
plt.savefig(filename, dpi=600)
plt.close('all')

# ----------------
# VPI "Binned" Summary Plot (with drugged controls) seperated by Sociality
 
plt.figure(figsize=(10.24,7.68))
plt.title("Temporal VPI (one minute bins) with drug controls and sociality")
plt.hlines(0, 0, 15, colors='k', linestyles='dashed')

ALL = VPI_S_BINS_CTL
SPP = VPI_S_BINS_CTL[VPI_S_CTL > 0.5, :]
SMM = VPI_S_BINS_CTL[VPI_S_CTL < -0.5, :]

m = np.nanmean(ALL, 0)
std = np.nanstd(ALL, 0)
valid = (np.logical_not(np.isnan(ALL)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'k', LineWidth=4)
plt.plot(m, 'ko', MarkerSize=7)
plt.plot(m+se, 'k', LineWidth=1)
plt.plot(m-se, 'k', LineWidth=1)

m = np.nanmean(SPP, 0)
std = np.nanstd(SPP, 0)
valid = (np.logical_not(np.isnan(SPP)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'r--', LineWidth=4)
plt.plot(m, 'ro', MarkerSize=7)
plt.plot(m+se, 'r', LineWidth=1)
plt.plot(m-se, 'r', LineWidth=1)

m = np.nanmean(SMM, 0)
std = np.nanstd(SMM, 0)
valid = (np.logical_not(np.isnan(SMM)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'b--', LineWidth=4)
plt.plot(m, 'bo', MarkerSize=7)
plt.plot(m+se, 'b', LineWidth=1)
plt.plot(m-se, 'b', LineWidth=1)


ALL = VPI_S_BINS_CTL_DRUG
SPP = VPI_S_BINS_CTL_DRUG[VPI_S_CTL_DRUG > 0.5, :]
SMM = VPI_S_BINS_CTL_DRUG[VPI_S_CTL_DRUG < -0.5, :]

m = np.nanmean(ALL, 0)
std = np.nanstd(ALL, 0)
valid = (np.logical_not(np.isnan(ALL)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'y', LineWidth=4)
plt.plot(m, 'yo', MarkerSize=7)
plt.plot(m+se, 'y', LineWidth=1)
plt.plot(m-se, 'y', LineWidth=1)

m = np.nanmean(SPP, 0)
std = np.nanstd(SPP, 0)
valid = (np.logical_not(np.isnan(SPP)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'm--', LineWidth=4)
plt.plot(m, 'mo', MarkerSize=7)
plt.plot(m+se, 'm', LineWidth=1)
plt.plot(m-se, 'm', LineWidth=1)

m = np.nanmean(SMM, 0)
std = np.nanstd(SMM, 0)
valid = (np.logical_not(np.isnan(SMM)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'c--', LineWidth=4)
plt.plot(m, 'co', MarkerSize=7)
plt.plot(m+se, 'c', LineWidth=1)
plt.plot(m-se, 'c', LineWidth=1)





m = np.nanmean(VPI_S_BINS_ISO, 0)
std = np.nanstd(VPI_S_BINS_ISO, 0)
valid = (np.logical_not(np.isnan(VPI_S_BINS_ISO)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'gray', LineWidth=4)
plt.plot(m, 'ko', MarkerSize=7)
plt.plot(m+se, 'gray', LineWidth=1)
plt.plot(m-se, 'gray', LineWidth=1)

m = np.nanmean(VPI_S_BINS_DRUG, 0)
std = np.nanstd(VPI_S_BINS_DRUG, 0)
valid = (np.logical_not(np.isnan(VPI_S_BINS_DRUG)))
n = np.sum(valid, 0)
print(n)
se = std/np.sqrt(n-1)
plt.plot(m, 'g', LineWidth=4)
plt.plot(m, 'go', MarkerSize=7)
plt.plot(m+se, 'g', LineWidth=1)
plt.plot(m-se, 'g', LineWidth=1)

plt.axis([0, 14, -1.0, 1.0])
plt.xlabel('minutes')
plt.ylabel('VPI')

# Format plot layout as tight (applies to both subplots)
plt.tight_layout() 
filename = figureFolder +'/Figure_4B_wCtlDrug_sociality.png'
plt.savefig(filename, dpi=600)
filename = figureFolder +'/Figure_4B_wCtlDrug_sociality.eps'
plt.savefig(filename, dpi=600)
plt.close('all')

#-----------------------------------------------------------------------------
# Comparison statistics
#-----------------------------------------------------------------------------
report_path = figureFolder + '/report_Figure4B.txt'
report_file = open(report_path, 'w')

# Controls vs Short Isolation Drugged
for i in range(15):
    S1 = VPI_S_BINS_CTL[:,i]
    valid = (np.logical_not(np.isnan(S1)))
    S1 = S1[valid]

    S2 = VPI_S_BINS_DRUG[:,i]
    valid = (np.logical_not(np.isnan(S2)))
    S2 = S2[valid]
 
    line = '\nControls (S1) vs Short Isolation Drugged (S2): Bin - ' + str(i) 
    report_file.write(line + '\n')
    print(line)
    line = 'Mean (S1): ' + str(np.mean(S1)) + '\nMean (S2): ' + str(np.mean(S2)) 
    report_file.write(line + '\n')
    print(line)

    # Statistics: Compare S1 vs. S2 (relative TTEST)
    result = stats.ttest_ind(S1, S2)
    line = "P-Value (S1 vs. S2):" + str(result[1]) + ' (Un-paired T-Test)'
    report_file.write(line + '\n')
    print(line)

    # Non-parametric version of independent TTest
    result = stats.mannwhitneyu(S1, S2, True)
    line = "P-Value (S1 vs. S2):" + str(result[1]) + ' (Mann-Whitney U-Test)'
    report_file.write(line + '\n')
    print(line)

# Short Isolation vs Short Isolation Drugged
for i in range(15):
    S1 = VPI_S_BINS_ISO[:,i]
    valid = (np.logical_not(np.isnan(S1)))
    S1 = S1[valid]

    S2 = VPI_S_BINS_DRUG[:,i]
    valid = (np.logical_not(np.isnan(S2)))
    S2 = S2[valid]


    line = '\nShort Isolation (S1) vs Short Isolation Drugged (S2): Bin - ' + str(i) 
    report_file.write(line + '\n')
    print(line)
    line = 'Mean (S1): ' + str(np.mean(S1)) + '\nMean (S2): ' + str(np.mean(S2)) 
    report_file.write(line + '\n')
    print(line)

    # Statistics: Compare S1 vs. S2 (relative TTEST)
    result = stats.ttest_ind(S1, S2)
    line = "P-Value (S1 vs. S2):" + str(result[1]) + ' (Un-paired T-Test)'
    report_file.write(line + '\n')
    print(line)

    # Non-parametric version of independent TTest
    result = stats.mannwhitneyu(S1, S2, True)
    line = "P-Value (S1 vs. S2):" + str(result[1]) + ' (Mann-Whitney U-Test)'
    report_file.write(line + '\n')
    print(line)


report_file.close()

# FIN
