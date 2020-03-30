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
analysisFolder_ctl = base_path + r'/Controls'
analysisFolder_iso = base_path + r'/Isolated_Controls'
analysisFolder_drug_30 = base_path + r'/Isolated_Drugged/30'
analysisFolder_drug_50 = base_path + r'/Isolated_Drugged/50'

# Find all the npz files saved for each group and fish with all the information
npzFiles_ctl = glob.glob(analysisFolder_ctl + '/*.npz')
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
# VPI Summary Plot

#Make histogram and plot it with lines 
a_ns,c=np.histogram(VPI_NS_DRUG,  bins=8, range=(-1,1))
a_s,c=np.histogram(VPI_S_DRUG,  bins=8, range=(-1,1))
centers = (c[:-1]+c[1:])/2

#Normalize by tot number of fish
Tot_Fish_NS=numFiles

a_ns_float = np.float32(a_ns)
a_s_float = np.float32(a_s)

a_ns_nor_medium=a_ns_float/Tot_Fish_NS
a_s_nor_medium=a_s_float/Tot_Fish_NS 
 
#-----------
plt.figure(figsize=(10.24,7.68), frameon=False)

plt.vlines(-1.1, 0, 0.85, linewidth= 5)
plt.hlines(0, -1.1, 1.1, linewidth= 5)
plt.axvline(linewidth=1, color='black', linestyle= (0, (5, 5)))
plt.bar(centers, a_s_nor_medium, width=0.25, color=['#204796', '#204796', '#FFFFFF', '#FFFFFF', '#FFFFFF','#FFFFFF','#ED2224','#ED2224'], linewidth=1.0, edgecolor='black')  
#plt.title('Controls', fontsize=52, fontweight='bold')
#plt.ylabel('Rel. Frequency', fontsize=42)
plt.xlabel('VPI', fontsize=42)
plt.axis([-1.1, 1.1, 0, 0.55])
pl.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=32)
#plt.axis([-1.1, 1.1, 0, 0.85])
#pl.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], fontsize=32)
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=32)
plt.tight_layout() 
label = numFiles
plt.text(x= 0.5, y= 0.45, s='n='+ np.str(numFiles), size= 42)

plt.savefig(figureFolder + '/Figure_4A.png', dpi=600)
plt.savefig(figureFolder + '/Figure_4A.eps', dpi=600)
plt.close('all')


#-----------------------------------------------------------------------------
# Comparison statistics
#-----------------------------------------------------------------------------
report_path = figureFolder + '/report_Figure4A.txt'
report_file = open(report_path, 'w')

# Controls vs Short Isolation
S1 = VPI_S_CTL
S2 = VPI_S_ISO
line = '\nControls (S1) vs Short Isolation (S2):'
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

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Controls vs Short Isolation Drugged
S1 = VPI_S_CTL
S2 = VPI_S_DRUG
line = '\nControls (S1) vs Short Isolation Drugged (S2):'
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

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

# Short Isolation vs Short Isolation Drugged
S1 = VPI_S_ISO
S2 = VPI_S_DRUG
line = '\nShort Isolation (S1) vs Short Isolation Drugged (S2):'
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
