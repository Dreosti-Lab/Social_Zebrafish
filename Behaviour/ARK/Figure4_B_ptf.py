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

## Local Functions
## ------------------------------------------------------------------------
def fill_pauses(pauses, FPS, freeze_threshold):
    # Add first and last pause?
    if FPS == 100:
        long_pause_threshold = freeze_threshold*100
    else:
        long_pause_threshold = freeze_threshold*120
    pausing_frames = np.zeros(200000)
    for pause in pauses:
        start  = np.int(pause[0])
        stop = np.int(pause[4])
        duration = np.int(pause[8])
        if(duration > long_pause_threshold):
            pausing_frames[start:stop] = 1
    if (FPS == 100):
        pausing_frames = pausing_frames[:90000]
    else:
        pausing_frames = pausing_frames[:108000]
    return pausing_frames

def bin_frames(frames, FPS):
    bin_size = 60 * FPS
    reshaped_frames = np.reshape(frames, (bin_size, -1), order='F')
    bins = np.sum(reshaped_frames, 0) / bin_size
    return bins*100

def load_and_process_npz(npzFiles):
    # Analysis Settings
    freeze_threshold = 3.0 # ...in seconds

    # Calculate how many files
    numFiles = np.size(npzFiles, 0)

    # Allocate space for summary data
    PTF_NS_BINS = np.zeros((numFiles,15))
    PTF_S_BINS = np.zeros((numFiles,15))

    # Go through all the files contained in the analysis folder
    for f, filename in enumerate(npzFiles):

        # Load each npz file
        dataobject = np.load(filename)
        
        # Extract from the npz file
        Bouts_NS = dataobject['Bouts_NS']    
        Bouts_S = dataobject['Bouts_S']   
        Pauses_NS = dataobject['Pauses_NS']    
        Pauses_S = dataobject['Pauses_S']

        # Guess FPS
        last_frame_NS = Pauses_NS[-1,4]
        last_frame_S = Pauses_S[-1,4]
        last_frame = (last_frame_NS + last_frame_S) / 2
        if(last_frame < 107000):
            FPS = 100
        else:
            FPS = 120

        # Compute percent time freezing in one minute bins
        pausing_frames_NS = fill_pauses(Pauses_NS, FPS, freeze_threshold)
        pausing_frames_S = fill_pauses(Pauses_S, FPS, freeze_threshold)
        PTF_NS_BINS[f] = bin_frames(pausing_frames_NS, FPS)
        PTF_S_BINS[f] = bin_frames(pausing_frames_S, FPS)

    return PTF_NS_BINS, PTF_S_BINS

def do_stats(S1_in, name_1, S2_in, name_2, report_file):

    # Controls vs Full Isolation (NS)
    # -------------------------------
    valid = (np.logical_not(np.isnan(S1_in)))
    S1 = S1_in[valid]

    valid = (np.logical_not(np.isnan(S2_in)))
    S2 = S2_in[valid]

    line = '\n' + name_1 + '(S1) vs ' + name_2 + '(S2)' 
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

    return

## ------------------------------------------------------------------------

# Specify save folder
figureFolder = base_path + r'/Revised Figure 4'

# Specify Analysis folder
analysisFolder_ctl = base_path + r'/All_Controls/Analysis'
analysisFolder_iso = base_path + r'/48hrs_Isolation/Analysis'
analysisFolder_drug_30 = base_path + r'/Buspirone_30uM/Analysis/Isolated_Drugged'
analysisFolder_drug_50 = base_path + r'/Buspirone_50uM/Analysis/Isolated_Drugged'

# Find all the npz files saved for each group and fish with all the information
npzFiles_ctl = glob.glob(analysisFolder_ctl + '/*.npz')
npzFiles_iso = glob.glob(analysisFolder_iso + '/*.npz')
npzFiles_drug_30 = glob.glob(analysisFolder_drug_30 + '/*.npz')
npzFiles_drug_50 = glob.glob(analysisFolder_drug_50 + '/*.npz')
npzFiles_drug = npzFiles_drug_30 + npzFiles_drug_50

# LOAD CONTROLS
PTF_NS_BINS_CTL, PTF_S_BINS_CTL = load_and_process_npz(npzFiles_ctl)

# LOAD ISO
PTF_NS_BINS_ISO, PTF_S_BINS_ISO = load_and_process_npz(npzFiles_iso)

# LOAD DRUGGED
PTF_NS_BINS_DRUG, PTF_S_BINS_DRUG = load_and_process_npz(npzFiles_drug)

# ----------------------------------
# Plots
# ----------------------------------

# PTF binned 
plt.figure(figsize=(10.24,7.68))
plt.title("Percent Time Freezing (one minute bins)")

m = np.nanmean(PTF_S_BINS_CTL, 0)
std = np.nanstd(PTF_S_BINS_CTL, 0)
valid = (np.logical_not(np.isnan(PTF_S_BINS_CTL)))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'k', LineWidth=4)
plt.plot(m, 'ko', MarkerSize=7)
plt.plot(m+se, 'k', LineWidth=1)
plt.plot(m-se, 'k', LineWidth=1)

m = np.nanmean(PTF_S_BINS_ISO, 0)
std = np.nanstd(PTF_S_BINS_ISO, 0)
valid = (np.logical_not(np.isnan(PTF_S_BINS_ISO)))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'gray', LineWidth=4)
plt.plot(m, 'gray', Marker='o', MarkerSize=7)
plt.plot(m+se, 'gray', LineWidth=1)
plt.plot(m-se, 'gray', LineWidth=1)

m = np.nanmean(PTF_S_BINS_DRUG, 0)
std = np.nanstd(PTF_S_BINS_DRUG, 0)
valid = (np.logical_not(np.isnan(PTF_S_BINS_DRUG)))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'g', LineWidth=4)
plt.plot(m, 'go', MarkerSize=7)
plt.plot(m+se, 'g', LineWidth=1)
plt.plot(m-se, 'g', LineWidth=1)

#plt.axis([0, 14, 0.0, 0.02])
plt.xlabel('minutes')
plt.ylabel('% Freezing')

plt.tight_layout() 
filename = figureFolder +'/Figure_4B_ptf.png'
plt.savefig(filename, dpi=600)
filename = figureFolder +'/Figure_4B_ptf.eps'
plt.savefig(filename, dpi=600)
plt.close('all')


#-----------------------------------------------------------------------------
# Comparison statistics
#-----------------------------------------------------------------------------
report_path = figureFolder + '/report_Figure4B_ptf.txt'
report_file = open(report_path, 'w')

# Controls vs Short Isolation Drugged
for i in range(15):
    bin_label = str(i)
    do_stats(PTF_S_BINS_CTL[:,i], bin_label + " Controls S", PTF_S_BINS_ISO[:,i], "Partial Isolation S", report_file)

report_file.close()

# FIN
