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
sns.set_style("whitegrid")
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
def load_and_process_npz(npzFiles):
    # Analysis Settings
    bins = np.arange(0.0, 10.2, 0.2) # 500 200 ms bins
    long_pause_threshold = 3.0 # ...in seconds

    # Calculate how many files
    numFiles = np.size(npzFiles, 0)

    # Allocate space for summary data
    PTF_NS = np.zeros(numFiles)
    PTF_S = np.zeros(numFiles)

    # Go through all the files contained in the analysis folder
    all_pause_durations_NS = []
    all_pause_durations_S = []
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

        # Extract pause durations
        pause_durations_NS = Pauses_NS[:,8]/FPS
        pause_durations_S = Pauses_S[:,8]/FPS

        # Append durations
        all_pause_durations_NS = np.hstack((all_pause_durations_NS, pause_durations_NS))
        all_pause_durations_S = np.hstack((all_pause_durations_S, pause_durations_S))

        # Compute freezes
        freezes_NS = pause_durations_NS[pause_durations_NS > long_pause_threshold]
        percent_freezing_NS = np.sum(freezes_NS) / (last_frame_NS/FPS)
        
        freezes_S = pause_durations_S[pause_durations_S > long_pause_threshold]
        percent_freezing_S = np.sum(freezes_S) / (last_frame_S/FPS)

        PTF_NS[f] = percent_freezing_NS*100
        PTF_S[f] = percent_freezing_S*100

    # Compute histogram of all pause durations
    dur_hist_NS, edges = np.histogram(all_pause_durations_NS, bins)
    dur_hist_S, edges  = np.histogram(all_pause_durations_S, bins)
    dur_hist_NS = dur_hist_NS / np.sum(dur_hist_NS)
    dur_hist_S = dur_hist_S / np.sum(dur_hist_S)

    return PTF_NS, PTF_S, dur_hist_NS, dur_hist_S

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
figureFolder = base_path + r'/Revised Figure 1'

# Specify Analysis folder
analysisFolder_ctl = base_path + r'/All_Controls/Analysis'
analysisFolder_full = base_path + r'/Long_Isolation/Analysis'
analysisFolder_partial = base_path + r'/48hrs_Isolation/Analysis'
analysisFolder_drug_30 = base_path + r'/Buspirone_30uM/Analysis/Isolated_Drugged'
analysisFolder_drug_50 = base_path + r'/Buspirone_50uM/Analysis/Isolated_Drugged'

analysisFolder_ctl_MM = base_path + r'/All_Controls/Catagorised_NPZs_redone/MM_below_-0.5'
analysisFolder_full_MM = base_path + r'/Long_Isolation/NPZ_catergorsied_redone/MM_below_-0.5'
analysisFolder_partial_MM = base_path + r'/48hrs_Isolation/NPZs_Catergorised_redone/MM_below_-0.5'

analysisFolder_ctl_PP = base_path + r'/All_Controls/Catagorised_NPZs_redone/PP_above_0.5'
analysisFolder_full_PP = base_path + r'/Long_Isolation/NPZ_catergorsied_redone/PP_above_0.5'
analysisFolder_partial_PP = base_path + r'/48hrs_Isolation/NPZs_Catergorised_redone/PP_above_0.5'

# Find all the npz files saved for each group and fish with all the information
npzFiles_ctl = glob.glob(analysisFolder_ctl + '/*.npz')
npzFiles_full = glob.glob(analysisFolder_full + '/*.npz')
npzFiles_partial = glob.glob(analysisFolder_partial + '/*.npz')
npzFiles_drug_30 = glob.glob(analysisFolder_drug_30 + '/*.npz')
npzFiles_drug_50 = glob.glob(analysisFolder_drug_50 + '/*.npz')
npzFiles_drug = npzFiles_drug_30 + npzFiles_drug_50

npzFiles_ctl_MM = glob.glob(analysisFolder_ctl_MM + '/*.npz')
npzFiles_full_MM = glob.glob(analysisFolder_full_MM + '/*.npz')
npzFiles_partial_MM = glob.glob(analysisFolder_partial_MM + '/*.npz')

npzFiles_ctl_PP = glob.glob(analysisFolder_ctl_PP + '/*.npz')
npzFiles_full_PP = glob.glob(analysisFolder_full_PP + '/*.npz')
npzFiles_partial_PP = glob.glob(analysisFolder_partial_PP + '/*.npz')

# LOAD CONTROLS
PTF_NS_CTL, PTF_S_CTL, dur_hist_NS_CTL, dur_hist_S_CTL = load_and_process_npz(npzFiles_ctl)

# LOAD FULL ISOLATION
PTF_NS_FULL, PTF_S_FULL, dur_hist_NS_FULL, dur_hist_S_FULL = load_and_process_npz(npzFiles_full)

# LOAD PARTIAL ISO
PTF_NS_PARTIAL, PTF_S_PARTIAL, dur_hist_NS_PARTIAL, dur_hist_S_PARTIAL = load_and_process_npz(npzFiles_partial)

# LOAD DRUGGED
PTF_NS_DRUG, PTF_S_DRUG, dur_hist_NS_DRUG, dur_hist_S_DRUG = load_and_process_npz(npzFiles_drug)

# LOAD CONTROLS, FULL, and PARTIAL (MM)
PTF_NS_CTL_MM, PTF_S_CTL_MM, dur_hist_NS_CTL_MM, dur_hist_S_CTL_MM = load_and_process_npz(npzFiles_ctl_MM)
PTF_NS_FULL_MM, PTF_S_FULL_MM, dur_hist_NS_FULL_MM, dur_hist_S_FULL_MM = load_and_process_npz(npzFiles_full_MM)
PTF_NS_PARTIAL_MM, PTF_S_PARTIAL_MM, dur_hist_NS_PARTIAL_MM, dur_hist_S_PARTIAL_MM = load_and_process_npz(npzFiles_partial_MM)

# LOAD CONTROLS, FULL, and PARTIAL (MM)
PTF_NS_CTL_PP, PTF_S_CTL_PP, dur_hist_NS_CTL_PP, dur_hist_S_CTL_PP = load_and_process_npz(npzFiles_ctl_PP)
PTF_NS_FULL_PP, PTF_S_FULL_PP, dur_hist_NS_FULL_PP, dur_hist_S_FULL_PP = load_and_process_npz(npzFiles_full_PP)
PTF_NS_PARTIAL_PP, PTF_S_PARTIAL_PP, dur_hist_NS_PARTIAL_PP, dur_hist_S_PARTIAL_PP = load_and_process_npz(npzFiles_partial_PP)

# Plots
# ----------------------------------

# Pause duration histograms
plt.figure(figsize=(10.24,7.68))
plt.title("Pause Duration Distribution")
centres = np.arange(0.1, 10, 0.2) # 1000 100 ms bins

plt.subplot(1,2,1)
plt.plot(centres, dur_hist_NS_CTL, 'k')
plt.plot(centres, dur_hist_NS_FULL, 'r')
plt.plot(centres, dur_hist_NS_PARTIAL, 'gray')
plt.plot(centres, dur_hist_NS_DRUG, 'g')
plt.yscale("log")
plt.xlabel('Relative Frequency')
plt.ylabel('Pause Duration (s)')

plt.subplot(1,2,2)
plt.plot(centres, dur_hist_S_CTL, 'k')
plt.plot(centres, dur_hist_S_FULL, 'r')
plt.plot(centres, dur_hist_S_PARTIAL, 'gray')
plt.plot(centres, dur_hist_S_DRUG, 'g')
plt.yscale("log")
plt.xlabel('Relative Frequency')
plt.ylabel('Pause Duration (s)')

plt.tight_layout() 
filename = figureFolder +'/Figure1_pause_durations.png'
plt.savefig(filename, dpi=600)
filename = figureFolder +'/Figure1_pause_durations.eps'
plt.savefig(filename, dpi=600)
plt.close('all')

# Percent time freezing Summary Plot (NS)
plt.figure(figsize=(10.24,7.68))
plt.title("Percent Time Freezing (NS)")

series_list = []
s = pd.Series(PTF_NS_CTL, name="S: " + 'WT')
series_list.append(s)
s = pd.Series(PTF_NS_FULL, name="S: " + 'Long Iso')
series_list.append(s)
s = pd.Series(PTF_NS_PARTIAL, name="S: " + 'Short Iso')
series_list.append(s)
s = pd.Series(PTF_NS_DRUG, name="S: " + 'Drug')
series_list.append(s)
sns.swarmplot(data=series_list, orient="v", size=6, color="#919395",  zorder=1)
with plt.rc_context({'lines.linewidth': 1.0}):
    sns.pointplot(data=series_list, orient="v", linewidth=2.1, ci=68, capsize=0.4, join=False, color='black', zorder=100)
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=18)
plt.xticks(np.arange(0, 4, step= 1), ('C', 'Fi','Pi', 'Pi+Drug'), fontsize=18)

plt.tight_layout() 
filename = figureFolder +'/Figure1_NS_ptf.png'
plt.savefig(filename, dpi=600)
filename = figureFolder +'/Figure1_NS_ptf.eps'
plt.savefig(filename, dpi=600)
plt.close('all')

# Percent time freezing Summary Plot (S)
plt.figure(figsize=(10.24,7.68))
plt.title("Percent Time Freezing (S)")

series_list = []
s = pd.Series(PTF_S_CTL, name="S: " + 'WT')
series_list.append(s)
s = pd.Series(PTF_S_FULL, name="S: " + 'Long Iso')
series_list.append(s)
s = pd.Series(PTF_S_PARTIAL, name="S: " + 'Short Iso')
series_list.append(s)
s = pd.Series(PTF_S_DRUG, name="S: " + 'Drug')
series_list.append(s)
sns.swarmplot(data=series_list, orient="v", size=6, color="#919395",  zorder=1)
with plt.rc_context({'lines.linewidth': 1.0}):
    sns.pointplot(data=series_list, orient="v", linewidth=2.1, ci=68, capsize=0.4, join=False, color='black', zorder=100)
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=18)
plt.xticks(np.arange(0, 4, step= 1), ('C', 'Fi','Pi', 'Pi+Drug'), fontsize=18)

plt.tight_layout() 
filename = figureFolder +'/Figure1_S_ptf.png'
plt.savefig(filename, dpi=600)
filename = figureFolder +'/Figure1_S_ptf.eps'
plt.savefig(filename, dpi=600)
plt.close('all')

# Percent time freezing Summary Plot (MM NS)
plt.figure(figsize=(10.24,7.68))
plt.title("Percent Time Freezing (MM NS)")

series_list = []
s = pd.Series(PTF_NS_CTL_MM, name="S: " + 'WT')
series_list.append(s)
s = pd.Series(PTF_NS_FULL_MM, name="S: " + 'Long Iso')
series_list.append(s)
s = pd.Series(PTF_NS_PARTIAL_MM, name="S: " + 'Short Iso')
series_list.append(s)
sns.swarmplot(data=series_list, orient="v", size=6, color="#204896",  zorder=1)
with plt.rc_context({'lines.linewidth': 1.0}):
    sns.pointplot(data=series_list, orient="v", linewidth=2.1, ci=68, capsize=0.4, join=False, color='black', zorder=100)
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=18)
plt.xticks(np.arange(0, 3, step= 1), ('C', 'Fi','Pi'), fontsize=18)

plt.tight_layout() 
filename = figureFolder +'/Figure1_NS_MM_ptf.png'
plt.savefig(filename, dpi=600)
filename = figureFolder +'/Figure1_NS_MM_ptf.eps'
plt.savefig(filename, dpi=600)
plt.close('all')

# Percent time freezing Summary Plot (MM S)
plt.figure(figsize=(10.24,7.68))
plt.title("Percent Time Freezing (MM S)")

series_list = []
s = pd.Series(PTF_S_CTL_MM, name="S: " + 'WT')
series_list.append(s)
s = pd.Series(PTF_S_FULL_MM, name="S: " + 'Long Iso')
series_list.append(s)
s = pd.Series(PTF_S_PARTIAL_MM, name="S: " + 'Short Iso')
series_list.append(s)
sns.swarmplot(data=series_list, orient="v", size=6, color="#204896",  zorder=1)
with plt.rc_context({'lines.linewidth': 1.0}):
    sns.pointplot(data=series_list, orient="v", linewidth=2.1, ci=68, capsize=0.4, join=False, color='black', zorder=100)
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=18)
plt.xticks(np.arange(0, 3, step= 1), ('C', 'Fi','Pi'), fontsize=18)

plt.tight_layout() 
filename = figureFolder +'/Figure1_S_MM_ptf.png'
plt.savefig(filename, dpi=600)
filename = figureFolder +'/Figure1_S_MM_ptf.eps'
plt.savefig(filename, dpi=600)
plt.close('all')

# Percent time freezing Summary Plot (PP NS)
plt.figure(figsize=(10.24,7.68))
plt.title("Percent Time Freezing (PP) NS)")

series_list = []
s = pd.Series(PTF_NS_CTL_PP, name="S: " + 'WT')
series_list.append(s)
s = pd.Series(PTF_NS_FULL_PP, name="S: " + 'Long Iso')
series_list.append(s)
s = pd.Series(PTF_NS_PARTIAL_PP, name="S: " + 'Short Iso')
series_list.append(s)
sns.swarmplot(data=series_list, orient="v", size=6, color="#E62928",  zorder=1)
with plt.rc_context({'lines.linewidth': 1.0}):
    sns.pointplot(data=series_list, orient="v", linewidth=2.1, ci=68, capsize=0.4, join=False, color='black', zorder=100)
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=18)
plt.xticks(np.arange(0, 3, step= 1), ('C', 'Fi','Pi'), fontsize=18)

plt.tight_layout() 
filename = figureFolder +'/Figure1_NS_PP_ptf.png'
plt.savefig(filename, dpi=600)
filename = figureFolder +'/Figure1_NS_PP_ptf.eps'
plt.savefig(filename, dpi=600)
plt.close('all')

# Percent time freezing Summary Plot (PP S)
plt.figure(figsize=(10.24,7.68))
plt.title("Percent Time Freezing (PP S)")

series_list = []
s = pd.Series(PTF_S_CTL_PP, name="S: " + 'WT')
series_list.append(s)
s = pd.Series(PTF_S_FULL_PP, name="S: " + 'Long Iso')
series_list.append(s)
s = pd.Series(PTF_S_PARTIAL_PP, name="S: " + 'Short Iso')
series_list.append(s)
sns.swarmplot(data=series_list, orient="v", size=6, color="#E62928",  zorder=1)
with plt.rc_context({'lines.linewidth': 1.0}):
    sns.pointplot(data=series_list, orient="v", linewidth=2.1, ci=68, capsize=0.4, join=False, color='black', zorder=100)
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=18)
plt.xticks(np.arange(0, 3, step= 1), ('C', 'Fi','Pi'), fontsize=18)

plt.tight_layout() 
filename = figureFolder +'/Figure1_S_PP_ptf.png'
plt.savefig(filename, dpi=600)
filename = figureFolder +'/Figure1_S_PP_ptf.eps'
plt.savefig(filename, dpi=600)
plt.close('all')

#-----------------------------------------------------------------------------
# Comparison statistics
#-----------------------------------------------------------------------------
report_path = figureFolder + '/report_Figure1_ptf.txt'
report_file = open(report_path, 'w')

do_stats(PTF_NS_CTL, "Controls NS", PTF_NS_FULL, "Full Isolation NS", report_file)
do_stats(PTF_NS_CTL, "Controls NS", PTF_NS_PARTIAL, "Partial Isolation NS", report_file)
do_stats(PTF_NS_CTL, "Controls NS", PTF_NS_DRUG, "Isolation Drug NS", report_file)
do_stats(PTF_S_CTL, "Controls S", PTF_S_FULL, "Full Isolation S", report_file)
do_stats(PTF_S_CTL, "Controls S", PTF_S_PARTIAL, "Partial Isolation S", report_file)
do_stats(PTF_S_CTL, "Controls S", PTF_S_DRUG, "Isolation Drug S", report_file)

do_stats(PTF_NS_CTL_MM, "Controls NS (MM)", PTF_NS_FULL_MM, "Full Isolation NS (MM)", report_file)
do_stats(PTF_NS_CTL_MM, "Controls NS (MM)", PTF_NS_PARTIAL_MM, "Partial Isolation NS (MM)", report_file)
do_stats(PTF_S_CTL_MM, "Controls S (MM)", PTF_S_FULL_MM, "Full Isolation S (MM)", report_file)
do_stats(PTF_S_CTL_MM, "Controls S (MM)", PTF_S_PARTIAL_MM, "Partial Isolation S (MM)", report_file)

do_stats(PTF_NS_CTL_PP, "Controls NS (PP)", PTF_NS_FULL_PP, "Full Isolation NS (PP)", report_file)
do_stats(PTF_NS_CTL_PP, "Controls NS (PP)", PTF_NS_PARTIAL_PP, "Partial Isolation NS (PP)", report_file)
do_stats(PTF_S_CTL_PP, "Controls S (PP)", PTF_S_FULL_PP, "Full Isolation S (PP)", report_file)
do_stats(PTF_S_CTL_PP, "Controls S (PP)", PTF_S_PARTIAL_PP, "Partial Isolation S (PP)", report_file)

report_file.close()

# FIN
