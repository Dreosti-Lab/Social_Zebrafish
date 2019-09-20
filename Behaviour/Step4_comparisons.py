# -*- coding: utf-8 -*-
"""
Compare summaries of analyzed social preference experiments

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

# Set analysis folder and label for experiment/condition A
analysisFolder_A = base_path + r'/Isolation_Experiments/Python_Analysis_Long_Isolation_New_Script3/Analysis_Folder/Controls/All'
conditionName_A = "Controls-All"

# Set analysis folder and label for experiment/condition B
analysisFolder_B = base_path + r'/Isolation_Experiments/Python_Analysis_Short_Isolation/Analysis_Folder/48h/All'
conditionName_B = "Isolated-All"

# Assemble lists
analysisFolders = [analysisFolder_A, analysisFolder_B]
conditionNames = [conditionName_A, conditionName_B]

# Summary Containers
VPI_NS_summary = []
VPI_S_summary = []
BPS_NS_summary = []
BPS_S_summary = []
Distance_NS_summary = []
Distance_S_summary = []
Freezes_NS_summary = []
Freezes_S_summary = []
Long_Freezes_NS_summary = []
Long_Freezes_S_summary = []
Percent_Moving_NS_summary = []
Percent_Moving_S_summary = []

# Go through each condition (analysis folder)
for i, analysisFolder in enumerate(analysisFolders):
    
    # Freeze time threshold
    freeze_threshold = 500 # more than 5 seconds
    Long_freeze_threshold = 24000 #More than 4 minutes
    
    # Find all the npz files saved for each group and fish with all the information
    npzFiles = glob.glob(analysisFolder+'/*.npz')
    
    # Calculate how many files
    numFiles = np.size(npzFiles, 0)

    # Allocate space for summary data
    VPI_NS_ALL = np.zeros(numFiles)
    VPI_S_ALL = np.zeros(numFiles)        
    BPS_NS_ALL = np.zeros(numFiles)
    BPS_S_ALL = np.zeros(numFiles)
    Distance_NS_ALL = np.zeros(numFiles)
    Distance_S_ALL = np.zeros(numFiles)    
    Freezes_NS_ALL = np.zeros(numFiles)
    Freezes_S_ALL = np.zeros(numFiles)
    Percent_Moving_NS_ALL = np.zeros(numFiles)
    Percent_Moving_S_ALL = np.zeros(numFiles)
    Long_Freezes_NS_ALL = np.zeros(numFiles)
    Long_Freezes_S_ALL = np.zeros(numFiles)
    
    # Go through all the files contained in the analysis folder
    for f, filename in enumerate(npzFiles):
    
        # Load each npz file
        dataobject = np.load(filename)
        
        # Extract from the npz file
        VPI_NS = dataobject['VPI_NS']    
        VPI_S = dataobject['VPI_S']   
        BPS_NS = dataobject['BPS_NS']   
        BPS_S = dataobject['BPS_S']
        Distance_NS = dataobject['Distance_NS']   
        Distance_S = dataobject['Distance_S']   
        Pauses_NS = dataobject['Pauses_NS']   
        Pauses_S = dataobject['Pauses_S']
        Percent_Moving_NS = dataobject['Percent_Moving_NS']   
        Percent_Moving_S = dataobject['Percent_Moving_S']

        # Count Freezes
        Freezes_NS_ALL[f] = np.sum(Pauses_NS[:,8] > freeze_threshold)
        Freezes_S_ALL[f] = np.sum(Pauses_S[:,8] > freeze_threshold)
        
        Long_Freezes_NS_ALL[f] = np.sum(Pauses_NS[:,8] > Long_freeze_threshold)
        Long_Freezes_S_ALL[f] = np.sum(Pauses_S[:,8] > Long_freeze_threshold)
        
        # Make an array with all summary stats
        VPI_NS_ALL[f] = VPI_NS
        VPI_S_ALL[f] = VPI_S
        BPS_NS_ALL[f] = BPS_NS
        BPS_S_ALL[f] = BPS_S
        Distance_NS_ALL[f] = Distance_NS
        Distance_S_ALL[f] = Distance_S
        Percent_Moving_NS_ALL[f] = Percent_Moving_NS
        Percent_Moving_S_ALL[f] = Percent_Moving_S
    
    # Add to summary lists
    VPI_NS_summary.append(VPI_NS_ALL)
    VPI_S_summary.append(VPI_S_ALL)
    
    BPS_NS_summary.append(BPS_NS_ALL)
    BPS_S_summary.append(BPS_S_ALL)
    
    Distance_NS_summary.append(Distance_NS_ALL)
    Distance_S_summary.append(Distance_S_ALL)
    
    Freezes_NS_summary.append(Freezes_NS_ALL)
    Freezes_S_summary.append(Freezes_S_ALL)

    Percent_Moving_NS_summary.append(Percent_Moving_NS_ALL)
    Percent_Moving_S_summary.append(Percent_Moving_S_ALL)
    
    Long_Freezes_NS_summary.append(Long_Freezes_NS_ALL)
    Long_Freezes_S_summary.append(Long_Freezes_S_ALL)

    # ----------------


#------------------------
# Summary plots
plt.figure()

# VPI
plt.subplot(2,3,1)
plt.title('VPI')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(VPI_NS_summary[i], name="NS: " + name)
    series_list.append(s)
    
for i, name in enumerate(conditionNames):
    s = pd.Series(VPI_S_summary[i], name="S: " + name)
    series_list.append(s)
df = pd.concat(series_list, axis=1)
sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], ci=95, capsize=0.05, errwidth=2)
sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray")

# BPS
plt.subplot(2,3,2)
plt.title('BPS')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(BPS_NS_summary[i], name="NS: " + name)
    series_list.append(s)
    
for i, name in enumerate(conditionNames):
    s = pd.Series(BPS_S_summary[i], name="S: " + name)
    series_list.append(s)
df = pd.concat(series_list, axis=1)
sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], ci=95, capsize=0.05, errwidth=2)
sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray")

# Distance
plt.subplot(2,3,3)
plt.title('Distance Traveled')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(Distance_NS_summary[i], name="NS: " + name)
    series_list.append(s)
    
for i, name in enumerate(conditionNames):
    s = pd.Series(Distance_S_summary[i], name="S: " + name)
    series_list.append(s)
df = pd.concat(series_list, axis=1)
sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], ci=95, capsize=0.05, errwidth=2)
sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray")

# Freezes
plt.subplot(2,3,4)
plt.title('Freezes')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(Freezes_NS_summary[i], name="NS: " + name)
    series_list.append(s)
    
for i, name in enumerate(conditionNames):
    s = pd.Series(Freezes_S_summary[i], name="S: " + name)
    series_list.append(s)
df = pd.concat(series_list, axis=1)
sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], ci=95, capsize=0.05, errwidth=2)
sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray")

# Percent Paused
plt.subplot(2,3,5)
plt.title('Percent Time Moving')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(Percent_Moving_NS_summary[i], name="NS: " + name)
    series_list.append(s)
    
for i, name in enumerate(conditionNames):
    s = pd.Series(Percent_Moving_S_summary[i], name="S: " + name)
    series_list.append(s)
df = pd.concat(series_list, axis=1)
sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], ci=95, capsize=0.05, errwidth=2)
sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray")

# Long Freezes
plt.subplot(2,3,6)
plt.title('Long Freezes')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(Long_Freezes_NS_summary[i], name="NS: " + name)
    series_list.append(s)
    
for i, name in enumerate(conditionNames):
    s = pd.Series(Long_Freezes_S_summary[i], name="S: " + name)
    series_list.append(s)
df = pd.concat(series_list, axis=1)
sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], ci=95, capsize=0.05, errwidth=2)
sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray")

#FIN
