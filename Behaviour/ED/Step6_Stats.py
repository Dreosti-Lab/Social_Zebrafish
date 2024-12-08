# -*- coding: utf-8 -*-
"""
Statistics of analyzed social preference experiments

@author: Elena Dreosti
"""
# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH') + "/../Behaviour/ED/libs"
base_path = os.getenv('BASE_PATH')


# Set Library Paths
import sys
sys.path.append(libs_path)


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
import SZ_utilities_ED as SZU
import SZ_macros_ED as SZM
import SZ_video_ED as SZV
import SZ_analysis_ED as SZA
import SZ_summary_ED as SZS
import BONSAI_ARK_ED
import glob
import pylab as pl

# -----------------------------------------------------------------------------
# Stats Helper Function
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------

# Set analysis folder and label for experiment/condition A
analysisFolder_A = base_path + r'/Akap11/Analysis_TPI'
conditionName_A = "A11"
# Set analysis folder and label for experiment/condition B
analysisFolder_B = base_path + r'/cacnag1g/Analysis_TPI'
conditionName_B = "c1g"
# Set analysis folder and label for experiment/condition C
analysisFolder_C = base_path + r'/gria3/Analysis_TPI'
conditionName_C = "G3"
# Set analysis folder and label for experiment/condition D
analysisFolder_D = base_path + r'/grin2a/Analysis_TPI'
conditionName_D = "G2a"
# Set analysis folder and label for experiment/condition E
analysisFolder_E = base_path + r'/hcn4/Analysis_TPI'
conditionName_E = "H4"
# Set analysis folder and label for experiment/condition F
analysisFolder_F = base_path + r'/herc1/Analysis_TPI'
conditionName_F = "H1"
# Set analysis folder and label for experiment/condition G
analysisFolder_G = base_path + r'/nr3c2/Analysis_TPI'
conditionName_G = "n3"
# Set analysis folder and label for experiment/condition H
analysisFolder_H = base_path + r'/Sp4/Analysis_TPI'
conditionName_H = "Sp4"
# Set analysis folder and label for experiment/condition I
analysisFolder_I = base_path + r'/trio/Analysis_TPI'
conditionName_I = "Trio"
# Set analysis folder and label for experiment/condition L
analysisFolder_L = base_path + r'/xpo7/Analysis_TPI'
conditionName_L = "X7"
# Set analysis folder and label for experiment/condition M
analysisFolder_M = base_path + r'/Scrambled/Analysis_TPI'
conditionName_M = "w_t"

# Assemble lists
analysisFolders = [analysisFolder_A, analysisFolder_B, analysisFolder_C, analysisFolder_D, analysisFolder_E, analysisFolder_F, analysisFolder_G, analysisFolder_H, analysisFolder_I, analysisFolder_L, analysisFolder_M]
conditionNames = [conditionName_A, conditionName_B, conditionName_C, conditionName_D, conditionName_E, conditionName_F, conditionName_G, conditionName_H, conditionName_I, conditionName_L, conditionName_M]

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
    
        # Freeze time threshold
        freeze_threshold = 500 # more than 5 seconds
        long_freeze_threshold = 24000 #More than 4 minutes

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
        Freezes_NS = np.array(np.sum(Pauses_NS[:,8] > freeze_threshold))
        Freezes_S = np.array(np.sum(Pauses_S[:,8] > freeze_threshold))
        Freezes_NS_ALL[f] = Freezes_NS
        Freezes_S_ALL[f] = Freezes_S
        
        # Count Long Freezes
        Long_Freezes_NS = np.array(np.sum(Pauses_NS[:,8] > long_freeze_threshold))
        Long_Freezes_S = np.array(np.sum(Pauses_S[:,8] > long_freeze_threshold))
        Long_Freezes_NS_ALL[f] = Long_Freezes_NS
        Long_Freezes_S_ALL[f] = Long_Freezes_S
        
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

#-----------------------------------------------------------------------------
# Comparison statistics
#-----------------------------------------------------------------------------
control_index = 10
for i, name in enumerate(conditionNames):
    print(f"\n\n{name}\n---------------")
    report_path = f"{analysisFolders[i]}/report_{name}.txt"
    report_file = open(report_path, 'w')
    do_stats(VPI_NS_summary[control_index], "VPI: Controls (NS)", VPI_NS_summary[i], f"VPI: {name} (NS)", report_file)
    do_stats(VPI_S_summary[control_index], "VPI: Controls (S)", VPI_S_summary[i], f"VPI: {name} (S)", report_file)

    do_stats(BPS_NS_summary[control_index], "BPS: Controls (NS)", BPS_NS_summary[i], f"BPS: {name} (NS)", report_file)
    do_stats(BPS_S_summary[control_index], "BPS: Controls (S)", BPS_S_summary[i], f"BPS: {name} (S)", report_file)

    do_stats(Distance_NS_summary[control_index], "Distance: Controls (NS)", Distance_NS_summary[i], f"Distance: {name} (NS)", report_file)
    do_stats(Distance_S_summary[control_index], "Distance: Controls (S)", Distance_S_summary[i], f"Distance: {name} (S)", report_file)

    do_stats(Freezes_NS_summary[control_index], "Freezes: Controls (NS)", Freezes_NS_summary[i], f"Freezes: {name} (NS)", report_file)
    do_stats(Freezes_S_summary[control_index], "Freezes: Controls (S)", Freezes_S_summary[i], f"Freezes: {name} (S)", report_file)

    do_stats(Percent_Moving_NS_summary[control_index], "Percent Moving: Controls (NS)", Percent_Moving_NS_summary[i], f"Percent Moving: {name} (NS)", report_file)
    do_stats(Percent_Moving_S_summary[control_index], "Percent Moving: Controls (S)", Percent_Moving_S_summary[i], f"Percent Moving: {name} (S)", report_file)

    do_stats(Long_Freezes_NS_summary[control_index], "Long Freezes: Controls (NS)", Long_Freezes_NS_summary[i], f"Long Freezes: {name} (NS)", report_file)
    do_stats(Long_Freezes_S_summary[control_index], "Long Freezes: Controls (S)", Long_Freezes_S_summary[i], f"Long Freezes: {name} (S)", report_file)
    report_file.close()

#FIN
