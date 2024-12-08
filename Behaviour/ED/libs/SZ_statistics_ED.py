# -*- coding: utf-8 -*-
"""
Created on December 3 2024

@author: dreostilab (Elena Dreosti)
"""
# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import glob
import cv2
import BONSAI_ARK_ED
import math
import matplotlib.patches as patches

# Utilities for performing statistics for Social Experiments
#Index Perform stats: do_stats: it calculates the mean, and performs TTest and mannwhitneyu test


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

