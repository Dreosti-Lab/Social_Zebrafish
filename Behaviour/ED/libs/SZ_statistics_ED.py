# -*- coding: utf-8 -*-
"""
Created on December 3 2024

@author: dreostilab (Elena Dreosti)
"""
# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
from scipy import stats
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
    # The function np.isnan returns a boolen where it si True if that numebr is Nan. 
    # So we then run the np.logical_not, which inverts the boolean array. Now you have true when there is a number that is NOT Na
    valid = (np.logical_not(np.isnan(S1_in))) 
    # Then you pass the boolean through S1_in so that that array only contains numbers and NOT Nan!
    S1 = S1_in[valid]

    valid = (np.logical_not(np.isnan(S2_in)))
    S2 = S2_in[valid]

    # The \n send the text to the next line. Themn it write down the names fo the arrays that are compared
    line = '\n' + name_1 + '(S1) vs ' + name_2 + '(S2)' 
    # Jump anotehr line. 
    report_file.write(line + '\n')
    print(line)

    # This calculate the MEAN of both arrays
    line = 'Mean (S1): ' + str(np.mean(S1)) + '\nMean (S2): ' + str(np.mean(S2)) 
    # Junm a line again
    report_file.write(line + '\n')
    print(line)

    # PARAMETRIC STATISTICS: Compare S1 vs. S2 (relative TTEST)
    result = stats.ttest_ind(S1, S2)
    line = "P-Value (S1 vs. S2):" + str(result[1]) + ' (Un-paired T-Test)'
    report_file.write(line + '\n')
    print(line)

    # NON-PARAMETRIC STATISTIC version of independent TTest
    result = stats.mannwhitneyu(S1, S2, True)
    line = "P-Value (S1 vs. S2):" + str(result[1]) + ' (Mann-Whitney U-Test)'
    report_file.write(line + '\n')
    print(line)

    return
# -----------------------------------------------------------------------------

