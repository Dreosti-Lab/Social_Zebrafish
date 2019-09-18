# -*- coding: utf-8 -*-
"""
Measure SPI in time bins across assay duration

@author: dreostilab (Adam Kampff)
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Zebrafish/libs'
#-----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Set "Base Path" for this analysis session
base_path = r'/home/kampff/Data/Zebrafish'
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

# -----------------------------------------------------------------------------
# Function to load all summary statistics and make a summary figure
# -----------------------------------------------------------------------------
npzFiles = []
# Analysis folder
#analysisFolder = base_path + r'/Alcohol_0.125%/Analysis_folder/Control_Controls'
#analysisFolder = base_path + r'/Alcohol_0.125%/Analysis_folder/Control_Drugged'
analysisFolder = base_path + r'/Alcohol_0.125%/Analysis_folder/Isolated_Controls'
#analysisFolder = base_path + r'/Alcohol_0.125%/Analysis_folder/Isolated_Drugged'
#npzFiles.extend(glob.glob(analysisFolder+'/*.npz'))

#analysisFolder = base_path + r'/Robustness/Analysis_folder/Combined_Morning'
#analysisFolder = base_path + r'/Robustness/Analysis_folder/Combined_Afternoon'

#analysisFolder = base_path + r'/Buspirone/5uM/Analysis_Folder/Combined_Control_Controls'
#analysisFolder = base_path + r'/Buspirone/5uM/Analysis_Folder/Combined_Control_Drugged'
analysisFolder = base_path + r'/Buspirone/5uM/Analysis_Folder/Combined_Isolated_Controls'
#analysisFolder = base_path + r'/Buspirone/5uM/Analysis_Folder/Combined_Isolated_Drugged'
#npzFiles.extend(glob.glob(analysisFolder+'/*.npz'))

#analysisFolder = base_path + r'/Buspirone/10uM/Analysis_Folder/Combined_Control_Controls'
#analysisFolder = base_path + r'/Buspirone/10uM/Analysis_Folder/Combined_Control_Drugged'
analysisFolder = base_path + r'/Buspirone/10uM/Analysis_Folder/Combined_Isolated_Controls'
#analysisFolder = base_path + r'/Buspirone/10uM/Analysis_Folder/Combined_Isolated_Drugged'
#npzFiles.extend(glob.glob(analysisFolder+'/*.npz'))

#analysisFolder = base_path + r'/Buspirone/15uM/Analysis_Folder/Combined_Control_Controls'
#analysisFolder = base_path + r'/Buspirone/15uM/Analysis_Folder/Combined_Control_Drugged'
analysisFolder = base_path + r'/Buspirone/15uM/Analysis_Folder/Combined_Isolated_Controls'
#analysisFolder = base_path + r'/Buspirone/15uM/Analysis_Folder/Combined_Isolated_Drugged'
npzFiles.extend(glob.glob(analysisFolder+'/*.npz'))

# Calculate how many files
numFiles = np.size(npzFiles, 0)

# Create empty data structures
vis_vs_non_NS_ALL = np.zeros((numFiles,15))
vis_vs_non_S_ALL = np.zeros((numFiles,15))

# Go through all the files contained in the analysis folder
for f, filename in enumerate(npzFiles):

    #Load each npz file
    dataobject = np.load(filename)
    
    # Extract from the npz file
    VPI_NS = dataobject['VPI_NS']    
    VPI_S = dataobject['VPI_S']   
    Bouts_NS = dataobject['Bouts_NS']   
    Bouts_S = dataobject['Bouts_S']

    # Individual temporal bout analysis    
    print('VPI(S): %f' % (VPI_S))
    vis_vs_non_S_ALL[f, :] = SZS.analyze_temporal_bouts(Bouts_S, 6000)
    vis_vs_non_NS_ALL[f, :] = SZS.analyze_temporal_bouts(Bouts_NS, 6000)

# Display temporal bouts
plt.figure()
m = np.nanmean(vis_vs_non_S_ALL, 0)
std = np.nanstd(vis_vs_non_S_ALL, 0)
valid = (np.logical_not(np.isnan(vis_vs_non_S_ALL)))
n = np.sum(valid, 0)
se = std/np.sqrt(n)
plt.plot(vis_vs_non_S_ALL.T, LineWidth=1, Color=[0,0,0,0.1])
plt.plot(m, 'k', LineWidth=2)
plt.plot(m, 'ko', MarkerSize=5)
plt.plot(m+se, 'r', LineWidth=1)
plt.plot(m-se, 'r', LineWidth=1)
plt.axis([0, 15, -1.1, 1.1])
plt.show()

# FIN
