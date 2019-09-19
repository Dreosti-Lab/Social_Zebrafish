# -*- coding: utf-8 -*-
"""
Quickly check the result of a social preference experiment

@author: dreostilab (Elena Dreosti)
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
import scipy.signal as signal
import CV_ARK
import scipy.misc as misc
from scipy import stats

# Import local modules
import SZ_utilities as SZU
import SZ_macros as SZM
import SZ_video as SZV

# Specify Folder List
folderListFile = base_path + r'/Python_ED/Folder_List'

# Set Flags
control= False

# Read Folder List
groups, ages, folderNames, fishStatus = SZU.read_folder_list(folderListFile)

# Load all the folders with experiments
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, C_folder = SZU.get_folder_names(folder)
            
    # Process Video (NS)
    SZV.process_video_summary_images(NS_folder, False)

    # Check if this is a control experiment
    if control:
        # Process Video (NS_2) - Control
        SZV.process_video_summary_images(C_folder, False)

    else:
        # Process Video (S)
        SZV.process_video_summary_images(S_folder, True)
        
    # Report Progress
    print (groups[idx])
    
#FIN
