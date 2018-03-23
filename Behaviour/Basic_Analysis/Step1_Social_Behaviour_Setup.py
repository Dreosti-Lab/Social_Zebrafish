# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:24:46 2014

@author: dreostilab (Elena Dreosti)
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'

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
import SZ_utilities as SBU
import SZ_macros as SZM
import SZ_video as SZV

# 3) Read Folder List. The folder that contains all the info

#folderListFile = (base_path + r'\Python_ED\Folder_List')
folderListFile = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Python_Analysis_Long_Isolation_New_Script3\Folder_List\SocialFolderList_PreProcessing_isolation_All_Isolated.txt'



#folderListFile = (base_path + r'\Python_ED\PreProcessing')
#folderListFile = folderListFile + r'\SocialFolderList_PreProcessing.txt'


groups, ages, folderNames, fishStatus = SBU.read_folder_list(folderListFile)

control= False
# 4) Load all the folders with experiments

for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, C_folder = SBU.get_folder_names(folder)
            
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
    
# FIN
    
