# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:24:46 2014

@author: dreostilab (Elena Dreosti)
"""
#==============================================================================
# This function is used to make the sumamry image and to check that all the 
#folders contain the bonsai ROi files
#==============================================================================


# -----------------------------------------------------------------------------
# 1) Function to find Dropbox Folder Path on each computer. In this way you 
# can keep files in Dropbox and do analyisis with different computers.

import os
import json

# Find Dropbox Path
try:
    appdata_path = os.getenv('APPDATA')
    with open(appdata_path + '\Dropbox\info.json') as data_file:
        data = json.load(data_file)
except:
    local_appdata_path = os.getenv('LOCALAPPDATA')
    with open(local_appdata_path + '\Dropbox\info.json') as data_file:
        data = json.load(data_file)
dropbox_path = data['personal']['path']

# -----------------------------------------------------------------------------
# Set Base Path Aas the shared Dropbox Folder (unique to each computer)
base_path = dropbox_path 
# Use this in case the dropbox doesn't work (base_path=r'C:\Users\elenadreosti
#\Dropbox (Personal)'
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# 2) Set Library Paths
import sys
sys.path.append(base_path + r'\Python_ED\Libraries')


# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats

# Import local modules
import SZ_utilities as SBU
import SZ_macros as SZM
import SZ_video as SZV

# 3) Read Folder List. The folder that contains all the info

folderListFile = (base_path + r'\Python_ED\Folder_List')
folderListFile = folderListFile + r'\SocialFolderList_2017_05_25_Condition_A_all.txt'

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
    
