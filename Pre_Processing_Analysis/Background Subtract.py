# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:24:46 2014

Function that allows to check immediately if the tracking has worked or not 
right after the experiment. 
Before running select the folderlist text file containing the correst path 
where your video you want to check is saved. 

@author: Adam and Elena
"""
# -----------------------------------------------------------------------------
#==============================================================================
# -----------------------------------------------------------------------------
# 1) Function to find Dropbox Folder Path on each computer. In this way you 
# can keep files in Dropbox and do analyisis with different computers.

import os
import json

# Find Dropbox Path function taken from the internet
try:
    appdata_path = os.getenv('APPDATA')
    with open(appdata_path + '\Dropbox\info.json') as data_file:
        data = json.load(data_file)
except:
    local_appdata_path = os.getenv('LOCALAPPDATA')
    with open(local_appdata_path + '\Dropbox\info.json') as data_file:
        data = json.load(data_file)
dropbox_home = data['personal']['path']
#
#
## -----------------------------------------------------------------------------
## 2) Set Base Path (Shared Dropbox Folder)
base_path = dropbox_home()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Set Library Paths
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
import SZ_utilities as SZU
import SZ_macros as SZM
import SZ_video as SZV

# Read Folder List
folderListFile = base_path + r'\Adam_Ele\Shared Programming\Python\Social Zebrafish\PreProcessing'
folderListFile = folderListFile + r'\SocialFolderList_PreProcessing.txt'


dark = False
control = False
groups, ages, folderNames, fishStatus = SZU.read_folder_list(folderListFile)

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, D_folder, C_folder = SZU.get_folder_names(folder)

    # ----------------------
            
    # Process Video (NS)
    SZV.pre_process_video_summary_images(NS_folder, False)
    
    # Check if this is a control experiment
    if control:
        # Process Video (NS_2) - Control
        SZV.pre_process_video_summary_images(C_folder, False)

    else:
        # Process Video (S)
        SZV.pre_process_video_summary_images(S_folder, True)
    
        # Process Video (D-dark)
        if dark:
            SZV.pre_process_video_summary_images(D_folder, True)
        
    # Report Progress
 #print groups[idx]
    
# FIN
    
