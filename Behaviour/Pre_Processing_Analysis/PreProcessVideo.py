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
# -----------------------------------------------------------------------------
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
#\Dropbox (Personal)'--------------------------------------------

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
import SZ_macros as SZM
import SZ_video as SZV
import SZ_utilities as SZU

# Read Folder List
folderListFile = base_path + r'\Python_ED\Social _Behaviour_Setup\PreProcessing'
folderListFile = folderListFile + r'\SocialFolderList_PreProcessing_2017_08_25_subset.txt'

control = False
groups, ages, folderNames, fishStatus = SZU.read_folder_list(folderListFile)

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, C_folder = SZU.get_folder_names(folder)

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
#    print (cv2.__version__) 
        # Process Video (D-dark)
       
    # Report Progress
 #print groups[idx]
    
# FIN
    