# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:59:42 2016

@author: dreostilab (Elena Dreosti)
"""
#==============================================================================
# This is the Algorithm used
    # 1. Subtract pre-computed Background frame from Current frame (Note: Not AbsDiff!)
    # 2. Extract Crop regions from ROIs
    # 3. Threshold ROI using mean/7 of each crop region, Binary Close image using 5 rad disc
    # 4. Find largest particle (Contour)
    # 5. - Compute Weighted Centroid (X,Y) for Eye Region (10% of brightest pixels)
    # 6. - Compute Binary Centroid of Body Region (50% of brightest pixels - eyeRegion)
    # 7. - Compute Heading

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
dropbox_path = data['personal']['path']
#
#
## -----------------------------------------------------------------------------
## 2) Set Base Path (Shared Dropbox Folder)
#base_path = dropbox_home()
base_path= dropbox_path
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# Set Library Paths
import sys
#sys.path.append(base_path + r'\Adam_Ele\Shared Programming\Python\ARK')
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

## Read Folder List
#folderListFile = base_path + r'\Adam_Ele\Shared Programming\Python\Social Zebrafish\Folder Lists\Dreosti_LAb\\Wt_Nacre'
#
#folderListFile = folderListFile + r'\Folder_List_Wt_Nacre.txt'

#folderListFile = (base_path + r'\Adam_Ele\Shared Programming\Python\Social Zebrafish\Folder Lists\Dreosti_LAb\Wt_Nacre')
#folderListFile = folderListFile + r'\Folder_List_Wt_Nacre_test.txt'

#folderListFile = (base_path + r'\Adam_Ele\Shared Programming\Python\Social Zebrafish\Folder Lists\Isolation_experiments')
#folderListFile = folderListFile + r'\Folder_List_Isolation_test.txt'


folderListFile = (base_path + r'\Python_ED\Folder_List')
folderListFile = folderListFile + r'\SocialFolderList_2017_05_25_Condition_A.txt'



dark = False
control = False  
multiple = False
groups, ages, folderNames, fishStatus = SZU.read_folder_list(folderListFile)

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, C_folder = SZU.get_folder_names(folder)

    # ---------------------
    # Process Video (NS)
    fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SZV.process_video_track_fish(NS_folder, False, False )

    # Save Tracking (NS)
    for i in range(0,6):
        # Save NS
        filename = NS_folder + r'\tracking'+ str(i+1) + '.npz'
        fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
        np.savez(filename, tracking=fish.T)
    
    # ---------------------
    # Check if this is a control experiment
    if control:    
        # Process Video (NS)
        fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SZV.process_video_track_fish(C_folder, False, False )
    
        # Save Tracking (NS)
        for i in range(0,6):
            # Save NS
            filename = C_folder + r'\tracking'+ str(i+1) + '.npz'
            fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
            np.savez(filename, tracking=fish.T)
    
    else:
        # ----------------------
        # Process Video (S)
        fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS, fxS_s, fyS_s, bxS_s, byS_s, exS_s, eyS_s, areaS_s, ortS_s, motS_s = SZV.process_video_track_fish(S_folder, True, multiple)
    
        # Save Tracking (S)
        for i in range(0,6):
            # Save S_test
            filename = S_folder + r'\tracking'+ str(i+1) + '.npz'
            fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
            np.savez(filename, tracking=fish.T)
    
            # Save S_social
            filename = S_folder + r'\Social_Fish\tracking'+ str(i+1) + '.npz'
            fish = np.vstack((fxS_s[:,i], fyS_s[:,i], bxS_s[:,i], byS_s[:,i], exS_s[:,i], eyS_s[:,i], areaS_s[:,i], ortS_s[:,i], motS_s[:,i]))
            np.savez(filename, tracking=fish.T)
    
        # ----------------------
#        # Process Video (D-dark)
#        if D_folder != -1:
#            # Process Video (D)
#            fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SZV.process_video_track_fish(D_folder, True, multiple)
#    
#            # Save Tracking (D)
#            for i in range(0,6):
#                # Save D_test
#                filename = D_folder + r'\tracking'+ str(i+1) + '.npz'
#                fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
#                np.savez(filename, tracking=fish.T)
#        
#                # Save D_social
#                filename = D_folder + r'\Social_Fish\tracking'+ str(i+1) + '.npz'
##                fish = np.vstack((fxS_s[:,i], fyS_s[:,i], bxS_s[:,i], byS_s[:,i], exS_s[:,i], eyS_s[:,i], areaS_s[:,i], ortS_s[:,i], motS_s[:,i]))
#                np.savez(filename, tracking=fish.T)

    # Close Plots
    plt.close('all')
    
# End of Tracking    



