# -*- coding: utf-8 -*-
"""
Track all the 6 fish in a social preference experiment
Created on Nov 10 2024
@author: dreostilab (Elena Dreosti)
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


# Import local modules
import SZ_utilities_ED as SZU
# import SZ_macros as SZM
import SZ_video_ED as SZV

# Import useful libraries
# import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import BONSAI_ARK_ED

# Specify Folder List
#folderListFile = base_path + r'/FolderList/Gria3_test.txt' 
#folderListFile = base_path + r'/FolderList/Grin2a_test.txt' 
#folderListFile = base_path + r'/FolderList/hcn4_test.txt' 
#folderListFile = base_path + r'/FolderList/herc1_test.txt'
#folderListFile = base_path + r'/FolderList/nr3c2_test.txt'
#folderListFile = base_path + r'/FolderList/Sp4_test.txt'
#folderListFile = base_path + r'/FolderList/trio_test.txt'
folderListFile = base_path + r'/FolderList/Xpo7_test.txt'


# Set Flags
dark = False
control = False  
multiple = False

# Read Folder List
groups, ages, folderNames, fishStatus = SZU.read_folder_list(folderListFile)
print(groups, ages, folderNames, fishStatus)

# Bulk tracking of all folders in Folder List
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names path
    NS_folder, S_folder, C_folder = SZU.get_folder_names(folder)
    Stimulus_folder = S_folder + '/Social_Fish' # Create path_name for the stimulus folder

    # ---------------------
    # Process Video (NS)
    bonsaiFiles = glob.glob(NS_folder + '/*.bonsai') #find the filepath of the file that contains the word "bonsai" and * anything else before that. AND make a list of all these filepaths
    bonsaiFiles = bonsaiFiles[0]  #Get the first file of the bonsaiFiles list [0]. We assume though there is only 1 file
    ROIs = BONSAI_ARK_ED.read_bonsai_crop_rois(bonsaiFiles) # It returmns a 6*6 array where rows are fish number (6) and columns are X,Y,width, higth. 
    #ROIs = ROIs[:, :]
    print('Processing Non-Social fish')
    fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SZV.improved_fish_tracking(NS_folder, NS_folder, ROIs)

    # Save Tracking (NS)
    for i in range(0,6):
        # Save NS
        filename = NS_folder + r'/tracking'+ str(i+1) + '.npz' #r'/tracking' uses a raw string to ensure any special characters (like backslashes) are handled correctly in the file path.
        fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
        np.savez(filename, tracking=fish.T) # we transpose the 2D array
    
#     # ---------------------
    # Process Video (S)
    bonsaiFiles = glob.glob(S_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    ROIs = BONSAI_ARK_ED.read_bonsai_crop_rois(bonsaiFiles)
    ROIs = ROIs[:, :]
    print('Processing Social fish')
    fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SZV.improved_fish_tracking(S_folder, S_folder, ROIs)

    # Save Tracking (S)
    for i in range(0,6):
        # Save S_test
        filename = S_folder + r'/tracking'+ str(i+1) + '.npz'
        fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
        np.savez(filename, tracking=fish.T)

    # ---------------------
    # Process Video (Stimulus)
    bonsaiFiles = glob.glob(Stimulus_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    ROIs = BONSAI_ARK_ED.read_bonsai_crop_rois(bonsaiFiles)
    ROIs = ROIs[:, :]
    fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SZV.improved_fish_tracking(S_folder, Stimulus_folder, ROIs)

    # Save Tracking (Stimulus)
    for i in range(0,6):
        # Save S_test
        filename = Stimulus_folder + r'/tracking'+ str(i+1) + '.npz'
        fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
        np.savez(filename, tracking=fish.T)
    
    # Close Plots
    plt.close('all')
    
#FIN
