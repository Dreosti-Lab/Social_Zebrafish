# -*- coding: utf-8 -*-
"""
Quickly check the result of a social preference experiment
Created on Nov 10 2024
@author: dreostilab (Elena Dreosti)
"""
# This script generates:
# - the Bkg image needed for step1 analysis
# - ADDED: An image that plots the ROI social cue to check that ROIs are correct
# - A few summaruy images 

# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH') + "/../Behaviour/ED/libs"
base_path = os.getenv('BASE_PATH')

# Set Library Paths
import sys
sys.path.append(libs_path)

# Import useful libraries

# Import local modules
import SZ_utilities_ED as SZU
import SZ_video_ED as SZV
import BONSAI_ARK_ED
import glob

# Specify Folder List
#folderListFile = base_path + r'/FolderList/Grin2a_test.txt' 
#folderListFile = base_path + r'/FolderList/Gria3_test.txt' 
#analysisFolder = base_path + r'/grin2a/Analysis'
#analysisFolder = base_path + r'/gria3/Analysis
#folderListFile = base_path + r'/FolderList/h
# cn4_test.txt' 
#folderListFile = base_path + r'/FolderList/herc1_test.txt' 
#folderListFile = base_path + r'/FolderList/nr3c2_test.txt'
#folderListFile = base_path + r'/FolderList/Sp4_test.txt'
#folderListFile = base_path + r'/FolderList/trio_test.txt'
#folderListFile = base_path + r'/FolderList/Xpo7_test.txt'
folderListFile = base_path + r'/FolderList/Test.txt'


print (folderListFile)

# Set Flags
control = False

# Read Folder List
groups, ages, folderNames, fishStatus = SZU.read_folder_list(folderListFile)

# Load all the folders with experiments
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, C_folder = SZU.get_folder_names(folder)

    # Load S Stim Crop Regions
    bonsaiFiles = glob.glob(S_folder+'/Social_Fish/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    stim_ROIs = BONSAI_ARK_ED.read_bonsai_crop_rois(bonsaiFiles)
    S_stim_ROIs = stim_ROIs[:, :]
            
    # Process Video (NS)
    SZV.process_video_summary_images_modified(NS_folder, social=False)

    # Check if this is a control experiment
    if control:
        # Process Video (NS_2) - Control
        SZV.process_video_summary_images(C_folder, social=False)
    else:
        # Process Video (S)
        SZV.process_video_summary_images_modified(S_folder, social=True)
        

    # Report Progress
    print (groups[idx])
    
# #FIN
