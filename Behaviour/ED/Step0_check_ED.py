# -*- coding: utf-8 -*-
"""
Quickly check the result of a social preference experiment

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

# Import useful libraries

# Import local modules
import SZ_utilities as SZU
# import SZ_macros as SZM
import SZ_video as SZV

# Specify Folder List
folderListFile = base_path + r'/FolderList/Akap11_test.txt' 
analysisFolder = base_path + r'/Akap11/Analysis'

# Set Flags
control = False

# Read Folder List
groups, ages, folderNames, fishStatus = SZU.read_folder_list(folderListFile)

# Load all the folders with experiments
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, C_folder = SZU.get_folder_names(folder)
            
    # Process Video (NS)
    SZV.process_video_summary_images(NS_folder, social=False)

    # Check if this is a control experiment
    if control:
        # Process Video (NS_2) - Control
        SZV.process_video_summary_images(C_folder, social=False)
    else:
        # Process Video (S)
        SZV.process_video_summary_images(S_folder, social=True)
        
    # Report Progress
    print (groups[idx])
    
# #FIN
