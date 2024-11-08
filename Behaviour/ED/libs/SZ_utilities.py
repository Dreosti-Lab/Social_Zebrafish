# -*- coding: utf-8 -*-
"""
Created on 6 Nov 2024

@author: dreostilab (Elena Dreosti)
"""

# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH') + "/../Behaviour/ED/libs"
base_path = os.getenv('BASE_PATH')


# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal



#-----------------------------------------------------------------------------
# Utilities for loading and ploting "social zebrafish" data

# 1) Read Folder List file 6 fish 
def read_folder_list(folderListFile): 
    folderFile = open(folderListFile, "r") #"r" means read the file
    folderList = folderFile.readlines() # returns a list containing the lines

    # Set Data Path where the experiments are located
    data_path = base_path
    
    numFolders = len(folderList) 
    groups = np.zeros(numFolders)
    ages = np.zeros(numFolders)
    folderNames = [] # We use a LIST instead of a numoy array becasue we do not know the exact length
    fishStatus = np.zeros((numFolders, 6))
    
    for i, f in enumerate(folderList):  #enumerate tells you what folder is 'i'
        stringLine = f[:-1].split()
        groups[i] = int(stringLine[0])
        ages[i] = int(stringLine[1])
        expFolderName = data_path + stringLine[2]
        folderNames.append(expFolderName)
        fishStat = [int(stringLine[3]), int(stringLine[4]), int(stringLine[5]), int(stringLine[6]), int(stringLine[7]), int(stringLine[8])]    
        fishStatus[i,:] = np.array(fishStat)
        
    return groups, ages, folderNames, fishStatus
    

# FIN


# 2) Determine Data Folder Names from Root directory
def get_folder_names(folder):
    # Specifiy Folder Names
    NS_folder = folder + '/Non_Social_1'
    if os.path.exists(NS_folder) == False:
        print(f"The folder Non_Social_1 doesn't exist in {folder}")
        NS_folder = -1

    S_folder = folder + '/Social_1'
    if os.path.exists(S_folder) == False:
        print(f"The folder Social_1 doesn't exist in {folder}")
        S_folder = -1
    
    C_folder = folder + '/Non_Social_2'
    if os.path.exists(C_folder) == False:
        C_folder = -1    
    
    return NS_folder, S_folder, C_folder