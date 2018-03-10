# -*- coding: utf-8 -*-
"""
Created on 2017 March

@author: dreostilab (Elena Dreosti)
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'

# Set Library Paths
import sys
sys.path.append(lib_path)

# -----------------------------------------------------------------------------
# Set Base Path Aas the shared Dropbox Folder (unique to each computer)
base_path = dropbox_path 
# Use this in case the dropbox doesn't work (base_path=r'C:\Users\elenadreosti
#\Dropbox (Personal)'
#-----------------------------------------------------------------------------


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
import SZ_analysis as SZA
import BONSAI_ARK
import glob

# Read Folder List

# Read Folder List
#folderListFile = (base_path + r'\Python_ED\Social _Behaviour_Setup\Folder_Lists\Shank3AE2_BE5')
#folderListFile = folderListFile + r'\Shank3A_E2_3BE5.txt'


folderListFile = (base_path + r'\Python_ED\Folder_List')
folderListFile = folderListFile + r'\SocialFolderList_2017_05_25_Condition_A_all.txt'


# Set Analysis Folder Path
#analysisFolder = (base_path + r'\Python_ED\Social _Behaviour_Setup\Analysis_Folder\Shank3aE2_3bE5\Homozygous')

analysisFolder = (base_path + r'\Python_ED\Social _Behaviour_Setup\Analysis_Folder\C-fos')


# Plot Data
plot = True

# Load Folder list info
groups, ages, folderNames, fishStatus = SZU.read_folder_list(folderListFile)


# Make Empyt Arrays for SPI
NS_SPI = []
S_SPI = [] 

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
#for idx,folder in enumerate(folderNames[0:5]):

    # Get Folder Names
    NS_folder, S_folder, C_folder = SZU.get_folder_names(folder)
    

    # Load NS Test Crop Regions
    bonsaiFiles = glob.glob(NS_folder+'\*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    test_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    NS_test_ROIs = test_ROIs[:, :]
    
    # Load S Test Crop Regions
    bonsaiFiles = glob.glob(S_folder+'\*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    test_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    S_test_ROIs = test_ROIs[:, :]

    # Load S Stim Crop Regions
    bonsaiFiles = glob.glob(S_folder+'\Social_Fish\*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    stim_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    S_stim_ROIs = stim_ROIs[:, :]

    # Determine Fish Status       
    fishStat = fishStatus[idx, :]

        # ----------------------
    # Analyze and Plot each Fish
    for i in range(1,7):
        
        # Only use "good" fish
        if fishStat[i-1] == 1:
            
            # PLot Fish Data (maybe)
            if plot:
                # Make a Figure per each fish
                plt.figure() 
#                manager = plt.get_current_fig_manager()
#                manager.window.showMaximized()
        
            # Get X and Y coordinates of ROI Test and Stim ROIs extremes 
            x_min = min(NS_test_ROIs[i-1,0], S_stim_ROIs[i-1,0])
            y_min = min(NS_test_ROIs[i-1,1], S_stim_ROIs[i-1,1])
            x_max = max(NS_test_ROIs[i-1,0] + NS_test_ROIs[i-1,2], S_stim_ROIs[i-1,0] + S_stim_ROIs[i-1,2])
            y_max = max(NS_test_ROIs[i-1,1] + NS_test_ROIs[i-1,3], S_stim_ROIs[i-1,1] + S_stim_ROIs[i-1,3])
            
            
            # Filtering Tracking
            # - Area less than 20?

            # ----------------------
            # Analyze NS
            trackingFile = NS_folder + r'\tracking' + str(i) + '.npz'
            data = np.load(trackingFile)
            tracking = data['tracking']
            
            fx = tracking[:,0] 
            fy = tracking[:,1]
            bx = tracking[:,2]
            by = tracking[:,3]
            ex = tracking[:,4]
            ey = tracking[:,5]
            area = tracking[:,6]
            ort = tracking[:,7]
            motion = tracking[:,8]
            
            # Compute SPI (NS)
            SPI_ns, AllSocialFrames_TF, AllNONSocialFrames_TF = SZA.computeSPI(bx, by, NS_test_ROIs[i-1], S_stim_ROIs[i-1])
            NS_SPI.append(SPI_ns)
             
            # PLot NS (maybe)
            if plot:
                plt.subplot(3,2,1)
#                manager = plt.get_current_fig_manager()
#                manager.window.showMaximized()
                plt.axis('off')
                plt.plot(bx, by, '.', markersize=1, color = [0.0, 0.0, 0.0, 0.05])
                plt.plot(bx[AllSocialFrames_TF], by[AllSocialFrames_TF], '.', markersize=1, color = [1.0, 0.0, 0.0, 0.05], )
                plt.plot(bx[AllNONSocialFrames_TF], by[AllNONSocialFrames_TF], '.', markersize=1, color = [0.0, 0.0, 1.0, 0.05])
                
                
#                plt.plot(bx, by,  color = [0.5, 0.5, 0.5, 0.5])
#                plt.plot(bx[AllSocialFrames_TF], by[AllSocialFrames_TF],   color = [1.0, 0.0, 0.0, 0.5])
#                plt.plot(bx[AllNONSocialFrames_TF], by[AllNONSocialFrames_TF],   color = [0.0, 0.0, 1.0, 0.5])
                plt.axis([x_min, x_max, y_min, y_max])
                plt.gca().invert_yaxis()
                plt.subplot(3,2,3)
                motion[motion == -1.0] = -0.05
                plt.plot(motion)
                plt.subplot(3,2,5)
                plt.plot(area, 'r')

            # ----------------------
            # Analyze S
            trackingFile = S_folder + r'\tracking' + str(i) + '.npz'    
            data = np.load(trackingFile)
            tracking = data['tracking']
            
            fx = tracking[:,0]
            fy = tracking[:,1]
            bx = tracking[:,2]
            by = tracking[:,3]
            ex = tracking[:,4]
            ey = tracking[:,5]
            area = tracking[:,6]
            ort = tracking[:,7]
            motion = tracking[:,8]
            
            # Compute SPI (S)
            SPI_s, AllSocialFrames_TF, AllNONSocialFrames_TF = SZA.computeSPI(bx, by, S_test_ROIs[i-1], S_stim_ROIs[i-1])
            S_SPI.append(SPI_s)
               
            # PLot S (maybe)
            if plot:
                plt.subplot(3,2,2)
#                manager = plt.get_current_fig_manager()
#                manager.window.showMaximized()
                plt.axis('off')
                plt.plot(bx, by, '.', markersize=1, color = [0.0, 0.0, 0.0, 0.05])
                plt.plot(bx[AllSocialFrames_TF], by[AllSocialFrames_TF], '.', markersize=1, color = [1.0, 0.0, 0.0, 0.05], )
                plt.plot(bx[AllNONSocialFrames_TF], by[AllNONSocialFrames_TF], '.', markersize=1, color = [0.0, 0.0, 1.0, 0.05])
                plt.title([SPI_ns,SPI_s])
                plt.axis([x_min, x_max, y_min, y_max])
                plt.gca().invert_yaxis()
                plt.subplot(3,2,4)
                motion[motion == -1.0] = -0.05
                plt.plot(motion)
                plt.subplot(3,2,6)
                plt.plot(area, 'r')
                


            # ----------------------
            # Analyze S_stim
            trackingFile = S_folder + r'\Social_Fish\tracking' + str(i) + '.npz'    
            data = np.load(trackingFile)
            tracking = data['tracking']
            
            fx = tracking[:,0]
            fy = tracking[:,1]
            bx = tracking[:,2]
            by = tracking[:,3]
            ex = tracking[:,4]
            ey = tracking[:,5]
            area = tracking[:,6]
            ort = tracking[:,7]
            motion = tracking[:,8]
    
                        
            # PLot S (maybe)
            if plot:
                plt.subplot(3,2,2)
#                manager = plt.get_current_fig_manager()
#                manager.window.showMaximized()

                plt.plot(bx, by, '.', markersize=1, color = [1.0, 0.0, 0.0, 0.01])
                plt.plot(bx, by, '.', markersize=1, color = [0.0, 0.0, 1.0, 0.02])
                plt.plot(ex, ey, '.', markersize=1, color = [0.0, 1.0, 0.0, 0.02])
                plt.axis([x_min, x_max, y_min, y_max])
                plt.gca().invert_yaxis()
        
        
            # Save figure and data for each fish
            if plot:
                filename = analysisFolder + '\\' + str(np.int(groups[idx])) + '_B_' + str(i) +'.png'  
                plt.show()
                plt.savefig(filename, dpi=1200)
                plt.close('all')

            # Save Analyzed Data
            #filename = analysisFolder + '\\' + str(np.int(groups[idx])) + '_B_' + str(i) +'.npz'
            filename = analysisFolder + '\\' + str(np.int(groups[idx])) + '_B_' + str(i) +'.xls'
            np.savez(filename, SPI_NS=SPI_ns, SPI_S=SPI_s)
            # Here we will add functions to save more and more analyzed data!
        
        else:
            print ("Bad Fish")

    
    # Report Porgress
    print (idx)
    
    
# End of Analysis Loop

AllPref_NS = np.array(NS_SPI)
AllPref_S = np.array(S_SPI)

## Option 1)  Make histogram and plot it with lines 
#
#a_ns,c=np.histogram(AllPref_NS,  bins=10, range=(-1,1))
#a_s,c=np.histogram(AllPref_S,  bins=10, range=(-1,1))
#centers = (c[:-1]+c[1:])/2
#
##Normalize by tot number of fish
#Tot_Fish_NS=len(AllPref_NS)
#
#a_ns_float = np.float32(a_ns)
#a_s_float = np.float32(a_s)
#
#a_ns_nor_medium=a_ns_float/Tot_Fish_NS
#a_s_nor_medium=a_s_float/Tot_Fish_NS 
# 
#plt.figure()
#plt.plot(centers, a_ns_nor_medium, color=[0.5,0.5,0.5,0.6], linewidth=4.0)
#plt.plot(centers, a_s_nor_medium, color=[1.0,0.0,0.0,1], linewidth=4.0)
#plt.title('Non Social/Social PI', fontsize=20)
#plt.xlabel('Preference Index (PI_)', fontsize=20)
#plt.ylabel('Rel. Frequency', fontsize=20)
#plt.axis([-1, 1, 0, 0.4])
#pl.yticks([0, 0.1, 0.2, 0.3, 0.4], fontsize=20)
#pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=20)


# FIN
