# -*- coding: utf-8 -*-
"""
Created on 2018 January

@author: dreostilab (Elena Dreosti)
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\Python_LIbraries'
#-----------------------------------------------------------------------------
# Set "Base Path" for this analysis session
base_path = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Python_Analysis_Adam'

# Set Library Paths
import sys
sys.path.append(lib_path)

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
import SZ_summary as SZS
import BONSAI_ARK
import glob

# Read Folder List
#folderListFile = base_path + r'\Folder_List\SocialFolderList_PreProcessing_isolation_All_Isolated.txt'
folderListFile = base_path + r'\Folder_List\SocialFolderList_PreProcessing_isolation_All_Controls.txt'

# Set Analysis Folder Path
#analysisFolder = base_path + r'\Analysis_Folder\Isolated_Summary'
analysisFolder = base_path + r'\Analysis_Folder\Controls_Summary'

# Plot Data
plot = False

# Load Folder list File
groups, ages, folderNames, fishStatus = SZU.read_folder_list(folderListFile)

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):

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
    # Analyze and (maybe) Plot each Fish
    for i in range(1,7):
        
        # Only use "good" fish
        if fishStat[i-1] == 1:
            
            # PLot Fish Data (maybe)
            if plot:
                # Make a Figure per each fish
                plt.figure() 
        
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
            
            # Determine "Visible" Frames
            AllVisibleFrames = SZA.analyze_tracking_VISIBLE(NS_folder, i, NS_test_ROIs, S_stim_ROIs)
            AllNonVisibleFrames = np.logical_not(AllVisibleFrames)

            # Adjust orientation (0 is always facing the "stimulus" fish) - Depends on chamber
            ort = SZU.adjust_ort_test(ort, i)
                        
            # Compute SPI (NS)
            SPI_ns, AllSocialFrames_TF, AllNONSocialFrames_TF = SZA.computeSPI(bx, by, NS_test_ROIs[i-1], S_stim_ROIs[i-1])
            
            # Compute BPS (NS)
            BPS_ns, avgBout_ns = SZS.measure_BPS(motion)
        
            # Compute Orientation Histograms (NS)
            OrtHist_ns_NonSocialSide = SZS.ort_histogram(ort[AllNonVisibleFrames])
            OrtHist_ns_SocialSide = SZS.ort_histogram(ort[AllVisibleFrames])
            
            # Compute "Pauses" (NS)
            IBI_ns = SZS.interBout_intervals(motion)
            pauses_ns = len(np.where(IBI_ns > 500)[0])

            # PLot NS (maybe)
            if plot:
                plt.subplot(3,2,1)
                plt.axis('off')
                plt.plot(bx, by, '.', markersize=1, color = [0.0, 0.0, 0.0, 0.05])
                plt.plot(bx[AllSocialFrames_TF], by[AllSocialFrames_TF], '.', markersize=1, color = [1.0, 0.0, 0.0, 0.05], )
                plt.plot(bx[AllNONSocialFrames_TF], by[AllNONSocialFrames_TF], '.', markersize=1, color = [0.0, 0.0, 1.0, 0.05])
                
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

            # Determine "Visible" Frames
            AllVisibleFrames = SZA.analyze_tracking_VISIBLE(S_folder, i, S_test_ROIs, S_stim_ROIs)
            AllNonVisibleFrames = np.logical_not(AllVisibleFrames)
            
            # Adjust orientation (0 is always facing the "stimulus" fish) - Depends on chamber
            ort = SZU.adjust_ort_test(ort, i)

            # Compute SPI (S)
            SPI_s, AllSocialFrames_TF, AllNONSocialFrames_TF = SZA.computeSPI(bx, by, S_test_ROIs[i-1], S_stim_ROIs[i-1])
            
            # Compute BPS (S)
            BPS_s, avgBout_s = SZS.measure_BPS(motion)
            
            # Compute Orientation Histograms (S)
            OrtHist_s_NonSocialSide = SZS.ort_histogram(ort[AllNonVisibleFrames])
            OrtHist_s_SocialSide = SZS.ort_histogram(ort[AllVisibleFrames])
            
            # Compute "Pauses" (S)
            IBI_s = SZS.interBout_intervals(motion)
            pauses_s = len(np.where(IBI_s > 500)[0])
                
            # PLot S (maybe)
            if plot:
                plt.subplot(3,2,2)
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
    
            # PLot Stim (maybe)
            if plot:
                plt.subplot(3,2,2)
                plt.plot(bx, by, '.', markersize=1, color = [1.0, 0.0, 0.0, 0.01])
                plt.plot(bx, by, '.', markersize=1, color = [0.0, 0.0, 1.0, 0.02])
                plt.plot(ex, ey, '.', markersize=1, color = [0.0, 1.0, 0.0, 0.02])
                plt.axis([x_min, x_max, y_min, y_max])
                plt.gca().invert_yaxis()
        
            #-----------------------------------
            # Save figure and data for each fish
            if plot:
                filename = analysisFolder + '\\' + str(np.int(groups[idx])) + '_SPI_' + str(i) + '.png'  
                plt.show()
                plt.savefig(filename, dpi=1200)
                plt.close('all')


            #----------------------------
            # Save Analyzed Summary Data
            filename = analysisFolder + '\\' + str(np.int(groups[idx])) + '_SUMMARY_' + str(i) + '.npz'
            np.savez(filename, 
                     SPI_NS=SPI_ns, 
                     SPI_S=SPI_s,
                     BPS_NS=BPS_ns,
                     BPS_S=BPS_s,
                     IBI_NS=IBI_ns,
                     IBI_S=IBI_s,
                     Pauses_NS = pauses_ns,
                     Pauses_S = pauses_s,
                     OrtHist_NS_NonSocialSide = OrtHist_ns_NonSocialSide,
                     OrtHist_NS_SocialSide = OrtHist_ns_SocialSide,
                     OrtHist_S_NonSocialSide = OrtHist_s_NonSocialSide,
                     OrtHist_S_SocialSide = OrtHist_s_SocialSide,
                     )    
        else:
            print ("Bad Fish")
    
    # Report Porgress
    print (idx)
        
# End of Analysis Loop

# FIN
