# -*- coding: utf-8 -*-
"""
Analyze all tracked fish in a social preference experiment

@author: dreostilab (Elena Dreosti)
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
#lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Zebrafish/libs'
lib_path = r'C:/Repos/Dreosti-Lab/Social_Zebrafish/libs'
#-----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Set "Base Path" for this analysis session
#base_path = r'/home/kampff/Data/Zebrafish'
base_path = r'//128.40.155.187/data/D R E O S T I   L A B'
# -----------------------------------------------------------------------------

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

# Specify Folder List and Analysis Folder path
#folderListFile = base_path + r'/Analysis_folder/Control_Controls/All_Control_Controls.txt'
#analysisFolder = base_path + r'/Analysis_folder/Control_Controls'
folderListFile = base_path + r'/Analysis_folder/Isolated_Controls/All_Isolated_Controls.txt'
analysisFolder = base_path + r'/Analysis_folder/Isolated_Controls'
#folderListFile = base_path + r'/Analysis_folder/Isolated_Drugged_15/All_Isolated_Drugged_15.txt'
#analysisFolder = base_path + r'/Analysis_folder/Isolated_Drugged_15'

# Set Flags
plot = False
FPS = 120

# Set motion thresholds
motionStartThreshold = 0.03
motionStopThreshold = 0.01

# Read Folder List
groups, ages, folderNames, fishStatus = SZU.read_folder_list(folderListFile)

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):

    # Get Folder Names
    NS_folder, S_folder, C_folder = SZU.get_folder_names(folder)

    # Load NS Test Crop Regions
    bonsaiFiles = glob.glob(NS_folder+'/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    test_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    NS_test_ROIs = test_ROIs[:, :]
    
    # Load S Test Crop Regions
    bonsaiFiles = glob.glob(S_folder+'/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    test_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    S_test_ROIs = test_ROIs[:, :]

    # Load S Stim Crop Regions
    bonsaiFiles = glob.glob(S_folder+'/Social_Fish/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    stim_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    S_stim_ROIs = stim_ROIs[:, :]

    # Determine Fish Status       
    fishStat = fishStatus[idx, :]

    # ----------------------
    # Analyze and (maybe) plot each Fish
    for i in range(1,7):
        
        # Only use "good" fish
        if fishStat[i-1] == 1:
            
            # Plot Fish Data (maybe)
            if plot:
                # Make a Figure per each fish
                plt.figure(figsize=(10, 12), dpi=150)
        
            # Get X and Y coordinates of ROI Test and Stim ROIs extremes 
            x_min = min(NS_test_ROIs[i-1,0], S_stim_ROIs[i-1,0])
            y_min = min(NS_test_ROIs[i-1,1], S_stim_ROIs[i-1,1])
            x_max = max(NS_test_ROIs[i-1,0] + NS_test_ROIs[i-1,2], S_stim_ROIs[i-1,0] + S_stim_ROIs[i-1,2])
            y_max = max(NS_test_ROIs[i-1,1] + NS_test_ROIs[i-1,3], S_stim_ROIs[i-1,1] + S_stim_ROIs[i-1,3])
            
            # ----------------------
            # Analyze NS
            trackingFile = NS_folder + r'/tracking' + str(i) + '.npz'
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

            # Adjust orientation (0 is always facing the "stimulus" fish) - Depends on chamber
            ort = SZU.adjust_ort_test(ort, i)

            # Compute VPI (NS)
            VPI_ns, AllVisibleFrames, AllNonVisibleFrames, VPI_ns_bins = SZA.computeVPI(bx, by, NS_test_ROIs[i-1], S_stim_ROIs[i-1], FPS)

            # Compute SPI (NS)
            SPI_ns, AllSocialFrames_TF, AllNONSocialFrames_TF = SZA.computeSPI(bx, by, NS_test_ROIs[i-1], S_stim_ROIs[i-1])
            
            # Compute BPS (NS)
            BPS_ns, avgBout_ns = SZS.measure_BPS(motion, motionStartThreshold, motionStopThreshold)

            # Compute Distance Traveled (NS)
            Distance_ns = SZA.distance_traveled(bx, by, NS_test_ROIs[i-1])
        
            # Compute Orientation Histograms (NS)
            OrtHist_ns_NonSocialSide = SZS.ort_histogram(ort[AllNonVisibleFrames])
            OrtHist_ns_SocialSide = SZS.ort_histogram(ort[AllVisibleFrames])
            
            # Analyze "Bouts" amd "Pauses" (NS)
            Bouts_ns, Pauses_ns = SZS.analyze_bouts_and_pauses(tracking, NS_test_ROIs[i-1], S_stim_ROIs[i-1], AllVisibleFrames, motionStartThreshold, motionStopThreshold)
            Percent_Moving_ns = 100 * np.sum(Bouts_ns[:,8])/len(motion)
            Percent_Paused_ns = 100 * np.sum(Pauses_ns[:,8])/len(motion)

            # Plot NS (maybe)
            if plot:
                plt.subplot(5,2,1)
                plt.axis('off')
                plt.plot(bx, by, '.', markersize=1, color = [0.0, 0.0, 0.0, 0.05])
                plt.plot(bx[AllSocialFrames_TF], by[AllSocialFrames_TF], '.', markersize=1, color = [1.0, 0.0, 0.0, 0.05], )
                plt.plot(bx[AllNONSocialFrames_TF], by[AllNONSocialFrames_TF], '.', markersize=1, color = [0.0, 0.0, 1.0, 0.05])
                plt.title('SPI: ' + format(SPI_ns, '.3f') + ', VPI: ' + format(VPI_ns, '.3f'))
                plt.axis([x_min, x_max, y_min, y_max])
                plt.gca().invert_yaxis()

                plt.subplot(5,2,3)
                plt.title('BPS: ' + format(BPS_ns, '.3f') + ', %Paused: ' + format(Percent_Paused_ns, '.2f') + ', %Moving: ' + format(Percent_Moving_ns, '.2f'))
                motion[motion == -1.0] = -0.01
                plt.axhline(motionStartThreshold, c="green")
                plt.axhline(motionStopThreshold, c="red")
                plt.plot(motion)

                plt.subplot(5,2,5)
                motion[motion == -1.0] = -0.01
                plt.axhline(motionStartThreshold, c="green")
                plt.axhline(motionStopThreshold, c="red")
                plt.plot(motion[50000:51000])

                plt.subplot(5,2,7)
                plt.plot(avgBout_ns, 'k')
                
                plt.subplot(5,2,9)
                plt.plot(area, 'r')

            # ----------------------
            # Analyze S
            trackingFile = S_folder + r'/tracking' + str(i) + '.npz'    
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
            
             # Adjust orientation (0 is always facing the "stimulus" fish) - Depends on chamber
            ort = SZU.adjust_ort_test(ort, i)
            
            # Compute VPI (S)
            VPI_s, AllVisibleFrames, AllNonVisibleFrames, VPI_s_bins = SZA.computeVPI(bx, by, S_test_ROIs[i-1], S_stim_ROIs[i-1], FPS)

            # Compute SPI (S)
            SPI_s, AllSocialFrames_TF, AllNONSocialFrames_TF = SZA.computeSPI(bx, by, S_test_ROIs[i-1], S_stim_ROIs[i-1])
            
            # Compute BPS (S)
            BPS_s, avgBout_s = SZS.measure_BPS(motion, motionStartThreshold, motionStopThreshold)

            # Compute Distance Traveled (S)
            Distance_s = SZA.distance_traveled(bx, by, S_test_ROIs[i-1])
            
            # Compute Orientation Histograms (S)
            OrtHist_s_NonSocialSide = SZS.ort_histogram(ort[AllNonVisibleFrames])
            OrtHist_s_SocialSide = SZS.ort_histogram(ort[AllVisibleFrames])
            
            # Analyze "Bouts" amd "Pauses" (S)
            Bouts_s, Pauses_s = SZS.analyze_bouts_and_pauses(tracking, S_test_ROIs[i-1], S_stim_ROIs[i-1], AllVisibleFrames, motionStartThreshold, motionStopThreshold)
            Percent_Moving_s = 100 * np.sum(Bouts_s[:,8])/len(motion)
            Percent_Paused_s = 100 * np.sum(Pauses_s[:,8])/len(motion)
            
            # PLot S (maybe)
            if plot:
                plt.subplot(5,2,2)
                plt.axis('off')
                plt.plot(bx, by, '.', markersize=1, color = [0.0, 0.0, 0.0, 0.05])
                plt.plot(bx[AllSocialFrames_TF], by[AllSocialFrames_TF], '.', markersize=1, color = [1.0, 0.0, 0.0, 0.05], )
                plt.plot(bx[AllNONSocialFrames_TF], by[AllNONSocialFrames_TF], '.', markersize=1, color = [0.0, 0.0, 1.0, 0.05])
                plt.title('SPI: ' + format(SPI_s, '.3f') + ', VPI: ' + format(VPI_s, '.3f'))
                plt.axis([x_min, x_max, y_min, y_max])
                plt.gca().invert_yaxis()

                plt.subplot(5,2,4)
                plt.title('BPS: ' + format(BPS_s, '.3f') + ', %Paused: ' + format(Percent_Paused_s, '.2f') + ', %Moving: ' + format(Percent_Moving_s, '.2f'))
                motion[motion == -1.0] = -0.01
                plt.axhline(motionStartThreshold, c="green")
                plt.axhline(motionStopThreshold, c="red")
                plt.plot(motion)

                plt.subplot(5,2,6)
                motion[motion == -1.0] = -0.01
                plt.axhline(motionStartThreshold, c="green")
                plt.axhline(motionStopThreshold, c="red")
                plt.plot(motion[50000:51000])

                plt.subplot(5,2,8)
                plt.plot(avgBout_s, 'k')
                
                plt.subplot(5,2,10)
                plt.plot(area, 'r')
                

            # ----------------------
            # Analyze S_stim
            trackingFile = S_folder + r'/Social_Fish/tracking' + str(i) + '.npz'    
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
    
            # Plot Stim (maybe)
            if plot:
                plt.subplot(5,2,2)
                plt.plot(bx, by, '.', markersize=1, color = [1.0, 0.0, 0.0, 0.01])
                plt.plot(bx, by, '.', markersize=1, color = [0.0, 0.0, 1.0, 0.02])
                plt.plot(ex, ey, '.', markersize=1, color = [0.0, 1.0, 0.0, 0.02])
                plt.axis([x_min, x_max, y_min, y_max])
                plt.gca().invert_yaxis()
        
            #-----------------------------------
            # Save figure and data for each fish
            if plot:
                filename = analysisFolder + '/' + str(np.int(groups[idx])) + '_SPI_' + str(i) + '.png'  
                plt.savefig(filename, dpi=600)
                plt.close('all')

            #----------------------------
            # Save Analyzed Summary Data
            filename = analysisFolder + '/' + str(np.int(groups[idx])) + '_SUMMARY_' + str(i) + '.npz'
            np.savez(filename,
                     VPI_NS=VPI_ns,
                     VPI_NS_BINS=VPI_ns_bins,
                     VPI_S=VPI_s,
                     VPI_S_BINS=VPI_s_bins,
                     SPI_NS=SPI_ns, 
                     SPI_S=SPI_s,
                     BPS_NS=BPS_ns,
                     BPS_S=BPS_s,
                     Distance_NS = Distance_ns,
                     Distance_S = Distance_s,
                     OrtHist_NS_NonSocialSide = OrtHist_ns_NonSocialSide,
                     OrtHist_NS_SocialSide = OrtHist_ns_SocialSide,
                     OrtHist_S_NonSocialSide = OrtHist_s_NonSocialSide,
                     OrtHist_S_SocialSide = OrtHist_s_SocialSide,
                     Bouts_NS = Bouts_ns, 
                     Bouts_S = Bouts_s,
                     Pauses_NS = Pauses_ns,
                     Pauses_S = Pauses_s,
                     Percent_Moving_NS = Percent_Moving_ns,
                     Percent_Moving_S = Percent_Moving_s,
                     Percent_Paused_NS = Percent_Paused_ns,
                     Percent_Paused_S = Percent_Paused_s)
        else:
            print ("Bad Fish")
    
    # Report Porgress
    print (idx)
# End of Analysis Loop

#FIN
