# -*- coding: utf-8 -*-
"""
Analyze all tracked fish in a social preference experiment
Created on Nov 14 2024
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
import SZ_video_ED as SZV
import SZ_analysis_ED as SZA

import SZ_summary_ED as SZS
import BONSAI_ARK_ED
import pandas as pd
# import SZ_macros as SZM


# # Import useful libraries
# import os
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import scipy.misc as misc
# from scipy import stats
import seaborn as sns
import glob


# Specify Folder List and Analysis Folder path

# Specify Folder List
#folderListFile = base_path + r'/FolderList/Akap11_test.txt'  
folderListFile = base_path + r'/FolderList/hcn4_test.txt' 
#analysisFolder = base_path + r'/Akap11/Analysis'
analysisFolder = base_path + r'/hcn4/Analysis'
#analysisFolder = base_path + r'/herc1/Analysis'



# Set Flags
plot = True
FPS = 100
print (analysisFolder)

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
    print(bonsaiFiles)
    bonsaiFiles = bonsaiFiles[0]
    test_ROIs = BONSAI_ARK_ED.read_bonsai_crop_rois(bonsaiFiles)
    NS_test_ROIs = test_ROIs[:, :]
    
    # Load S Test Crop Regions
    bonsaiFiles = glob.glob(S_folder+'/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    test_ROIs = BONSAI_ARK_ED.read_bonsai_crop_rois(bonsaiFiles)
    S_test_ROIs = test_ROIs[:, :]

    # Load S Stim Crop Regions
    bonsaiFiles = glob.glob(S_folder+'/Social_Fish/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    stim_ROIs = BONSAI_ARK_ED.read_bonsai_crop_rois(bonsaiFiles)
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

            # Determine "social side" (if left = True, else right = False)
            test_x = NS_test_ROIs[i-1,0]
            stim_x = S_stim_ROIs[i-1,0]
            social_side_left = stim_x < test_x
            
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
            ort = SZU.adjust_ort_test(ort, social_side_left)

            # Compute VPI (NS)
            VPI_ns, AllVisibleFrames, AllNonVisibleFrames, VPI_ns_bins = SZA.computeVPI(bx, by, NS_test_ROIs[i-1], S_stim_ROIs[i-1], FPS)

            # Compute SPI (NS)
            SPI_ns, AllSocialFrames_TF, AllNONSocialFrames_TF = SZA.computeTPI(bx, by, NS_test_ROIs[i-1], S_stim_ROIs[i-1])
            
            # Compute BPS (NS)
            BPS_ns, avgBout_ns = SZS.measure_BPS(motion, motionStartThreshold, motionStopThreshold)

            # Compute Distance Traveled (NS)
            Distance_ns = SZA.distance_traveled(bx, by, NS_test_ROIs[i-1])
        
            # Compute Orientation Histograms (NS)
            OrtHist_ns_NonSocialSide = SZS.ort_histogram(ort[AllNONSocialFrames_TF])
            OrtHist_ns_SocialSide = SZS.ort_histogram(ort[AllSocialFrames_TF])

            # Analyze "Bouts" amd "Pauses" (NS)
            Bouts_ns, Pauses_ns = SZS.analyze_bouts_and_pauses(tracking, NS_test_ROIs[i-1], S_stim_ROIs[i-1], AllSocialFrames_TF, motionStartThreshold, motionStopThreshold)
            Percent_Moving_ns = 100 * np.sum(Bouts_ns[:,8])/len(motion)
            Percent_Paused_ns = 100 * np.sum(Pauses_ns[:,8])/len(motion)

            #Normalise bx and by to nx and ny
            nx, ny = SZA.normalized_arena_coords(bx, by, NS_test_ROIs[i-1], S_stim_ROIs[i-1])

            # Plot NS (maybe)
            if plot:
                plt.subplot(5,2,1)
                plt.axis('off')
                plt.plot(bx, by, '.', markersize=1, color = [0.0, 0.0, 0.0, 0.05])
                plt.plot(bx[AllSocialFrames_TF], by[AllSocialFrames_TF], '.', markersize=1, color = [1.0, 0.0, 0.0, 0.05] )
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
                #plt.plot(area[300:], 'r')

                plt.axis('off')
                plt.plot(nx, ny, '.', markersize=1, color = [0.0, 0.0, 0.0, 0.05])
                plt.plot(nx[AllSocialFrames_TF], ny[AllSocialFrames_TF], '.', markersize=1, color = [1.0, 0.0, 0.0, 0.05] )
                #plt.plot(nx[AllNONSocialFrames_TF], ny[AllNONSocialFrames_TF], '.', markersize=1, color = [0.0, 0.0, 1.0, 0.05])
                #plt.title('SPI: ' + format(SPI_ns, '.3f') + ', VPI: ' + format(VPI_ns, '.3f'))
                #plt.axis([x_min, x_max, y_min, y_max])
                plt.xlabel('bx by normalised')
                plt.gca().invert_yaxis()


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
            ort = SZU.adjust_ort_test(ort, social_side_left)
            
            # Compute VPI (S)
            VPI_s, AllVisibleFrames, AllNonVisibleFrames, VPI_s_bins = SZA.computeVPI(bx, by, S_test_ROIs[i-1], S_stim_ROIs[i-1], FPS)

            # Compute SPI (S)
            SPI_s, AllSocialFrames_TF, AllNONSocialFrames_TF = SZA.computeTPI(bx, by, S_test_ROIs[i-1], S_stim_ROIs[i-1])
            
            # Compute BPS (S)
            BPS_s, avgBout_s = SZS.measure_BPS(motion, motionStartThreshold, motionStopThreshold)

            # Compute Distance Traveled (S)
            Distance_s = SZA.distance_traveled(bx, by, S_test_ROIs[i-1])
            
            # Compute Orientation Histograms (S)
            OrtHist_s_NonSocialSide = SZS.ort_histogram(ort[AllNONSocialFrames_TF])
            OrtHist_s_SocialSide = SZS.ort_histogram(ort[AllSocialFrames_TF])
            xAxis = np.arange(-np.pi,np.pi+np.pi/18.0, np.pi/18.0)
            #print(xAxis)
            # plt.figure(figsize=(4,4))
            # plt.plot(xAxis, np.hstack((OrtHist_s_SocialSide, OrtHist_s_SocialSide[0])), linewidth = 3)
            # plt.show()           
            
            # Analyze "Bouts" amd "Pauses" (S)
            Bouts_s, Pauses_s = SZS.analyze_bouts_and_pauses(tracking, S_test_ROIs[i-1], S_stim_ROIs[i-1], AllSocialFrames_TF, motionStartThreshold, motionStopThreshold)
            Percent_Moving_s = 100 * np.sum(Bouts_s[:,8])/len(motion)
            Percent_Paused_s = 100 * np.sum(Pauses_s[:,8])/len(motion)

            #Normalise bx and by to nx and ny
            nx, ny = SZA.normalized_arena_coords(bx, by, S_test_ROIs[i-1], S_stim_ROIs[i-1])
                    
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
                #plt.plot(area[300:], 'r')

                plt.axis('off')
                plt.plot(nx, ny, '.', markersize=1, color = [0.0, 0.0, 0.0, 0.05])
                plt.plot(nx[AllSocialFrames_TF], ny[AllSocialFrames_TF], '.', markersize=1, color = [1.0, 0.0, 0.0, 0.05] )
                #plt.plot(nx[AllNONSocialFrames_TF], ny[AllNONSocialFrames_TF], '.', markersize=1, color = [0.0, 0.0, 1.0, 0.05])
                #plt.title('SPI: ' + format(SPI_ns, '.3f') + ', VPI: ' + format(VPI_ns, '.3f'))
                #plt.axis([x_min, x_max, y_min, y_max])
                plt.xlabel('bx by normalised')
                plt.gca().invert_yaxis()

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
                filename = analysisFolder + '/' + str(int(groups[idx])) + '_SPI_' + str(i) + '.png'  
                plt.savefig(filename, dpi=600)
                plt.close('all')

            #----------------------------
            # Save Analyzed Summary Data as npz files 
            filename = analysisFolder + '/' + str(int(groups[idx])) + '_SUMMARY_' + str(i) + '.npz'
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
