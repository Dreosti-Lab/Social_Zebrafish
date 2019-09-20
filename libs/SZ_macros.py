# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:44:40 2013

@author: Adam
"""
# -----------------------------------------------------------------------------
# Detect Platform
import platform
if(platform.system() == 'Linux'):
    # Set "Repo Library Path" - Social Zebrafish Repo
    lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Zebrafish/libs'
else:
    # Set "Repo Library Path" - Social Zebrafish Repo
    lib_path = r'C:/Repos/Dreosti-Lab/Social_Zebrafish/libs'

# Set Library Paths
import sys
sys.path.append(lib_path)
# -----------------------------------------------------------------------------

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal
import scipy.misc as misc
import CV_ARK
import SZ_utilities as SZU
import BONSAI_ARK
import glob


# Utilities for loading and ploting "social zebrafish" data

# Analyze Single Fish Behaviour Experiment
def analyze_single_fish_experiment(dataFolder, analysisFolder, group, age, fishStatus, save):

    # Get Folder Names
    NS_folder, S_folder, D_folder = SZU.get_folder_names(dataFolder);
    
    # Load SocialSide Defintion
    filename = NS_folder + '\\SocialSide.txt'   
    NS_socialSide = np.genfromtxt(filename, delimiter=' ', skiprows=0)
    filename = S_folder + '\\SocialSide.txt'   
    S_socialSide = np.genfromtxt(filename, delimiter=' ', skiprows=0)
    
    # Compare SocialSide Defintions betwen NS and S
    comp = NS_socialSide - S_socialSide
    if (np.sum(np.abs(comp)) != 0):
        print ("Social Side defintion mismatch!!")
        print ('Folder:', dataFolder)
        socialSide = S_socialSide
    else:
        socialSide = S_socialSide
    
    # Specify Social Angle
    socialAngle = [270,90,270,90,270,90]
            
    # Analyze Each of the 6 Fish per experiment    
    for f in range(0,6):
        
        # Make Plot
        fig = plt.figure(figsize = (12,9), dpi = 150)
        fig.subplots_adjust(hspace=0.5)
   
        # Load Test Fish Tracking (Non_social control)
        filename = NS_folder + '\/tracking' + str(f+1) + '.csv'   
        test_tracking_ns = SZU.load_tracking(filename) #Tracking: X, Y, Ort, Major, Minor, Area
        trackingErrors_test_ns = SZU.measure_tracking_errors(test_tracking_ns)
        X_test_ns = test_tracking_ns[:, 0]
        Y_test_ns = test_tracking_ns[:, 1]
        Ort_test_ns = test_tracking_ns[:, 2]        
        # Quantify Social Position Bias (during Non-social condition)
        if socialSide[f] == 0:
            socialY_ns = Y_test_ns > 225
            non_socialY_ns = Y_test_ns <= 225
        else:
            socialY_ns = Y_test_ns < 225  
            non_socialY_ns = Y_test_ns >= 225
        # Relative Angle w.r.t. social position
        RelOrt_test_ns = Ort_test_ns-socialAngle[f]
        for i in range(0,np.size(RelOrt_test_ns)):
            if RelOrt_test_ns[i] < 0:
                RelOrt_test_ns[i] = RelOrt_test_ns[i]+360      
        ort_hist_test_ns, edges_test_ns = np.histogram(RelOrt_test_ns[socialY_ns], 36, (0, 360))
        
        # Load Test Fish Tracking (Social condition)
        filename = S_folder + '\/tracking' + str(f+1) + '.csv'   
        test_tracking_s = SZU.load_tracking(filename) #Tracking: X, Y, Ort, Major, Minor, Area
        trackingErrors_test_s = SZU.measure_tracking_errors(test_tracking_s)
        X_test_s = test_tracking_s[:, 0]
        Y_test_s = test_tracking_s[:, 1]
        Ort_test_s = test_tracking_s[:, 2]
        # Quantify Social Position Bias (during Social condition)
        if socialSide[f] == 0:
            socialY_s = Y_test_s > 225
            non_socialY_s = Y_test_s <= 225
        else:
            socialY_s = Y_test_s < 225  
            non_socialY_s = Y_test_s >= 225
        # Relative Angle w.r.t. social position
        RelOrt_test_s = Ort_test_s-socialAngle[f]
        for i in range(0,np.size(RelOrt_test_s)):
            if RelOrt_test_s[i] < 0:
                RelOrt_test_s[i] = RelOrt_test_s[i]+360
        ort_hist_test_s, edges_test_s = np.histogram(RelOrt_test_s[socialY_s], 36, (0, 360))
        
        # Load Stimulus Fish Tracking (Social condition)
        filename = S_folder + '\\Social_Fish\/tracking' + str(f+1) + '.csv'   
        stim_tracking_s = SZU.load_tracking(filename) #Tracking: X, Y, Ort, Major, Minor, Area
        trackingErrors_stim_s = SZU.measure_tracking_errors(stim_tracking_s)
        X_stim_s = stim_tracking_s[:, 0]
        Y_stim_s = stim_tracking_s[:, 1]
        Ort_stim_s = stim_tracking_s[:, 2]        
        # Relative Angle w.r.t. social position
        RelOrt_stim_s = Ort_stim_s-socialAngle[f]
        for i in range(0,np.size(RelOrt_stim_s)):
            if RelOrt_stim_s[i] < 0:
                RelOrt_stim_s[i] = RelOrt_stim_s[i]+360
        ort_hist_stim_s, edges_stim_s = np.histogram(RelOrt_stim_s[socialY_s], 36, (0, 360))
 
        # Compute Social Preference Index / Bias (Non-Social Condition)
        totalFrames = float(np.size(X_test_ns))
        socialFrames = float(np.sum(socialY_ns))
        nonSocialFrames = float(np.sum(non_socialY_ns))
        SPI_ns = ((socialFrames-nonSocialFrames)/totalFrames)
             
        # Compute Social Preference Index / Bias (Social Condition)
        totalFrames = float(np.size(X_test_s))
        socialFrames = float(np.sum(socialY_s))
        nonSocialFrames = float(np.sum(non_socialY_s))
        SPI_s = ((socialFrames-nonSocialFrames)/totalFrames)
        
        ## Compute Correlation (Social Condition, when on Social-Side)
        corrLength = 500
        motion_filter = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Determine Movement (Test Fish) - NS       
        Movement = SZU.motion_signal(X_test_ns, Y_test_ns, Ort_test_ns)
        #movement_test_ns = signal.fftconvolve(motion_filter, Movement)
        movement_test_ns = Movement
        movement_test_ns = movement_test_ns[socialY_ns]
        
        # Determine Movement (Test Fish) - S
        Movement = SZU.motion_signal(X_test_s, Y_test_s, Ort_test_s)
        #movement_test_s = signal.fftconvolve(motion_filter, Movement)
        movement_test_s = Movement
        movement_test_s = movement_test_s[socialY_s]
        
        # Determine Movement (Stimulus Fish) - S
        Movement = SZU.motion_signal(X_stim_s, Y_stim_s, Ort_stim_s)
        #movement_stim_s = signal.fftconvolve(motion_filter, Movement)
        movement_stim_s = Movement
        movement_stim_s = movement_stim_s[socialY_s]

        # Cross Correlate - Zero Pad the Test Array
        z = np.zeros(corrLength)
        zero_padded_test_ns = np.concatenate((z, movement_test_ns, z), axis = 0)
        zero_padded_test_s = np.concatenate((z, movement_test_s, z), axis = 0)
        zero_padded_stim_s = np.concatenate((z, movement_stim_s, z), axis = 0)
        zero_padded_test_s_rev = zero_padded_test_s[::-1] # Scramble/Reverse Test
        
        # Compute Reversed Correlation
        crcor = np.correlate(zero_padded_test_s, movement_stim_s, mode="valid")
        crcor_rev = np.correlate(zero_padded_test_s_rev, movement_stim_s, mode="valid")
        
        # Compute Auto-Correlations
        auto_corr_test_ns = np.correlate(zero_padded_test_ns, movement_test_ns, mode="valid")
        auto_corr_test_s = np.correlate(zero_padded_test_s, movement_test_s, mode="valid")
        auto_corr_stim_s = np.correlate(zero_padded_stim_s, movement_stim_s, mode="valid")
        
        
        
        ## Compute Burst-Triggered Average of Test and Stimulus Fish
        FPS = 100
        btaOffset = 100
        z = np.zeros(corrLength)
        
        # Extract the bouts data
        bouts_test_ns, numBouts_test_ns = SZU.extract_bouts_from_motion(movement_test_ns, 2.0)
        #boutFreq_test_s = FPS * (numBouts_test_s/totalFrames)
        bouts_test_s, numBouts_test_s = SZU.extract_bouts_from_motion(movement_test_s, 2.0)
        #boutFreq_test_s = FPS * (numBouts_test_s/totalFrames)
        bouts_stim_s, numBouts_stim_s = SZU.extract_bouts_from_motion(movement_stim_s, 2.0)
        #boutFreq_stim_s = FPS * (numBouts_stim_s/totalFrames)
        
        peaks_test_ns = bouts_test_ns[:, 1]
        peaks_test_s = bouts_test_s[:, 1]
        peaks_stim_s = bouts_stim_s[:, 1]
        
        
        zero_padded = np.concatenate((movement_test_s, z), axis = 0)
        aligned = SZU.burst_triggered_alignment(peaks_stim_s, zero_padded, btaOffset, corrLength)
        BTA_test_s = np.mean(aligned, axis = 0)
        aligned = SZU.burst_triggered_alignment(peaks_test_s, zero_padded, btaOffset, corrLength)
        BTA_self_test_s = np.mean(aligned, axis = 0)
        
        zero_padded = np.concatenate((movement_stim_s, z), axis = 0)
        aligned = SZU.burst_triggered_alignment(peaks_test_s, zero_padded, btaOffset, corrLength)
        BTA_stim_s = np.mean(aligned, axis = 0)             
        aligned = SZU.burst_triggered_alignment(peaks_stim_s, zero_padded, btaOffset, corrLength)
        BTA_self_stim_s = np.mean(aligned, axis = 0)
        
        # Plot Tracking
        # ---------------#---------------- # 
        # Plot Test Non-Social Tracking
        plt.subplot(3, 3, 1)
        plt.plot(X_test_ns[non_socialY_ns],Y_test_ns[non_socialY_ns], '.', markersize = 1, color = [0.0,0.0,0.0,0.05])
        plt.plot(X_test_ns[socialY_ns],Y_test_ns[socialY_ns], '.', markersize = 1, color = [1.0,0.0,0.0,0.05])
        plt.axis([0, 180, 0, 450])
        plt.title('Test Fish: Non-Social Condition\nSPI= %.2f' % SPI_ns, fontsize=8)
        ax = plt.gca()
        ax.invert_yaxis()
        plt.tick_params(labelsize = 6)

        # Plot Test Social Tracking
        plt.subplot(3, 3, 2)
        plt.plot(X_test_s[non_socialY_s],Y_test_s[non_socialY_s], '.', markersize = 1, color = [0.0,0.0,0.0,0.05])
        plt.plot(X_test_s[socialY_s],Y_test_s[socialY_s], '.', markersize = 1, color = [1.0,0.0,0.0,0.05])
        plt.axis([0, 180, 0, 450])
        plt.title('Test Fish: Social Condition\nSPI= %.2f' % SPI_s, fontsize=8)
        ax = plt.gca()
        ax.invert_yaxis()
        plt.tick_params(labelsize = 6)

        # Plot Stimulus Fish Social Tracking
        plt.subplot(3, 3, 3)
        plt.plot(X_stim_s,Y_stim_s, '.', markersize = 1, color = [0.0,0.0,0.0,0.05])
        plt.axis([0, 180, 0, 180])
        plt.title('Stimulus Fish: Social Condition\n', fontsize=8)
        ax = plt.gca()
        ax.invert_yaxis()
        plt.tick_params(labelsize = 6)
                         
        # Plot Orientation Histogram
        # ---------------#---------------- # 
        # Make Orientation Plot, correct for social fish position
        plt.subplot(3, 3, 4, projection='polar')      
        SZU.polar_orientation(RelOrt_test_ns[socialY_ns])
        plt.title('Orientation Histogram (social side)\n', fontsize=8)
        plt.tick_params(labelsize = 6)
                         
        # Make Orientation Plot, correct for social fish position
        plt.subplot(3, 3, 5, projection='polar')      
        SZU.polar_orientation(RelOrt_test_s[socialY_s])
        plt.title('Orientation Histogram (social side)\n', fontsize=8)
        plt.tick_params(labelsize = 6)
                         
        # Make Orientation Plot, correct for social fish position
        plt.subplot(3, 3, 6, projection='polar')      
        SZU.polar_orientation(RelOrt_stim_s[socialY_s])
        plt.title('Orientation Histogram (test fish on social side)\n', fontsize=8)           
        plt.tick_params(labelsize = 6)
                         
        # Plot Correlation and Burst-Triggered Averages
        # ------------------------#---------------------------- # 
        # Plot correlation and reversed correlation
        corrAxis = range(0, (corrLength*2)+1)
        corrAxis = np.array(corrAxis)-corrLength
        btaTaxis =  range(0, corrLength)
        btaTaxis =  np.array(btaTaxis)-btaOffset

        plt.subplot(3, 3, 8)
        plt.plot(corrAxis, crcor_rev, 'y')
        plt.plot(corrAxis, crcor, 'g')                         
        plt.title('Cross Correlation (test fish on social side)\n', fontsize=8)                    
        plt.tick_params(labelsize = 6)
        
        # Plot BTA of test fish movement triggered on stimulus fish "burst peaks"
        plt.subplot(3, 3, 7)
        plt.plot(btaTaxis, BTA_test_s, 'k')
        plt.title('Stimulus-Fish-Burst Triggered Average of Test Fish Motion\n', fontsize=8)                    
        plt.tick_params(labelsize = 6)
        
        # Plot BTA of stimulus fish movement triggered on test fish "burst peaks"
        plt.subplot(3, 3, 9)
        plt.plot(btaTaxis, BTA_stim_s, 'k')
        plt.title('Test-Fish-Burst Triggered Average of Stimulus Fish Motion\n', fontsize=8)                    
        plt.tick_params(labelsize = 6)

        # Save Figure and Data
        if save:
            # Prepare Summary Data
            orts = np.vstack((ort_hist_test_ns, ort_hist_test_s, ort_hist_stim_s))
            correlations = np.vstack((crcor, crcor_rev, auto_corr_test_ns, auto_corr_test_s, auto_corr_stim_s))
            peaks = [peaks_test_ns, peaks_test_s, peaks_stim_s]
            btas = np.vstack((BTA_test_s, BTA_stim_s, BTA_self_test_s, BTA_self_stim_s))


            filename = analysisFolder + '\\' + str(np.int(group)) + '_' + str(f) +'.png'  
            plt.savefig(filename, dpi=300)
            plt.close('all')

            # Save Correlation Data
            filename = analysisFolder + '\\' + str(np.int(group)) + '_' + str(f) +'.npz'
            summary = np.array([socialFrames, SPI_ns, SPI_s])
            np.savez(filename, summary, orts, correlations, btas, peaks)
        
        # End of Loop
    
        
    return 0


# Load Social Behaviour Folder
def correlate_social_responses_folder(folderName, plot, save):
    
    # Specifiy Folder
    filename = folderName + '\\SocialSide.txt'   
    socialSide = np.genfromtxt(filename, delimiter=' ', skiprows=0)
    socialAngle = [270,90,270,90,270,90]
    
    """ Summary Data Structure: 
    0 - Tracking Errors Social            
    1 - AvgSpeedSocial
    2 - MaxOrtSocial
    3 - BoutFreqSocial
    4 - Correlation 
    """    
    summaryData = np.zeros((6, 5))    
    corrLength = 500
    crcorrs = np.zeros((6,2*corrLength+1))


    if plot:
        plt.figure("CROSSCOR")
        plt.figure("BTA")
        plt.figure("BDIST")
                
    
    socialComparison = [1,2,3,4,5,6]
    for f in range(0,6):

        # Load Test Fish Tracking
        filename = folderName + '\/tracking' + str(f+1) + '.csv'   
        test_tracking = SZU.load_tracking(filename) #Tracking: X, Y, Ort, Major, Minor, Area
        trackingErrorsTest = SZU.measure_tracking_errors(test_tracking)
        # Extract Tracking Variables        
        X_test = test_tracking[:, 0]
        Y_test = test_tracking[:, 1]
        Ort_test = test_tracking[:, 2]

        # Load Social Fish Tracking
        filename = folderName + '\\Social_Fish\/tracking' + str(socialComparison[f]) + '.csv'   
        social_tracking = SZU.load_tracking(filename) #Tracking: X, Y, Ort, Major, Minor, Area
        trackingErrorsSocial = SZU.measure_tracking_errors(social_tracking)
        # Extract Tracking Variables        
        X_social = social_tracking[:, 0]
        Y_social = social_tracking[:, 1]
        Ort_social = social_tracking[:, 2]
 
        
        # Compute Useful Measurements (SOCIAL)
        speedXY = SZU.compute_speed(X_social, Y_social)
        avgSpeedSocial = np.mean(speedXY)
        
        
        # Relative Angle w.r.t. social position
        RelOrt = Ort_social-socialAngle[f]
        for i in range(0,np.size(RelOrt)):
            if RelOrt[i] < 0:
                RelOrt[i] = RelOrt[i]+360
        
        ort_hist, edges = np.histogram(RelOrt, 18, (0, 360))
        maxOrtSocial = edges[np.argmax(ort_hist)]
        
        
        # Determine when Test fish is in "Social Y Position"
        if socialSide[f] == 0:
            socialY = Y_test > 225
            non_socialY = Y_test < 225
        else:
            socialY = Y_test < 225  
            non_socialY = Y_test > 225

        totalFrames = float(np.size(X_test))
        socialFrames = float(np.sum(socialY))
        nonSocialFrames = float(np.sum(non_socialY))
        SPI = ((socialFrames-nonSocialFrames)/totalFrames)
        
        
        # Determine when Test fish can SEE (view) in "Social Fish"
        socialView = (RelOrt > 225)  + (RelOrt < 135)
        socialView = socialView * socialY


        # Determine Movement
        Movement_test = SZU.motion_signal(X_test, Y_test, Ort_test)
        motion_filter = np.array([0.25, 0.25, 0.25, 0.25])
        output = signal.fftconvolve(motion_filter, Movement_test)
        BinaryMovement_Test = output > 2.0
        BinaryMovement_Test = BinaryMovement_Test.astype(float)
        
        Movement_social = SZU.motion_signal(X_social, Y_social, Ort_social)
        motion_filter = np.array([0.25, 0.25, 0.25, 0.25])
        output = signal.fftconvolve(motion_filter, Movement_social)
        BinaryMovement_Social = output > 2.0
        BinaryMovement_Social = BinaryMovement_Social.astype(float)
        
        # Only accept trials on Social Side
#        Movement_test_valid = Movement_test[socialY]
#        Movement_social_valid = Movement_social[socialY]
#        BinaryMovement_Test = BinaryMovement_Test[socialY]
#        BinaryMovement_Social = BinaryMovement_Social[socialY]
        
        # Zero Mean
        #BinaryMovement_Test = BinaryMovement_Test/np.mean(BinaryMovement_Test)     
        #BinaryMovement_Social = BinaryMovement_Social/np.mean(BinaryMovement_Social)     
        
        
        # Cross Correlate - Zero Pad the Test Array
        z = np.zeros(corrLength)
        zero_padded = np.concatenate((z, BinaryMovement_Test, z), axis = 0)
        
        # Scramble/Reverse Test
        zero_padded_rev = zero_padded[::-1]
        
        # Social View
        zero_padded_view = np.concatenate((z, BinaryMovement_Test[socialView], z), axis = 0)
        
        crcor = np.correlate(zero_padded, BinaryMovement_Social, mode="valid")
        crcor_rev = np.correlate(zero_padded_rev, BinaryMovement_Social, mode="valid")
        crcor_view = np.correlate(zero_padded_view, BinaryMovement_Social[socialView], mode="valid")
        # Normalize by reversed crccor??


        # Quantify peakiness (central 51 (+/- 250 ms) vs outer 51)
        signalRange = range(corrLength-24, corrLength+27)
#        noiseRange1 = range(0, 50)
#        noiseRange2 = range((corrLength*2)-50, (corrLength*2))
        
#        noise = (np.std(crcor[noiseRange1]) + np.std(crcor[noiseRange2]))/2
        sig = np.std(crcor[signalRange])        
        noise = np.std(crcor_rev[signalRange])        
        corrVal = sig/noise        
        
        # Compute Synchrony : probabilty that they are moving at the same time vs. random chance
        #percentMoving_test = np.sum(BinaryMovement_Test) / totalFrames
        #percentMoving_social = np.sum(BinaryMovement_Social) / totalFrames
        
        #prob_random = percentMoving_test * percentMoving_social
        #prob_actual = np.sum(BinaryMovement_Test *  BinaryMovement_Social)  / totalFrames

        percentMoving_test = np.sum(BinaryMovement_Test[socialView]) / np.sum(socialView)
        percentMoving_social = np.sum(BinaryMovement_Social[socialView]) / np.sum(socialView)
        
        prob_random = percentMoving_test * percentMoving_social
        prob_actual = np.sum(BinaryMovement_Test[socialView] *  BinaryMovement_Social[socialView])  / np.sum(socialView)
        
        CorrReport = 'Corr: ' + str(round(corrVal,3)) + ' Rand: ' + str(round(prob_random*100, 2)) + '  Act: ' + str(round(prob_actual*100,2))
        
        # Extract Bouts (Social)
        FPS = 100
        bouts_social, numBouts = SZU.extract_bouts(X_social, Y_social, Ort_social, 2)
        boutFreqSocial = FPS * (numBouts/totalFrames)
        bouts_test, numBouts = SZU.extract_bouts(X_test, Y_test, Ort_test, 2)

        # Bout Triggered Average      
        starts_social = bouts_social[:, 0]
        zero_padded = np.concatenate((Movement_test, z), axis = 0)
        aligned = SZU.burst_triggered_alignment(starts_social, zero_padded, 100, corrLength)
        BTA_test = np.mean(aligned, axis = 0)

        starts_test = bouts_test[:, 0]
        zero_padded = np.concatenate((Movement_social, z), axis = 0)
        aligned = SZU.burst_triggered_alignment(starts_test, zero_padded, 100, corrLength)
        BTA_social = np.mean(aligned, axis = 0)                

        Distance_Social2Test = SZU.find_dist_to_nearest_index(starts_social, starts_test)
        Distance_Test2Social = SZU.find_dist_to_nearest_index(starts_test, starts_social)
        DSt_hist, DSt_edges = np.histogram(Distance_Social2Test, corrLength*2, (-corrLength,corrLength))
        DTs_hist, DTs_edges = np.histogram(Distance_Test2Social, corrLength*2, (-corrLength,corrLength))
    
        if (plot or save):
            # Make Position Plot
            plt.figure("CROSSCOR")
            plt.subplot(2, 3, f+1)            
            plt.plot(crcor_rev, 'r')
            plt.plot(crcor_view, 'g')
            plt.plot(crcor)
            plt.title(CorrReport)
            plt.figure("BTA")
            plt.subplot(2, 3, f+1)
            plt.plot(BTA_test, 'r')
            plt.plot(BTA_social, 'b')
            plt.figure("BDIST")
            plt.subplot(2, 3, f+1)
            plt.plot(DSt_edges[0:-1], DSt_hist, 'r')
            plt.plot(DTs_edges[0:-1], DTs_hist, 'b')
                
        
        # Add Summary Data    
        """ Summary Data Structure: 
        0 - Tracking Errors Social            
        1 - AvgSpeedSocial
        2 - MaxOrtSocial
        3 - BoutFreqSocial
        4 - Correlation 
        """                
        summaryData[f][0] =  trackingErrorsSocial   # Tracking Errors Social  
        summaryData[f][1] =  avgSpeedSocial         # AvgSpeedSocial
        summaryData[f][2] =  maxOrtSocial           # MaxOrtSocial
        summaryData[f][3] =  boutFreqSocial         # BoutFreqSocial
        summaryData[f][4] =  corrVal                # Correlation

        crcorrs[f,:] = crcor


        # End of Loop

    
    if save:
        # Check of Analysis Folder
        analysisFolder = folderName + '\\Analysis'  
        if not os.path.exists(analysisFolder):
            os.makedirs(analysisFolder)
    
        plt.figure("CROSSCOR")        
        filename = analysisFolder + '\\CROSSCOR.png'  
        plt.savefig(filename)
        plt.figure("BTA")        
        filename = analysisFolder + '\\BTA.png'  
        plt.savefig(filename)
        plt.figure("BDIST")        
        filename = analysisFolder + '\\BDIST.png'  
        plt.savefig(filename)
        
        
        plt.close('all')    

    return summaryData, crcorrs

# FIN

