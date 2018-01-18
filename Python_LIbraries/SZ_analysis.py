# -*- coding: utf-8 -*-
"""
Created on Wed May 07 19:13:12 2014

@author: Elena
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\Python_LIbraries'

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal
import CV_ARK # it is like you have all the functions here in this file
import SZ_utilities as SZU

# Functions for analyzing "social zebrafish" data

# Compute Social Preference Index
def computeSPI(xPositions, yPositions, testROI, stimROI):
    
    # Find thresholds of X and Y Test ROI in order to define the social area to calculate the SPI
    socialPositionThreshold_X = testROI[0]+(testROI[2]/2)
    socialPositionThreshold_Y = testROI[1]+(testROI[3]/2) 
    
    # Define which frames are "social" in X and Y by comparing Tracking X and Y with the above calculated threshold
    socialTop = stimROI[1]<socialPositionThreshold_Y
    socialLeft = stimROI[0]<socialPositionThreshold_X
        
    # Compute Social Frames (depending on where the stimulus fish is (top or bottom))
    
   # Check X threshold 
    if socialLeft:
        
        AllSocialFrames_X_TF= xPositions<socialPositionThreshold_X    # True/False array
        
    else:
       AllSocialFrames_X_TF= xPositions>socialPositionThreshold_X   # Opposite True/False array
    
    # Check Y threshold 
    if socialTop:
        
        AllSocialFrames_Y_TF= yPositions<socialPositionThreshold_Y   # True/False array, BUT "Zero Y" remember is located conventionally at the TOP
        
           
    else:
        AllSocialFrames_Y_TF= yPositions>socialPositionThreshold_Y   # Opposite True/False array
    
        
    AllSocialFrames_TF= np.logical_and(AllSocialFrames_X_TF,AllSocialFrames_Y_TF)  # Final SOCIAL True/False array
    AllNONSocialFrames_TF=np.logical_and(AllSocialFrames_X_TF, np.logical_not(AllSocialFrames_Y_TF))   # Final NON SOCIAL True/False array
    
    # Count Social and Non-Social Frames
    numSocialFrames = np.float(np.sum(AllSocialFrames_TF))     # Final Sum of Social Frames
    numNONSocialFrames= np.float(np.sum(AllNONSocialFrames_TF))  # Final Sum of NON Social Frames 
        
    # Compute SPI
    if (numSocialFrames+numNONSocialFrames) == 0:
        SPI = 0.0
    else:
        SPI = (numSocialFrames-numNONSocialFrames)/(numSocialFrames+numNONSocialFrames)
        SPI = (numSocialFrames-numNONSocialFrames)/np.size(yPositions)
    
    
    return SPI, AllSocialFrames_TF, AllNONSocialFrames_TF



def computeSPI_3fish(xPositions, yPositions, testROI, stimLeft):
    
    # Find thresholds of X and Y Test ROI in order to define the social area to calculate the SPI
    socialPositionThreshold_X = testROI[0]+(testROI[2]/2)
#   
        
    # Compute Social Frames (depending on where the stimulus fish is (top or bottom))
    
   # If stimLeft is TRUE
    if stimLeft:
        AllSocialFrames_X_TF= xPositions<socialPositionThreshold_X    # Social Left     
    else:
       AllSocialFrames_X_TF= xPositions>socialPositionThreshold_X   # Opposite True/False array
    
    AllNONSocialFrames_X_TF=np.logical_not(AllSocialFrames_X_TF)   # Final NON SOCIAL True/False array

    # Count Social and Non-Social Frames
    numSocialFrames = np.float(np.sum(AllSocialFrames_X_TF))     # Final Sum of Social Frames
    numNONSocialFrames= np.float(np.sum(AllNONSocialFrames_X_TF))  # Final Sum of NON Social Frames 
        
    # Compute SPI
    if (numSocialFrames+numNONSocialFrames) == 0:
        SPI = 0.0
    else:
        SPI = (numSocialFrames-numNONSocialFrames)/(numSocialFrames+numNONSocialFrames)
#        SPI = (numSocialFrames-numNONSocialFrames)/np.size(yPositions)
    
    
    return SPI, AllSocialFrames_X_TF, AllNONSocialFrames_X_TF


# Compute VISIBLE Frames (when the Test Fish could potentially see the Stim Fish)
def computeVISIBLE(xPositions, yPositions, testROI, stimROI):
    
    # Find thresholds of Y from the Test ROI in order to define the Visible area
    visiblePositionThreshold_Y = testROI[1]+(testROI[3]/2) 
    
    # Define which frames are "VISIBLE" in Y by comparing Y with the above calculated threshold
    socialTop = stimROI[1]<visiblePositionThreshold_Y
        
    # Check Y threshold 
    if socialTop:  
        AllVisibleFrames_Y_TF= yPositions<visiblePositionThreshold_Y   # True/False array, BUT "Zero Y" remember is located conventionally at the TOP   
    else:
        AllVisibleFrames_Y_TF= yPositions>visiblePositionThreshold_Y   # Opposite True/False array
    
    
    return AllVisibleFrames_Y_TF
    

# Analyze Correlations between Test and Stimulus Fish
def analyze_tracking_SPI(folder, fishNumber, testROIs, stimROIs):
    
            # Analyze Tacking in Folder based on ROIs
            trackingFile = folder + r'\tracking' + str(fishNumber) + '.npz'    
            data = np.load(trackingFile)
            tracking = data['tracking']
            
            fx = tracking[:,0]      # Fish X (Centroid of binary particle)
            fy = tracking[:,1]      # Fish Y (centroid of binary particle)
            bx = tracking[:,2]      # Body X
            by = tracking[:,3]      # Body Y
            ex = tracking[:,4]      # Eye X
            ey = tracking[:,5]      # Eye Y
            area = tracking[:,6]    
            ort = tracking[:,7]     # Orientation
            motion = tracking[:,8]  # Frame-by-frame difference in particle
            
            # Compute SPI (NS)
            SPI, AllSocialFrames, AllNONSocialFrames = computeSPI(bx, by, testROIs[fishNumber-1], stimROIs[fishNumber-1])
            
            return SPI, AllSocialFrames, AllNONSocialFrames
            
# Analyze Correlations between Test and Stimulus Fish
def analyze_tracking_VISIBLE(folder, fishNumber, testROIs, stimROIs):
    
            # Analyze Tacking in Folder based on ROIs
            trackingFile = folder + r'\tracking' + str(fishNumber) + '.npz'    
            data = np.load(trackingFile)
            tracking = data['tracking']
            
            fx = tracking[:,0]      # Fish X (Centroid of binary particle)
            fy = tracking[:,1]      # Fish Y (centroid of binary particle)
            bx = tracking[:,2]      # Body X
            by = tracking[:,3]      # Body Y
            ex = tracking[:,4]      # Eye X
            ey = tracking[:,5]      # Eye Y
            area = tracking[:,6]    
            ort = tracking[:,7]     # Orientation
            motion = tracking[:,8]  # Frame-by-frame difference in particle
            
            # Compute AllVisibleFrames_Y_TFisible Frames
            AllVisibleFrames_Y_TF = computeVISIBLE(bx, by, testROIs[fishNumber-1], stimROIs[fishNumber-1])
            
            return AllVisibleFrames_Y_TF
            
# Analyze Correlations between Test and Stimulus Fish
def analyze_correlations(Test_folder, testNumber, Stim_folder, stimNumber, SocialFrames, corrLength, threshold):
    
    # Analyze Test
    trackingFile = Test_folder + r'\tracking' + str(testNumber) + '.npz'    
    data = np.load(trackingFile)
    tracking = data['tracking']
    
    fx = tracking[:,0] 
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort_test = tracking[:,7]
    motion_test = tracking[:,8]
        
    # Analyze Stim
    trackingFile = Stim_folder + r'\tracking' + str(stimNumber) + '.npz'    
    data = np.load(trackingFile)
    tracking = data['tracking']
    
    fx = tracking[:,0] 
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort_stim = tracking[:,7]
    motion_stim = tracking[:,8]
    
    # Filter Tracking based on Social Frames
    motion_test = motion_test[SocialFrames]
    motion_stim = motion_stim[SocialFrames]
    goodTracking = np.where((motion_test >= 0.0) * (motion_stim >= 0.0))
    goodTracking = goodTracking[0]
    motion_test = motion_test[goodTracking]
    motion_stim = motion_stim[goodTracking]
    
    # Threshold Motion Data to Remove Noise Correlations        
    baseline = np.median(motion_test)
    motion_test = motion_test - baseline
    sigma = np.std(motion_test)
    threshold_test = sigma*threshold
    motion_test = (motion_test > threshold_test) * motion_test

    baseline = np.median(motion_stim)
    motion_stim = motion_stim - baseline
    sigma = np.std(motion_stim)
    threshold_stim = sigma*threshold
    motion_stim = (motion_stim > threshold_stim) * motion_stim
    
    # Prepare "Motion" Arrays for Correlation Analysis (Pad with median "motion" value)
    padValue = np.median(motion_test)
    padding = np.zeros(corrLength)+padValue
    motion_test_padded = np.concatenate((padding, motion_test, padding), axis = 0)
    motion_stim_padded = np.concatenate((padding, motion_stim, padding), axis = 0)
    motion_test_padded_rev = motion_test_padded[::-1] # Scramble/Reverse Test
 
    # Compute Auto-Correlations
    auto_corr_test = np.correlate(motion_test_padded, motion_test, mode="valid")
    auto_corr_stim = np.correlate(motion_stim_padded, motion_stim, mode="valid")
    
    # Compute Cross-Correlations
    cross_corr = np.correlate(motion_test_padded, motion_stim, mode="valid")
    cross_corr_rev = np.correlate(motion_test_padded_rev, motion_stim, mode="valid")
    
    # Make Correlation Data Structure (2D array)
    corr_data = np.vstack((auto_corr_test, auto_corr_stim, cross_corr, cross_corr_rev))
    
    return corr_data

# Analyze Bouts of Test and Stimulus Fish
def analyze_bouts(testFolder, testNumber, stimFolder, stimNumber, visibleFrames, btaLength, threshold, testROI, stimROI):
    
    # Analyze Test
    trackingFile = testFolder + r'\tracking' + str(testNumber) + '.npz'    
    data = np.load(trackingFile)
    tracking = data['tracking']
    
    fx_test = tracking[:,0] 
    fy_test = tracking[:,1]
    bx_test = tracking[:,2]
    by_test = tracking[:,3]
    ex_test = tracking[:,4]
    ey_test = tracking[:,5]
    area_test = tracking[:,6]
    ort_test = tracking[:,7]
    motion_test = tracking[:,8]
        
    # Analyze Stim
    trackingFile = stimFolder + r'\tracking' + str(stimNumber) + '.npz'    
    data = np.load(trackingFile)
    tracking = data['tracking']
    
    fx_stim = tracking[:,0] 
    fy_stim = tracking[:,1]
    bx_stim = tracking[:,2]
    by_stim = tracking[:,3]
    ex_stim = tracking[:,4]
    ey_stim = tracking[:,5]
    area_stim = tracking[:,6]
    ort_stim = tracking[:,7]
    motion_stim = tracking[:,8]
    
    # Filter Tracking based on Social Frames (set to 0) - interpolate?
    motion_test[motion_test < 0.0] = 0.0
    motion_stim[motion_stim < 0.0] = 0.0
    
    # Compute Signal for bout detection (smoothed motion signal)   
    bout_filter = np.array([0.25, 0.25, 0.25, 0.25])
    boutSignal_test = signal.fftconvolve(motion_test, bout_filter, 'same')    
    boutSignal_stim = signal.fftconvolve(motion_stim, bout_filter, 'same')

    # Determine Threshold levels
    # - Determine the largest 100 values and take the median
    # - Use 10% of max level, divide by 10, for the base threshold (sigma)
    sorted_motion = np.sort(motion_test)
    max_norm = np.median(sorted_motion[-100:])    
    sigma_test = max_norm/10
    threshold_test = sigma_test*threshold    
    # - - - -
    print (threshold_test, max_norm)
    # - - - -
    sorted_motion = np.sort(motion_stim)
    max_norm = np.median(sorted_motion[-100:])    
    sigma_stim = max_norm/10
    threshold_stim = sigma_stim*threshold
    
    # Extract Bouts from Tracking Data
    bouts_test = SZU.extract_bouts_from_motion(bx_test, by_test, ort_test, boutSignal_test, threshold_test, threshold_test-sigma_test, testROI, True)
    bouts_stim = SZU.extract_bouts_from_motion(bx_stim, by_stim, ort_stim, boutSignal_stim, threshold_stim, threshold_stim-sigma_stim, stimROI, False)
    
    # Get Info about each bout (Align on STARTS!!)
    peaks_test = bouts_test[:, 1]
    peaks_stim = bouts_stim[:, 1]
    peaks_test = peaks_test.astype(int)
    peaks_stim = peaks_stim.astype(int)
    
    # Position at bout onset, offset

    # On Social(Visible side) at bout onset
    visible_testDuringStim = visibleFrames[peaks_stim]
    visible_stimDuringTest = visibleFrames[peaks_test]

    # Orientation of other fish during bout Peak
    # Zero degrees is towards bout generating fish, 180 is away, 90 is facing with REye, -90 is LEye
    # Seperate Orientations based on Chamber (1-6)
    # 1 - test right
    # 2 - test left
    # 3 - test right
    # 4 - test left
    # 5 - test right
    # 6 - test left

    ortOfTestDuringStim = ort_test[peaks_stim]
    ortOfStimDuringTest = ort_stim[peaks_test]
    
    # Adjust orientations so 0 is always pointing towards "other" fish
    if testNumber%2 == 0: # Test Fish facing Left
        for i,ort in enumerate(ortOfTestDuringStim):
            if ort >= 0: 
                ortOfTestDuringStim[i] = ort - 180
            else:
                ortOfTestDuringStim[i] = ort + 180    
    if stimNumber%2 == 1: # Stim fish facing Left
        for i,ort in enumerate(ortOfStimDuringTest):
            if ort >= 0: 
                ortOfStimDuringTest[i] = ort - 180
            else:
                ortOfStimDuringTest[i] = ort + 180  

    # Concatenate into Bouts Structure    
    bouts_test = np.hstack((bouts_test, np.transpose(np.atleast_2d(visible_stimDuringTest))))
    bouts_stim = np.hstack((bouts_stim, np.transpose(np.atleast_2d(visible_testDuringStim))))    
    
    bouts_test = np.hstack((bouts_test, np.transpose(np.atleast_2d(ortOfStimDuringTest))))
    bouts_stim = np.hstack((bouts_stim, np.transpose(np.atleast_2d(ortOfTestDuringStim))))

    return bouts_test, bouts_stim

# Compute BTA of Test and Stimulus Fish for different measures
def compute_BTA(bouts_test, bouts_stim, output_test, output_stim, btaLength):
        
    # Get Info about each bout (Align on Peaks!!)
    peaks_test = bouts_test[:, 1]
    peaks_stim = bouts_stim[:, 1]
    peaks_test = peaks_test.astype(int)
    peaks_stim = peaks_stim.astype(int)

    # Allocate Space for BTAs    
    BTA_test = np.zeros((np.size(peaks_test),2*btaLength,2))
    BTA_stim = np.zeros((np.size(peaks_stim),2*btaLength,2))

    # Pad OUTPUT variable for alignment
    padValue = np.median(output_test)
    padding = np.zeros(btaLength)+padValue
    output_test_padded = np.concatenate((padding, output_test, padding), axis = 0)
    padValue = np.median(output_stim)
    padding = np.zeros(btaLength)+padValue
    output_stim_padded = np.concatenate((padding, output_stim, padding), axis = 0)

    # Compute Burst Triggered Average (Auto-Corr)
    BTA_test[:,:,0] = SZU.burst_triggered_alignment(peaks_test, output_test_padded, 0, btaLength*2)
    BTA_stim[:,:,0] = SZU.burst_triggered_alignment(peaks_stim, output_stim_padded, 0, btaLength*2)

    # Compute Burst Triggered Average
    BTA_test[:,:,1] = SZU.burst_triggered_alignment(peaks_test, output_stim_padded, 0, btaLength*2)
    BTA_stim[:,:,1] = SZU.burst_triggered_alignment(peaks_stim, output_test_padded, 0, btaLength*2)
    
    return BTA_test, BTA_stim
    
# FIN
    