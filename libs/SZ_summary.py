# -*- coding: utf-8 -*-
"""
 SZ_summary:
     - Social Zebrafish - Summary Analysis Functions

@author: adamk
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal
import CV_ARK


#-----------------------------------------------------------------------------
# Utilities for summarizing and plotting "Social Zebrafish" experiments

# Compute activity level of the fish in bouts per second (BPS)
def measure_BPS(motion):

    # Estimate motion bout threshold
    motionThreshold = np.median(motion)+0.05

    # Exclude first 1 minute (6000 frames)
    motion = motion[6000:];
                   
    # Threshold motion signal
    motionBinary = np.array(motion>motionThreshold, dtype = np.int32)

    # Find starts and stops by measuring transitions from (0 to 1) = +1 or (1 to 0) = -1
    transitions = np.diff(motionBinary)

    # Find only the "starts": where the transition is +1
    startArray = np.array(transitions == 1, dtype = np.int32)

    # Find only the "stops": where the transition is -1
    stopArray = np.array(transitions == -1, dtype = np.int32)

    # Count number of bouts
    numBouts= np.sum(startArray)
    numberOfSeconds = np.size(motion)/100   ## Assume 100 Frames per Second

    # Set the bouts per second (BPS)
    boutsPerSecond = numBouts/numberOfSeconds
    
    # Measure averge bout trajectory
    boutStarts = np.where(startArray)[0]
    boutStarts = boutStarts[(boutStarts > 25) * (boutStarts < (len(motion)-75))]
    allBouts = np.zeros([len(boutStarts), 100])
    for b in range(0,len(boutStarts)):
        allBouts[b,:] = motion[(boutStarts[b]-25):(boutStarts[b]+75)];
    avgBout = np.mean(allBouts,0);

    return boutsPerSecond, avgBout

# Build a histogram of all orientation values
def ort_histogram(ort):

    # ORIENTATION ---------------------------
    numOrts = 36
    interval = 360/numOrts
    ortRange = np.arange(-180,180+interval, interval)    
    ortHistogram, bins = np.histogram(ort, ortRange)

    return ortHistogram


# Measure IBI lengths
def interBout_intervals(motion):

    # Estimate motion bout threshold
    motionThreshold = np.median(motion)+0.05

    # Exclude first 1 minute (6000 frames)
    motion = motion[6000:];
                   
    # Threshold motion signal
    motionBinary = np.array(motion>motionThreshold, dtype = np.int32)

    # Find starts and stops by measuring transitions from (0 to 1) = +1 or (1 to 0) = -1
    transitions = np.diff(motionBinary)

    # Find only the "starts": where the transition is +1
    startArray = np.array(transitions == 1, dtype = np.int32)

    # Find only the "stops": where the transition is -1
    stopArray = np.array(transitions == -1, dtype = np.int32)

    # Count number of bouts
    numBouts= np.sum(startArray)
    numberOfSeconds = np.size(motion)/100   ## Assume 100 Frames per Second

    # Set the bouts per second (BPS)
    boutsPerSecond = numBouts/numberOfSeconds
    
    # Measure inter bout intervals
    boutStarts = np.where(startArray)[0]
    interBoutIntervals = np.diff(boutStarts)
    
    return interBoutIntervals
        
# FIN
