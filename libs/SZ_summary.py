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
def measure_BPS(motion, startThreshold, stopThreshold):
                   
    # Find bouts starts and stops
    boutStarts = []
    boutStops = []
    moving = 0
    for i, m in enumerate(motion):
        if(moving == 0):
            if m > startThreshold:
                moving = 1
                boutStarts.append(i)
        else:
            if m < stopThreshold:
                moving = 0
                boutStops.append(i)
    
    # Extract all bouts (ignore last, if clipped)
    boutStarts = np.array(boutStarts)
    boutStops = np.array(boutStops)
    if(len(boutStarts) > len(boutStops)):
        boutStarts = boutStarts[:-1]

    # Count number of bouts
    numBouts= len(boutStarts)
    numberOfSeconds = np.size(motion)/100   ## Assume 100 Frames per Second

    # Set the bouts per second (BPS)
    boutsPerSecond = numBouts/numberOfSeconds
    
    # Measure averge bout trajectory
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
def analyze_bouts_and_pauses(tracking, startThreshold, stopThreshold):
    
    # Extract tracking details
    bx = tracking[:,2]
    by = tracking[:,3]
    ort = tracking[:,7]
    motion = tracking[:,8]                
    
    # Find bouts starts and stops
    boutStarts = []
    boutStops = []
    moving = 0
    for i, m in enumerate(motion):
        if(moving == 0):
            if m > startThreshold:
                moving = 1
                boutStarts.append(i)
        else:
            if m < stopThreshold:
                moving = 0
                boutStops.append(i)
    
    # Extract all bouts (ignore last, if clipped)
    boutStarts = np.array(boutStarts)
    boutStops = np.array(boutStops)
    if(len(boutStarts) > len(boutStops)):
        boutStarts = boutStarts[:-1]

    # Extract all bouts (startindex, startx, starty, startort, stopindex, stopx, stopy, stoport, duration)
    numBouts= len(boutStarts)
    bouts = np.zeros((numBouts, 9))
    for i in range(0, numBouts):
        bouts[i, 0] = boutStarts[i]
        bouts[i, 1] = bx[boutStarts[i]]
        bouts[i, 2] = by[boutStarts[i]]
        bouts[i, 3] = ort[boutStarts[i]]
        bouts[i, 4] = boutStops[i]
        bouts[i, 5] = bx[boutStops[i]]
        bouts[i, 6] = by[boutStops[i]]
        bouts[i, 7] = ort[boutStops[i]]
        bouts[i, 8] = boutStops[i] - boutStarts[i]
        
    # Analyse all pauses (startindex, startx, starty, startort, stopindex, stopx, stopy, stoport, duration)
    numPauses = numBouts+1
    pauses = np.zeros((numPauses, 9))

    # -Include first and last as pauses (clipped in video)
    # First Pause
    pauses[0, 0] = 0
    pauses[0, 1] = bx[0]
    pauses[0, 2] = by[0]
    pauses[0, 3] = ort[0]
    pauses[0, 4] = boutStarts[0]
    pauses[0, 5] = bx[boutStarts[0]]
    pauses[0, 6] = by[boutStarts[0]]
    pauses[0, 7] = ort[boutStarts[0]]
    pauses[0, 8] = boutStarts[0]
    # Other pauses
    for i in range(1, numBouts):
        pauses[i, 0] = boutStops[i-1]
        pauses[i, 1] = bx[boutStops[i-1]]
        pauses[i, 2] = by[boutStops[i-1]]
        pauses[i, 3] = ort[boutStops[i-1]]
        pauses[i, 4] = boutStarts[i]
        pauses[i, 5] = bx[boutStarts[i]]
        pauses[i, 6] = by[boutStarts[i]]
        pauses[i, 7] = ort[boutStarts[i]]
        pauses[i, 8] = boutStarts[i] - boutStops[i-1]
    # Last Pause
    pauses[-1, 0] = boutStops[-1]
    pauses[-1, 1] = bx[boutStops[-1]]
    pauses[-1, 2] = by[boutStops[-1]]
    pauses[-1, 3] = ort[boutStops[-1]]
    pauses[-1, 4] = len(motion)-1
    pauses[-1, 5] = bx[-1]
    pauses[-1, 6] = by[-1]
    pauses[-1, 7] = ort[-1]
    pauses[-1, 8] = len(motion)-1-boutStops[-1]
    
    return bouts, pauses
        
# FIN
