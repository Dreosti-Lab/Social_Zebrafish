# -*- coding: utf-8 -*-
"""
Created on Wed Dec 04 09:21:29 2013

@author: dreostilab (Elena Dreosti)
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
import CV_ARK


#-----------------------------------------------------------------------------
# Utilities for loading and ploting "social zebrafish" data

# 2) Read Folder List file 6 fish 
def read_folder_list(folderListFile): 
    folderFile = open(folderListFile, "r") #"r" means read the file
    folderList = folderFile.readlines() # returns a list containing the lines
    folderPath = folderList[0][:-1] # Read Data Path which is the first line
    folderList = folderList[1:] # Remove first line becasue it contains the path

    # Set Data Path where the experiments are located
    data_path = folderPath
    
    numFolders = len(folderList) 
    groups = np.zeros(numFolders)
    ages = np.zeros(numFolders)
    folderNames = [] # We use this becasue we do not know the exact lenght
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
    
    
  
## Read Folder List file 3 Fish
#def read_folder_list_3fish(folderListFile):
#    folderFile = open(folderListFile, "r")
#    folderList = folderFile.readlines()
#    folderPath = folderList[0][:-1] # Read Base Data Path
#    folderList = folderList[1:] # Remove first line
#
#    # Set Data Path
#    data_path = folderPath
#    
#    numFolders = len(folderList)
#    groups = np.zeros(numFolders)
#    ages = np.zeros(numFolders)
#    folderNames = []
#    fishStatus = np.zeros((numFolders, 3))
#    
#    for i, f in enumerate(folderList):
#        stringLine = f[:-1].split()
#        groups[i] = int(stringLine[0])
#        ages[i] = int(stringLine[1])
#        expFolderName = data_path + stringLine[2]
#        folderNames.append(expFolderName)
#        fishStat = [int(stringLine[3]), int(stringLine[4]), int(stringLine[5])]    
#        fishStatus[i,:] = np.array(fishStat)
#        
#    return groups, ages, folderNames, fishStatus
    
    

# 3) Determine Data Folder Names from Root directory
def get_folder_names(folder):
    # Specifiy Folder Names
    NS_folder = folder + r'\Non_Social_1'
    
    S_folder = folder + r'\Social_1'
    if os.path.exists(S_folder) == False:
        S_folder = folder + r'\Social_1'
    
    C_folder = folder + r'\Non_Social_2'
    if os.path.exists(C_folder) == False:
        C_folder = -1    
    
    return NS_folder, S_folder, C_folder
    
    
# Read ROIs (expecting 12)
def read_crop_ROIs(roiFilename):
    ROIs = CV_ARK.read_roi_zip(roiFilename)
    numROIs = len(ROIs)
    if (numROIs != 12):
        raise ValueError('roireader: Expecting 12 social ROIs, found %i' % numROIs)
    test_ROIs = np.zeros((6, 4))
    stim_ROIs = np.zeros((6, 4))
    
    test_ROIs[0,:] = np.array(ROIs[0])
    test_ROIs[1,:] = np.array(ROIs[3])
    test_ROIs[2,:] = np.array(ROIs[4])
    test_ROIs[3,:] = np.array(ROIs[7])
    test_ROIs[4,:] = np.array(ROIs[8])
    test_ROIs[5,:] = np.array(ROIs[11])
    
    stim_ROIs[0,:] = np.array(ROIs[1])
    stim_ROIs[1,:] = np.array(ROIs[2])
    stim_ROIs[2,:] = np.array(ROIs[5])
    stim_ROIs[3,:] = np.array(ROIs[6])
    stim_ROIs[4,:] = np.array(ROIs[9])
    stim_ROIs[5,:] = np.array(ROIs[10])
    
    return test_ROIs, stim_ROIs

# Load Tracking Data
def load_tracking(filename):

    # Read Column Headers
    datafile = open(filename, 'r')
    header = datafile.readline()
    header = header[:-1]                # Remove newline char
    column_lables = header.split(' ')

    # Read data from file
    data = np.genfromtxt(filename, delimiter=' ', skiprows=1)

    # Allocate space and fill data into appropriate columns
    tracking = np.zeros(np.shape(data))    
    tracking[:, 0] = data[:, column_lables.index('Centroid.X')]     
    tracking[:, 1] = data[:, column_lables.index('Centroid.Y')]     
    tracking[:, 2] = data[:, column_lables.index('Orientation')]     
    tracking[:, 3] = data[:, column_lables.index('MajorAxisLength')]     
    tracking[:, 4] = data[:, column_lables.index('MinorAxisLength')]     
    tracking[:, 5] = data[:, column_lables.index('Area')]
    
    return tracking

# Load Tracking Data
def plot_spatial_variable(X,Y,Var):
    
    width = np.ceil(np.max(X)-np.min(X))+1
    height = np.ceil(np.max(Y)-np.min(Y))+1
    X = X - np.min(X)
    Y = Y - np.min(Y)
    
    space = np.zeros([height, width])    
    data = np.vstack((X,Y,Var))
    data = data.T

    for x,y,var in data:
        space[np.round(y),np.round(x)] = var
    
    plt.imshow(space)

# Peak Detection
def find_peaks(values, threshold, refract):    
    over = 0
    r = 0
    starts = []
    peaks = []
    stops = []
    curPeakVal = 0
    curPeakIdx = 0
    
    numSamples = np.size(values)
    steps = range(numSamples)
    for i in steps[2:-100]:
        if over == 0:
            if values[i] > threshold:
                over = 1
                curPeakVal = values[i]
                curPeakIdx = i                                
                starts.append(i-1)
        else: #This is what happens when over the threshold
            if r < refract:
                r = r + 1
                if values[i] > curPeakVal:
                    curPeakVal = values[i]
                    curPeakIdx = i
            else:
                if values[i] > curPeakVal:
                    curPeakVal = values[i]
                    curPeakIdx = i
                elif values[i] < threshold:
                    over = 0
                    r = 0
                    curPeakVal = 0
                    peaks.append(curPeakIdx)
                    stops.append(i)
    
    return starts, peaks, stops

# Peak Detection
def find_peaks_dual_threshold(values, upper_threshold, lower_threshold):    
    over = 0
    starts = []
    peaks = []
    stops = []
    curPeakVal = 0
    curPeakIdx = 0
    
    numSamples = np.size(values)
    steps = range(numSamples)
    for i in steps[5:-100]:
        if over == 0:
            if values[i] > upper_threshold:
                over = 1
                curPeakVal = values[i]
                curPeakIdx = i                                
                starts.append(i)
        else: #This is what happens when over the upper_threshold
            if values[i] > curPeakVal:
                curPeakVal = values[i]
                curPeakIdx = i
            elif values[i] < lower_threshold:
                over = 0
                curPeakVal = 0
                peaks.append(curPeakIdx)
                stops.append(i)
    
    return starts, peaks, stops

def diffAngle(Ort):
    dAngle = np.diff(Ort)
    new_dAngle = [0]    
    for a in dAngle:
        if a < -270:
            new_dAngle.append(a + 360)
        elif a > 270:
            new_dAngle.append(a - 360)
        else:
            new_dAngle.append(a)
    
    return np.array(new_dAngle)

def filterTrackingFlips(dAngle):
    new_dAngle = []    
    for a in dAngle:
        if a < -90:
            new_dAngle.append(a + 180)
        elif a > 90:
            new_dAngle.append(a - 180)
        else:
            new_dAngle.append(a)
            
    return np.array(new_dAngle)

def compute_speed(X,Y):
    # Compute Speed (X-Y)    
    speed = np.sqrt(np.diff(X)*np.diff(X) + np.diff(Y)*np.diff(Y)) 
    speed = np.append([0], speed)
    return speed

def motion_signal(X, Y, Ort):
    # Compute Speed (X-Y)    
    SpeedXY = compute_speed(X,Y)

    # Compute Speed (Angular)
    SpeedAngle = diffAngle(Ort)
    SpeedAngle = filterTrackingFlips(SpeedAngle)
    
    # Absolute Value of angular speed
    SpeedAngle = np.abs(SpeedAngle)

    # Weight contribution by STD
    std_XY = np.std(SpeedXY)    
    std_Angle = np.std(SpeedAngle)    
    SpeedXY = SpeedXY/std_XY
    SpeedAngle = SpeedAngle/std_Angle

    # Filter Combined Signal
    motion_signal = SpeedXY+SpeedAngle
    #motion_signal = SpeedAngle
        

    return motion_signal

# Compute Dynamic Signal for Detecting Bouts (swims and turns)
def compute_bout_signals(X, Y, Ort):

    # Compute Speed (X-Y)    
    speedXY = compute_speed(X,Y)
    
#    # Filter Speed for outliers
#    sigma = np.std(speedXY)
#    baseline = np.median(speedXY)
#    speedXY[speedXY > baseline+10*sigma] = -1.0

    # Compute Speed (Angular)
    speedAngle = diffAngle(Ort)
    speedAngle = filterTrackingFlips(speedAngle)
    
    return speedXY, speedAngle

# Compute Dynamic Signal for Detecting Bouts (swims and turns)
def compute_bout_signals_calibrated(X, Y, Ort, ROI, test):
    
    # Calibrate X and Y in ROI units
    offX = ROI[0]
    offY = ROI[1]
    width = ROI[2]
    height = ROI[3] 
    X = (X - offX)/width
    Y = (Y - offY)/height
    if test:
        X = X * 14; # Convert to mm
        Y = Y * 42; # Convert to mm
    else:
        X = X * 14; # Convert to mm
        Y = Y * 14; # Convert to mm
        

    # Compute Speed (X-Y)    
    speedXY = compute_speed(X,Y)
    
#    # Filter Speed for outliers
#    sigma = np.std(speedXY)
#    baseline = np.median(speedXY)
#    speedXY[speedXY > baseline+10*sigma] = -1.0

    # Compute Speed (Angular)
    speedAngle = diffAngle(Ort)
    speedAngle = filterTrackingFlips(speedAngle)
    
    return speedXY, speedAngle

# Adjust Orientation (Test Fish)
def adjust_ort_test(ort, chamber):
    # Adjust orientations so 0 is always pointing towards "other" fish
    if chamber%2 == 0: # Test Fish facing Left
        for i,angle in enumerate(ort):
            if angle >= 0: 
                ort[i] = angle - 180
            else:
                ort[i] = angle + 180
    return ort

# Adjust Orientation (Stim Fish)
def adjust_ort_stim(ort, chamber):
    # Adjust orientations so 0 is always pointing towards "other" fish
    if chamber%2 == 1: # Stim Fish facing Left
        for i,angle in enumerate(ort):
            if angle >= 0: 
                ort[i] = angle - 180
            else:
                ort[i] = angle + 180
    return ort

# Extract Bouts from Motion Signal
def extract_bouts_from_motion(X, Y, Ort, motion, upper_threshold, lower_threshold, ROI, test):

    if test:
        SpeedXY, SpeedAngle = compute_bout_signals_calibrated(X, Y, Ort, ROI, True)
    else:
        SpeedXY, SpeedAngle = compute_bout_signals_calibrated(X, Y, Ort, ROI, False)        
     
    # Find Peaks in Motion Signal 
    starts, peaks, stops = find_peaks_dual_threshold(motion, upper_threshold, lower_threshold)
    numBouts = np.size(peaks)    
    bouts = np.zeros([numBouts, 6])

    for i in range(numBouts):
        bouts[i, 0] = starts[i]-4 # Point 4 frames (40 ms) before Upper threshold crossing 
        bouts[i, 1] = peaks[i] # Peak
        bouts[i, 2] = stops[i]+1 # Point 1 frame (10 ms) after lower threshold crossing
        bouts[i, 3] = stops[i]-starts[i] # Durations
        bouts[i, 4] = np.sum(SpeedAngle[starts[i]:stops[i]]) # Net angle change  
        bouts[i, 5] = np.sum(SpeedXY[starts[i]:stops[i]]) # Net distance change

    return bouts


# Make Polar Plot of Orientation
def polar_orientation(Ort):
    ort_hist, edges = np.histogram(Ort, 18, (0, 360))
    plt.plot(edges/(360/(2*np.pi)), np.append(ort_hist, ort_hist[0]))
    max_ort = edges[np.argmax(ort_hist)]
    return max_ort

# Quantify Tracking Data (remove errors in tracking)
def measure_tracking_errors(tracking):
    X = tracking[:, 0]
    Y = tracking[:, 1]
    Ort = tracking[:, 2]
    MajAx = tracking[:, 3]
    MinAx = tracking[:, 4]
    Area = tracking[:, 5]
    
    # Filter out and interpolate between tracking errors
    speedXY =  compute_speed(X,Y)
    tooFastorSmall = (speedXY > 50) + (Area < 75)    
    trackingErrors = np.sum(tooFastorSmall)    
    
    return trackingErrors


# Quantify Tracking Data (remove errors in tracking)
def burst_triggered_alignment(starts, variable, offset, length):
    starts = starts[starts > offset]
    numStarts = np.size(starts)
    aligned = np.zeros((numStarts, length))

    for s in range(0, numStarts):
        aligned[s, :] = variable[starts[s]-offset:starts[s]-offset+length]

    return aligned

# Find nearest index in array 2 from array 1
def find_dist_to_nearest_index(array1, array2):
    numIndices = np.size(array1)    
    distances = np.zeros(np.shape(array1))
    for i in range(0, numIndices):
        diffArray = array2 - array1[i]
        closestIndex = np.argmin(np.abs(diffArray))
        distances[i] = diffArray[closestIndex]
    
    return distances
    
# Find nearest index in array 2 AFTER those in array 1
def find_next_nearest_index(array1, array2):
    numIndices = np.size(array1)    
    nextIndices = np.zeros(np.shape(array1))
    for i in range(0, numIndices):
        diffArray = array2 - array1[i]
        positive = np.where(diffArray > 0)[0]
        if np.size(positive) != 0:
            nextIndex = positive[0]
            nextIndices[i] = diffArray[nextIndex]
    
    return nextIndices



    
def get_folder_names_controls(folder):
    # Specifiy Folder Names
    NS_folder = folder + r'\Non_Social_1'
    
    S_folder = folder + r'\Non_Social_2'
    if os.path.exists(S_folder) == False:
        S_folder = folder + r'\Social_1_Real'
        
#    D_folder = folder + r'\Social_Dark'
#    if os.path.exists(D_folder) == False:
#        D_folder = -1
#    
#    C_folder = folder + r'\Non_Social_2'
#    if os.path.exists(C_folder) == False:
#        C_folder = -1    
    
    return NS_folder, S_folder
    
    
    
# FIN