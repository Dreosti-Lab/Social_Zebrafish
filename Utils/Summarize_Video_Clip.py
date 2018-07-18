# -*- coding: utf-8 -*-
"""
Makes a summary image from a video clip

@author: dreostilab (Elena Dreosti)
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
#lib_path = r'C:/Repos/Dreosti-Lab\Social_Zebrafish\libs'
lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Zebrafish/libs'

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
import scipy.misc as misc
import math
import glob
import cv2
import SZ_utilities as SZU

# Specify moive clip name
movie_name = r'/home/kampff/Dropbox/Adam_Ele/Movie_Clip/Social_1.avi'
#movie_name = r'/home/kampff/Dropbox/Adam_Ele/Movie_Clip/Social_1_Isolated.avi'

# Specify output folder
save_folder = r'/home/kampff/Dropbox/Adam_Ele/Movie_Clip/Summary'

# Define crop region


# Load Video
vid = cv2.VideoCapture(movie_name)
numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

# Read First Frame
ret, im = vid.read()
previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
width = np.size(previous, 1)
height = np.size(previous, 0)

# Compute Background
stepFrames = 500 # Add a background frame every 0.5 seconds for 50 seconds
bFrames = 100
thresholdValue=10
backgroundStack = np.zeros((height, width, bFrames), dtype = float)
background = np.zeros((height, width), dtype = float)
bCount = 0
for i, f in enumerate(range(0, bFrames*stepFrames, stepFrames)):
    
    vid.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, im = vid.read()
    current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Add to background stack
    if(bCount < bFrames):
        backgroundStack[:,:,bCount] = current
        bCount = bCount + 1
    
    print (numFrames-f)

# Compute Background Frame (median or mode)
background = np.median(backgroundStack, axis = 2)
background = background.astype(np.uint8)

# Subtract Background
stepFrames = 50
thresholdValue=10
accumulated_diff = np.zeros((height, width), dtype = float)
for i, f in enumerate(range(0, numFrames, stepFrames)):
    
    vid.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, im = vid.read()
    current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    absDiff = cv2.absdiff(background, current)
    level, threshold = cv2.threshold(absDiff,thresholdValue,255,cv2.THRESH_TOZERO)
    
    # Accumulate differences
    accumulated_diff = accumulated_diff + threshold
        
    print (numFrames-f)
#        print (bCount)

vid.release()

# Normalize accumulated difference image
accumulated_diff = accumulated_diff/np.max(accumulated_diff)
accumulated_diff = np.ubyte(accumulated_diff*255)

# Enhance Contrast (Histogram Equalize)
equ = cv2.equalizeHist(accumulated_diff)


misc.imsave(save_folder + r'/difference.png', equ)    
cv2.imwrite(save_folder + r'/background.png', background)
# Using SciPy to save caused a weird rescaling when the images were dim.
# This will change not only the background in the beginning but the threshold estimate

