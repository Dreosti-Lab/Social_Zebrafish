# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:44:40 2013

@author: dreostilab (Elena Dreosti)
"""
# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import glob
import cv2
import BONSAI_ARK

# Utilities for processing videos of Social Experiments


# Process Video : Make Summary Images
def process_video_summary_images(folder, social=False):
    
    # Load Video
    aviFiles = glob.glob(folder+'/*.avi')  # make a list of all the path names of the .avi videos
    aviFile = aviFiles[0] # get the path name of the 1st .avi file in the list
    vid = cv2.VideoCapture(aviFile) # Creates the video capture object
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) # Gets one of the properties of the VC object by using method .get. 
    #In this case we want to get the Num of frames so we use cv2.CAP_PROP_FRAME_COUNT. But there are otehr properties like:
    #cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FPS,  cv2.CAP_PROP_BRIGHTNESS 
    #And also you can convert it to RGB by using cv2.CAP_PROP_CONVERT_RGB
    
    # Read First Frame
    ret, im = vid.read() #ret = is a boolean value that is True if the frame exist, im = is a variable storing the image frame
    previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # Cv2.cvtColor is a library used to convert the colours of images. cv2.COLOR_BGR2GRAY = converts from RGB to grayscale
    width = np.size(previous, 1) # Get pixel sise of frame
    height = np.size(previous, 0) 
    
    
    # Alloctae Image Space
    stepFrames = 1500 # Add a background frame every 15 seconds for 600 seconds
    bFrames = 40
    accumulated_diff = np.zeros((height, width), dtype = float) #Make an empty 2D np arrray of same size of frame  
    backgroundStack = np.zeros((height, width, bFrames), dtype = float) #Make an empty 3D np arrray of same size of frame and lenght based on bFrame
    background = np.zeros((height, width), dtype = float)
    croppedTest = np.zeros((height, width), dtype = float)
    croppedStim = np.zeros((height, width), dtype = float)
    
    bCount = 0 # make a loop that goes from frame 0 to the whole movie with steps of 15 sec
    for i, f in enumerate(range(0, numFrames, stepFrames)):
        
        vid.set(cv2.CAP_PROP_POS_FRAMES, f) #The .set() allows us to set the video on the frame we specifcy "f", which is our current frame
        ret, im = vid.read() #The read finctiomn returns the frame (im) and a boolean value (True if there is a frame)
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # convert to grayscale
        absDiff = cv2.absdiff(previous, current) # Calculate an image that is the absolute difference between 2 images
        level, threshold = cv2.threshold(absDiff,20,255,cv2.THRESH_TOZERO) # Threshold the absDiff image using 20 as the min threshold pixel value. 
        # 255 as maximum value, and use the method "Threshold zero"  
        #Threshold zero means that all values between 20 and 255 will remain the same. Below 20 will be assigned "0" 
        previous = current # Set the previous frame to current frame to do the absdiff
       
        # Accumulate differences
        accumulated_diff = accumulated_diff + threshold # This adds one frame threshold after the other. it is not a sum (similar to np.add)

        # Add to background stack
        if(bCount < bFrames):
            backgroundStack[:,:,bCount] = current # Add the current grayscale frame to the background
            bCount = bCount + 1 # This is to makle the bcount ot increase of 1 every time.
        
        print (numFrames-f) #This is to check what frame it is processing

    vid.release()  # It closes the video stream or file uploaded

    # Normalize accumulated difference image
    accumulated_diff = accumulated_diff/np.max(accumulated_diff) #Get values from 0 to 1
    accumulated_diff = np.ubyte(accumulated_diff*255)  # expand range from 0 to 255
    # print(accumulated_diff)
    
    # Enhance Contrast (Histogram Equalize)
    equ = cv2.equalizeHist(accumulated_diff) #Use the funciton to get a nicer contrast image

    # Compute Background Frame (median or mode) 
    background = np.median(backgroundStack, axis = 2)
    
    # Maybe Display Background
    plt.figure()    # make a figure
    plt.imshow(background, cmap = plt.cm.gray, vmin = 0, vmax = 255) # plot the image using gray scale from cmap, set min and max
    plt.figure()
    plt.imshow(equ, cmap = plt.cm.gray, vmin = 0, vmax = 255) 
    #plt.show()
    
    # plt.draw() # This is to update the figure
    # plt.pause(0.0001) #This tells how often to update the figure in sec?
    
    # Show Crop Regions

    # Load Test Crop Regions
    bonsaiFiles = glob.glob(folder+'/*.bonsai') # find the path of the file
    bonsaiFiles = bonsaiFiles[0] # Get the first path name. It assumes there is only 1 per folder
    test_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    #test_ROIs = test_ROIs[:, :]
    croppedTest = np.copy(background) # Copy the background image
    #print(test_ROIs)
    
    # Load Stim Crop Regions
    if social:
        bonsaiFiles = glob.glob(folder+'/Social_Fish/*.bonsai')
        bonsaiFiles = bonsaiFiles[0]
        stim_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
        #stim_ROIs = stim_ROIs[:, :]
        croppedStim = np.copy(background)    
    
    for i in range(0,6):
        r1 = int(test_ROIs[i, 1]) # get a row value of Y from the 6 fish 
        r2 = int(r1+test_ROIs[i, 3])  # get a row value of length and add it to Y from the 6 fish
        c1 = int(test_ROIs[i, 0])  # get a row values of X from the 6 fish
        c2 = int(c1+test_ROIs[i, 2]) # get a row values of width and add it to X from the 6 fish
        croppedTest[r1:r2, c1:c2] = 0 # Crop the background image to get only the chamber of each fish
        
        
        if social:
            r1 = int(stim_ROIs[i, 1])
            r2 = int(r1+stim_ROIs[i, 3])
            c1 = int(stim_ROIs[i, 0])
            c2 = int(c1+stim_ROIs[i, 2])
            croppedStim[r1:r2, c1:c2] = 0
    
    
#    
    summary = np.zeros((height, width, 3), dtype = float) # make a 3D arrays that has all the r1
    #summaryEqu = np.zeros((height, width, 3), dtype = float) # make a 3D arrays that has all the r1
    #summaryTest = np.zeros((height, width, 3), dtype = float) # make a 3D arrays that has all the r1
    if social:
        summary[:,:, 0] = croppedStim;
    
    summary[:,:, 1] = equ;
    summary[:,:, 2] = croppedTest;

    summary = summary / np.max(summary)
   
    # Save images
    saveFolder = folder
    plt.imsave(saveFolder + '/background2.png', background, cmap='gray')
    cv2.imwrite(saveFolder + '/background.png', background) # Used to do more analysis
    plt.imsave(saveFolder + '/difference.png', equ, cmap='cividis')
    plt.imsave(saveFolder + r'/summary.png', summary)
    #plt.imsave(saveFolder + r'/summary2.png', summaryEqu)
    #plt.imsave(saveFolder + r'/summary3.png', summaryTest)
    
    
    # Be aware that **plt.imsave** Automatically normalizes data if the input image array has values outside the [0, 1] or [0, 255] range. 
    # For example, if you provide float data, plt.imsave will scale it to fit the image intensity range.
    # **cv2.imwrite** Expects the data to be in a specific range: [0, 255] for 8-bit images. If the data isn't in this range, you'll need 
    # to normalize or scale it manually, or cv2.imwrite may not save it correctly.

    return 0



