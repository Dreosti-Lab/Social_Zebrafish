# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:44:40 2013

@author: dreostilab (Elena Dreosti)
"""
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
import BONSAI_ARK

# Utilities for processing videos of Social Experiments

# Process Video : Make Summary Images
def pre_process_video_summary_images(folder, social):
    
    # Load Video
    aviFiles = glob.glob(folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read First Frame
    ret, im = vid.read()
    previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    width = np.size(previous, 1)
    height = np.size(previous, 0)
    
    # Alloctae Image Space
    stepFrames = 250 # Add a background frame every 2.5 seconds for 50 seconds
    bFrames = 50
    thresholdValue=10
    accumulated_diff = np.zeros((height, width), dtype = float)
    backgroundStack = np.zeros((height, width, bFrames), dtype = float)
    background = np.zeros((height, width), dtype = float)
    bCount = 0
    for i, f in enumerate(range(0, numFrames, stepFrames)):
        
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        absDiff = cv2.absdiff(previous, current)
        level, threshold = cv2.threshold(absDiff,thresholdValue,255,cv2.THRESH_TOZERO)
        previous = current
       
        # Accumulate differences
        accumulated_diff = accumulated_diff + threshold
        
        # Add to background stack
        if(bCount < bFrames):
            backgroundStack[:,:,bCount] = current
            bCount = bCount + 1
        
        print (numFrames-f)
#        print (bCount)

    vid.release()

    # Normalize accumulated difference image
    accumulated_diff = accumulated_diff/np.max(accumulated_diff)
    accumulated_diff = np.ubyte(accumulated_diff*255)
    
    # Enhance Contrast (Histogram Equalize)
    equ = cv2.equalizeHist(accumulated_diff)

    # Compute Background Frame (median or mode)
    background = np.median(backgroundStack, axis = 2)

    saveFolder = folder
    misc.imsave(saveFolder + r'/difference.png', equ)    
    cv2.imwrite(saveFolder + r'/background.png', background)
    # Using SciPy to save caused a weird rescaling when the images were dim.
    # This will change not only the background in the beginning but the threshold estimate

    return 0


# Process Video : Make Summary Images
def process_video_summary_images(folder, social):
    
    # Load Video
    aviFiles = glob.glob(folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read First Frame
    ret, im = vid.read()
    previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    width = np.size(previous, 1)
    height = np.size(previous, 0)
    
    # Alloctae Image Space
    #stepFrames = 250 # Add a background frame every 2.5 seconds for 50 seconds
    #bFrames = 20
    stepFrames = 1500 # Add a background frame every 15 seconds for 600 seconds
    bFrames = 40
    accumulated_diff = np.zeros((height, width), dtype = float)
    backgroundStack = np.zeros((height, width, bFrames), dtype = float)
    background = np.zeros((height, width), dtype = float)
    croppedTest = np.zeros((height, width), dtype = float)
    croppedStim = np.zeros((height, width), dtype = float)
    bCount = 0
    for i, f in enumerate(range(0, numFrames, stepFrames)):
        
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        absDiff = cv2.absdiff(previous, current)
        level, threshold = cv2.threshold(absDiff,20,255,cv2.THRESH_TOZERO)
        previous = current
       
        # Accumulate differences
        accumulated_diff = accumulated_diff + threshold
        
        # Add to background stack
        if(bCount < bFrames):
            backgroundStack[:,:,bCount] = current
            bCount = bCount + 1
        
        print (numFrames-f)

    vid.release()

    # Normalize accumulated difference image
    accumulated_diff = accumulated_diff/np.max(accumulated_diff)
    accumulated_diff = np.ubyte(accumulated_diff*255)
    
    # Enhance Contrast (Histogram Equalize)
    equ = cv2.equalizeHist(accumulated_diff)

    # Compute Background Frame (median or mode)
    background = np.median(backgroundStack, axis = 2)
    
    # Maybe Display Background
    
    plt.figure()    
    plt.imshow(background, cmap = plt.cm.gray, vmin = 0, vmax = 255)
    plt.draw()
    plt.pause(0.001)
    
    ## Show Crop Regions
    
    # Load Test Crop Regions
    bonsaiFiles = glob.glob(folder+'/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    test_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    test_ROIs = test_ROIs[:, :]
    croppedTest = np.copy(background)
    
    # Load Stim Crop Regions
    if social:
        bonsaiFiles = glob.glob(folder+'/Social_Fish/*.bonsai')
        bonsaiFiles = bonsaiFiles[0]
        stim_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
        stim_ROIs = stim_ROIs[:, :]
        croppedStim = np.copy(background)    
    
    for i in range(0,6):
        r1 = np.int(test_ROIs[i, 1])
        r2 = np.int(r1+test_ROIs[i, 3])
        c1 = np.int(test_ROIs[i, 0])
        c2 = np.int(c1+test_ROIs[i, 2])
        croppedTest[r1:r2, c1:c2] = 0
        
        
        if social:
            r1 = np.int(stim_ROIs[i, 1])
            r2 = np.int(r1+stim_ROIs[i, 3])
            c1 = np.int(stim_ROIs[i, 0])
            c2 = np.int(c1+stim_ROIs[i, 2])
            croppedStim[r1:r2, c1:c2] = 0
    
        
#    plt.figure()
#    plt.imshow(croppedTest, cmap = plt.cm.gray, vmin = 0, vmax = 255)
#    plt.draw()
#    plt.pause(0.001)
#    
#        
#    plt.figure()    
#    plt.imshow(croppedStim, cmap = plt.cm.gray, vmin = 0, vmax = 255)
#    plt.draw()
#    plt.pause(0.001)
#    
#    plt.figure()    
#    plt.imshow(equ, cmap = plt.cm.gray, vmin = 0, vmax = 255)
#    plt.draw()
#    plt.pause(0.001)
#    
    summary = np.zeros((height, width, 3), dtype = float)
    if social:
        summary[:,:, 0] = croppedStim;
    
    summary[:,:, 1] = equ;
    summary[:,:, 2] = croppedTest;
    
    saveFolder = folder
    misc.imsave(saveFolder + r'/background_old.png', background)
    misc.imsave(saveFolder + r'/summary.png', summary)    
#    cv2.imwrite(saveFolder + r'/background.png', cv2.fromarray(background))
    cv2.imwrite(saveFolder + r'/background.png', background)
    # Using SciPy to save caused a weird rescaling when the images were dim.
    # This will change not only the background in the beginning but the threshold estimate

    return 0

# Process Video : Make Summary Images
def process_video_summary_images_3fish(folder, social):
    
    # Load Video
    aviFiles = glob.glob(folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    
    # Read First Frame
    ret, im = vid.read()
    previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    width = np.size(previous, 1)
    height = np.size(previous, 0)
    
    # Alloctae Image Space
    stepFrames = 250 # Add a background frame every 2.5 seconds for 50 seconds
    bFrames = 20
    accumulated_diff = np.zeros((height, width), dtype = float)
    backgroundStack = np.zeros((height, width, bFrames), dtype = float)
    background = np.zeros((height, width), dtype = float)
    croppedTest = np.zeros((height, width), dtype = float)
    croppedStim = np.zeros((height, width), dtype = float)
    bCount = 0
    for i, f in enumerate(range(0, numFrames, stepFrames)):
        
        vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        absDiff = cv2.absdiff(previous, current)
        level, threshold = cv2.threshold(absDiff,20,255,cv2.THRESH_TOZERO)
        previous = current
       
        # Accumulate differences
        accumulated_diff = accumulated_diff + threshold
        
        # Add to background stack
        if(bCount < bFrames):
            backgroundStack[:,:,bCount] = current
            bCount = bCount + 1
        
        print (numFrames-f)

    vid.release()

    # Normalize accumulated difference image
    accumulated_diff = accumulated_diff/np.max(accumulated_diff)
    accumulated_diff = np.ubyte(accumulated_diff*255)
    
    # Enhance Contrast (Histogram Equalize)
    equ = cv2.equalizeHist(accumulated_diff)

    # Compute Background Frame (median or mode)
    background = np.median(backgroundStack, axis = 2)
    
#    # Maybe Display Background
    plt.clf()
    plt.imshow(background, cmap = plt.cm.gray, vmin = 0, vmax = 255)
    plt.draw()
    plt.pause(0.001)
    
    ## Show Crop Regions
    
    # Load Test Crop Regions
    bonsaiFiles = glob.glob(folder+'/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    test_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    test_ROIs = test_ROIs[1:, :]
    croppedTest = np.copy(background)
    
    # Load Stim Crop Regions
#    if social:
#        bonsaiFiles = glob.glob(folder+'/Social_Fish/*.bonsai')
#        bonsaiFiles = bonsaiFiles[0]
#        stim_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
#        stim_ROIs = stim_ROIs[1:, :]
#        croppedStim = np.copy(background)    
    
    for i in range(0,3):
        r1 = test_ROIs[i, 1]
        r2 = r1+test_ROIs[i, 3]
        c1 = test_ROIs[i, 0]
        c2 = c1+test_ROIs[i, 2]
        croppedTest[r1:r2, c1:c2] = 0
        
#        if social:
#            r1 = stim_ROIs[i, 1]
#            r2 = r1+stim_ROIs[i, 3]
#            c1 = stim_ROIs[i, 0]
#            c2 = c1+stim_ROIs[i, 2]
#            croppedStim[r1:r2, c1:c2] = 0
    
    summary = np.zeros((height, width, 3), dtype = float)
#    if social:
#        summary[:,:, 0] = croppedStim;
    
    summary[:,:, 1] = equ;
    summary[:,:, 2] = croppedTest;
    
    saveFolder = folder
    misc.imsave(saveFolder + r'/background_old.png', background)
    misc.imsave(saveFolder + r'/summary.png', summary)    
    cv2.cv.SaveImage(saveFolder + r'/background.png', cv2.cv.fromarray(background))
    # Using SciPy to save caused a weird rescaling when the images were dim.
    # This will change not only the background in the beginning but the threshold estimate

    return 0


def process_video_track_fish_3fish(folder, social, multiple):
    
    # Load -Initial- Background Frame (histogram from first 50 seconds)
    backgroundFile = folder + r'/background.png'
    background = misc.imread(backgroundFile, False)
    
    # Alloctae Space for Background Stack/Model
    stepFrames = 250 # Add a background frame every 2.5 seconds for 50 seconds
    bFrames = 20
    bCount = 0
    height = np.size(background, 0)
    width = np.size(background, 1)
    backgroundStack = np.zeros((height, width, bFrames), dtype = np.uint8)
    # Load Background Stack with "starting" background frame (Initialize)
    for i in range(0, bFrames):
        backgroundStack[:,:,i] = background
    
    # Load Test Crop Regions
    bonsaiFiles = glob.glob(folder+'/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    test_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    
#    Change to zero if NO CROPPING!!!
    test_ROIs = test_ROIs[1:, :]
#    test_ROIs = test_ROIs[0:, :]
#    print test_ROIs
    
    # Load Stim Crop Regions (if social experiment)
#    if social:
#        bonsaiFiles = glob.glob(folder+'/Social_Fish/*.bonsai')
#        bonsaiFiles = bonsaiFiles[0]
#        stim_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
#        stim_ROIs = stim_ROIs[1:, :]

    # Determine Thresholds
    test_thresholds = np.zeros(3)
    stim_thresholds = np.zeros(3)

    # Difference MEAN VS MEDIAN!    
    for i in range(0,3):
            crop, xOff, yOff = get_ROI_crop(background, test_ROIs, i)
            test_thresholds[i] = np.median(crop)/4            
#            if social:
#                crop, xOff, yOff = get_ROI_crop(background, stim_ROIs, i)
#                stim_thresholds[i] = np.mean(crop)/7            

    # Load Video
    aviFiles = glob.glob(folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    width = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    
    # Algorithm
    # 1. Subtract pre-computed Background frame from Current frame (Note: Not AbsDiff!)
    # 2. Extract Crop regions from ROIs
    # 3. Threshold ROI using mean/7 of each crop region, Binary Close image using 5 rad disc
    # 4. Find largest particle (Contour)
    # 5. - Compute Weighted Centroid (X,Y) for Eye Region (10% of brightest pixels)
    # 6. - Compute Binary Centroid of Body Region (50% of brightest pixels - eyeRegion)
    # 7. - Compute Heading
    
    # Alloctae Image Space
    subtraction = np.zeros((height, width), dtype = float)
    enhanced = np.zeros((height, width), dtype = float)
    
    # Allocate ROI (crop region) space (list of images)
    masked_TestROIs = []
    masked_StimROIs = []
    for i in range(0,3):
        w, h = get_ROI_size(test_ROIs, i)
        masked_TestROIs.append(np.zeros((h, w), dtype = np.uint8))
#        if social:
#            w, h = get_ROI_size(stim_ROIs, i)
#            masked_StimROIs.append(np.zeros((h, w), dtype = np.uint8))
    
    # Allocate Tracking Data Space (Test)
    fxS = np.zeros((numFrames,3))           # Fish X
    fyS = np.zeros((numFrames,3))           # Fish Y
    bxS = np.zeros((numFrames,3))           # Body X
    byS = np.zeros((numFrames,3))           # Body Y
    exS = np.zeros((numFrames,3))           # Eye X
    eyS = np.zeros((numFrames,3))           # Eye Y
    areaS = np.zeros((numFrames,3))         # area (-1 if error)
    ortS = np.zeros((numFrames,3))          # heading/orientation (angle from body to eyes)
    motS = np.zeros((numFrames,3))          # frame-by-frame change in segmented particle
    
     # Allocate Tracking Data Space (Stim)
#    if social:       
#        fxS_s = np.zeros((numFrames,6))           # Fish X
#        fyS_s = np.zeros((numFrames,6))           # Fish Y
#        bxS_s = np.zeros((numFrames,6))           # Body X
#        byS_s = np.zeros((numFrames,6))           # Body Y
#        exS_s = np.zeros((numFrames,6))           # Eye X
#        eyS_s = np.zeros((numFrames,6))           # Eye Y
#        areaS_s = np.zeros((numFrames,6))         # area (-1 if error)
#        ortS_s = np.zeros((numFrames,6))          # heading/orientation (angle from body to eyes)
#        motS_s = np.zeros((numFrames,6))          # frame-by-frame change in segmented particle
    
    # Toggle Display
    display = True
    if display:       
        plt.figure()  
    
    # Track within each ROI (test and, if available, stim)
    for f in range(0,numFrames):
        #vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, f)        
        
        # Read next frame        
        ret, im = vid.read()
        
        # Convert to grayscale (uint8)
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # Subtract Background (not absDiff, fish are always darker!)
        subtraction = cv2.subtract(background, current)
        
        # Process each ROI (Test and Stim (if social condition))
        for i in range(0,3):
            
            # Extract Crop Region
            crop, xOff, yOff = get_ROI_crop(subtraction, test_ROIs, i)
            
            # Threshold            
            level, threshold = cv2.threshold(crop,test_thresholds[i],255,cv2.THRESH_BINARY)    
            
            # Binary Close
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
            # Find Binary Contours            
            contours,hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            
            # If there are NO contours, then skip tracking
            if len(contours) == 0:
                if f!= 0:
                    area = -1.0
                    fX = fxS[f-1, i] - xOff
                    fY = fyS[f-1, i] - yOff
                    bX = bxS[f-1, i] - xOff
                    bY = byS[f-1, i] - yOff
                    eX = exS[f-1, i] - xOff
                    eY = eyS[f-1, i] - yOff
                    heading = ortS[f-1, i]
                    motion = -1.0
                else:
                    area = -1.0
                    fX = xOff
                    fY = yOff
                    bX = xOff
                    bY = yOff
                    eX = xOff
                    eY = yOff
                    heading = -181.0
                    motion = -1.0
            
            else:
                # Get Largest Contour (fish, ideally)
                largest_cnt, area = get_largest_contour(contours)
                
                # If the particle to too small to consider, skip frame
                if area == 0.0:
                    if f!= 0:
                        fX = fxS[f-1, i] - xOff
                        fY = fyS[f-1, i] - yOff
                        bX = bxS[f-1, i] - xOff
                        bY = byS[f-1, i] - yOff
                        eX = exS[f-1, i] - xOff
                        eY = eyS[f-1, i] - yOff
                        heading = ortS[f-1, i]
                        motion = -1.0
                    else:
                        area = -1.0
                        fX = xOff
                        fY = yOff
                        bX = xOff
                        bY = yOff
                        eX = xOff
                        eY = yOff
                        heading = -181.0
                        motion = -1.0
                        
                else:
                    # Create Binary Mask Image (1 for Fish, 0 for Background)
                    mask = np.zeros(crop.shape,np.uint8)
                    cv2.drawContours(mask,[largest_cnt],0,1,-1) # -1 draw the contour filled
                    pixelpoints = np.transpose(np.nonzero(mask))
                    
                    # Get Area (again)
                    area = np.size(pixelpoints, 0)
                    
                    # ---------------------------------------------------------------------------------
                    # Compute Frame-by-Frame Motion of Maksed Particle (ignore first frame)
                    currentMasked = mask * crop
                    if (f != 0):
                        absDiff = cv2.absdiff(masked_TestROIs[i], currentMasked)
                        level, threshold = cv2.threshold(absDiff,test_thresholds[i],255,cv2.THRESH_TOZERO)
                        motion = np.mean(threshold)
                    else:
                        motion = 0
                    
                    # Save Masked Fish Image from ROI (for subsequent frames motion calculation)
                    masked_TestROIs[i] = currentMasked
                    
                    # ---------------------------------------------------------------------------------
                    # Find Body and Eye Centroids
                    area = np.float(area)
                    
                    # Highlight 50% of the birghtest pixels (body + eyes)                    
                    numBodyPixels = np.ceil(area/2)
                    
                    # Highlight 10% of the birghtest pixels (mostly eyes)     
                    numEyePixels = np.ceil(area/10)
                    
                    # Fish Pixel Values
                    fishValues = crop[pixelpoints[:,0], pixelpoints[:,1]]
                    sortedFishValues = np.sort(fishValues)
                    
                    bodyThreshold = sortedFishValues[-numBodyPixels]                    
                    eyeThreshold = sortedFishValues[-numEyePixels]

                    # Compute Binary/Weighted Centroids
                    r = pixelpoints[:,0]
                    c = pixelpoints[:,1]
                    all_values = crop[r,c]
                    all_values = all_values.astype(float)
                    r = r.astype(float)
                    c = c.astype(float)
                    
                    # Fish Centroid
                    values = np.copy(all_values)
                    values = (values-test_thresholds[i]+1)
                    acc = np.sum(values)
                    fX = np.float(np.sum(c*values))/acc
                    fY = np.float(np.sum(r*values))/acc
                    
                    # Eye Centroid (a weighted centorid)
                    values = np.copy(all_values)                   
                    values = (values-eyeThreshold+1)
                    values[values < 0] = 0
                    acc = np.sum(values)
                    eX = np.float(np.sum(c*values))/acc
                    eY = np.float(np.sum(r*values))/acc
    
                    # Body Centroid (a binary centroid, excluding "eye" pixels)
                    values = np.copy(all_values)                   
                    values[values < bodyThreshold] = 0
                    values[values >= bodyThreshold] = 1                                                            
                    values[values > eyeThreshold] = 0                                                            
                    acc = np.sum(values)
                    bX = np.float(np.sum(c*values))/acc
                    bY = np.float(np.sum(r*values))/acc
                    
                    # ---------------------------------------------------------------------------------
                    # Heading (0 deg to right, 90 deg up)
                    if (bY != eY) or (eX != bX):
                        heading = math.atan2((bY-eY), (eX-bX)) * (360.0/(2*np.pi))
                    else:
                        heading = -181.00
            
            # ---------------------------------------------------------------------------------
            # Store data in arrays
            
            # Shift X,Y Values by ROI offset and store in Matrix
            fxS[f, i] = fX + xOff
            fyS[f, i] = fY + yOff
            bxS[f, i] = bX + xOff
            byS[f, i] = bY + yOff
            exS[f, i] = eX + xOff
            eyS[f, i] = eY + yOff
            areaS[f, i] = area
            ortS[f, i] = heading
            motS[f, i] = motion
            
            
# - - - - - -  
       
       # ---------------------------------------------------------------------------------
        # Recompute Background every ~2.5 seconds (exclude first frame) using a 25 second history
        if (f%stepFrames == 0) and (f != 0):
            backgroundStack[:,:,bCount] = current
            bCount = (bCount + 1)%bFrames
            
            # Compute Background Frame (median or mode)
            background = np.median(backgroundStack, axis = 2)
            background = background.astype(np.uint8)
            
        # ---------------------------------------------------------------------------------
        # Display Tracking
        plotType = 0
        
#        # Report Values
#        if display:
#            print fxS[f, 5], fyS[f, 5], exS[f, 5], eyS[f, 5], bxS[f, 5], byS[f, 5], areaS[f, 5], acc
#        
        # Plot All Fish in Movie with Tracking Overlay
        if display and (plotType == 0) and (f%stepFrames == 0):
            plt.clf()
            enhanced = cv2.multiply(subtraction, 5)
            color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            plt.imshow(color)
            plt.axis('image')
            for i in range(0,3):
                plt.plot(fxS[f, i],fyS[f, i],'m.')
                plt.plot(exS[f, i],eyS[f, i],'r.')
                plt.plot(bxS[f, i],byS[f, i],'g.')
                plt.text(bxS[f, i]+10,byS[f, i]+10,  '{0:.1f}'.format(ortS[f, i]), color = [1.0, 1.0, 0.0, 0.5])
                plt.text(bxS[f, i]+10,byS[f, i]+30,  '{0:.0f}'.format(areaS[f, i]), color = [1.0, 0.5, 0.0, 0.5])
#                print i,(fxS[f, i]
#            if social:
#                for i in range(0,3):
#                    plt.plot(fxS_s[f, i],fyS_s[f, i],'m.')
#                    plt.plot(exS_s[f, i],eyS_s[f, i],'r.')
#                    plt.plot(bxS_s[f, i],byS_s[f, i],'c.')
#                    plt.text(bxS_s[f, i]+10,byS_s[f, i]+10,  '{0:.1f}'.format(ortS_s[f, i]), color = [0.0, 1.0, 1.0, 0.5])
#                    plt.text(bxS_s[f, i]+10,byS_s[f, i]+30,  '{0:.0f}'.format(areaS_s[f, i]), color = [0.0, 0.5, 1.0, 0.5])
            plt.draw()
            plt.pause(0.001)
            
        # Plot only 1 fish in Movie, cropped
        if display and (plotType == 1) and (f%stepFrames == 0):
            # Extract Crop Region
            i = 0
            # Crop
            crop, xOff, yOff = get_ROI_crop(subtraction, test_ROIs, i)
            # Threshold
            level, threshold = cv2.threshold(crop,test_thresholds[i],255,cv2.THRESH_BINARY)
            # Binary Close
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            plt.clf()
            plt.subplot(1,2,1)
            color = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
            plt.imshow(color)
            plt.axis('image')
 
            plt.subplot(1,2,2)
            color = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
            plt.imshow(color)
            plt.axis('image')
            plt.plot(exS[f, i]-xOff,eyS[f, i]-yOff,'r.')
            plt.plot(bxS[f, i]-xOff,byS[f, i]-yOff,'c.')
            plt.text(bxS[f, i]-xOff+10,byS[f, i]-yOff+10,  '{0:.1f}'.format(ortS[f, i]), color = [0.0, 1.0, 1.0, 0.5])
            plt.text(bxS[f, i]-xOff+10,byS[f, i]-yOff+30,  '{0:.0f}'.format(areaS[f, i]), color = [0.0, 0.5, 1.0, 0.5])
            plt.draw()
            plt.pause(0.001)

        # Report Progress
        if (f%stepFrames) == 0:
            print (numFrames-f)
    
    # ---------------------------------------------------------------------------------
    # Close Video File
    vid.release()
    
    # Return Tracking Data!
#    if social:
#        return fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS, fxS_s, fyS_s, bxS_s, byS_s, exS_s, eyS_s, areaS_s, ortS_s, motS_s
#    else:
    return fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS




# Process Video : Track fish parameters within ROIs
def process_video_track_fish(folder, social, multiple):
    
    # Load -Initial- Background Frame (histogram from first 50 seconds)
    backgroundFile = folder + r'/background.png'
    background = misc.imread(backgroundFile, False)
    
    # Alloctae Space for Background Stack/Model
    #stepFrames = 250 # Add a background frame every 2.5 seconds for 50 seconds
    #bFrames = 20
    stepFrames = 1500 # Add a background frame every 15 seconds for 600 seconds
    bFrames = 40
    bCount = 0
    height = np.size(background, 0)
    width = np.size(background, 1)
    backgroundStack = np.zeros((height, width, bFrames), dtype = np.uint8)
    # Load Background Stack with "starting" background frame (Initialize)
    for i in range(0, bFrames):
        backgroundStack[:,:,i] = background
    
    # Load Test Crop Regions
    bonsaiFiles = glob.glob(folder+'/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    test_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    test_ROIs = test_ROIs[:, :]
    
    # Load Stim Crop Regions (if social experiment)
    if social:
        bonsaiFiles = glob.glob(folder+'/Social_Fish/*.bonsai')
        bonsaiFiles = bonsaiFiles[0]
        stim_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
        stim_ROIs = stim_ROIs[:, :]
        

    # Determine Thresholds
    test_thresholds = np.zeros(6)
    stim_thresholds = np.zeros(6)

    # Difference MEAN VS MEDIAN!    
    for i in range(0,6):
            crop, xOff, yOff = get_ROI_crop(background, test_ROIs, i)
            test_thresholds[i] = np.median(crop)/7            
            if social:
                crop, xOff, yOff = get_ROI_crop(background, stim_ROIs, i)
                stim_thresholds[i] = np.mean(crop)/7            

    # Load Video
    aviFiles = glob.glob(folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Algorithm
    # 1. Subtract pre-computed Background frame from Current frame (Note: Not AbsDiff!)
    # 2. Extract Crop regions from ROIs
    # 3. Threshold ROI using mean/7 of each crop region, Binary Close image using 5 rad disc
    # 4. Find largest particle (Contour)
    # 5. - Compute Weighted Centroid (X,Y) for Eye Region (10% of brightest pixels)
    # 6. - Compute Binary Centroid of Body Region (50% of brightest pixels - eyeRegion)
    # 7. - Compute Heading
    
    # Alloctae Image Space
    subtraction = np.zeros((height, width), dtype = float)
    enhanced = np.zeros((height, width), dtype = float)
    
    # Allocate ROI (crop region) space (list of images)
    masked_TestROIs = []
    masked_StimROIs = []
    for i in range(0,6):
        w, h = get_ROI_size(test_ROIs, i)
        masked_TestROIs.append(np.zeros((h, w), dtype = np.uint8))
        if social:
            w, h = get_ROI_size(stim_ROIs, i)
            masked_StimROIs.append(np.zeros((h, w), dtype = np.uint8))
    
    # Allocate Tracking Data Space (Test)
    fxS = np.zeros((numFrames,6))           # Fish X
    fyS = np.zeros((numFrames,6))           # Fish Y
    bxS = np.zeros((numFrames,6))           # Body X
    byS = np.zeros((numFrames,6))           # Body Y
    exS = np.zeros((numFrames,6))           # Eye X
    eyS = np.zeros((numFrames,6))           # Eye Y
    areaS = np.zeros((numFrames,6))         # area (-1 if error)
    ortS = np.zeros((numFrames,6))          # heading/orientation (angle from body to eyes)
    motS = np.zeros((numFrames,6))          # frame-by-frame change in segmented particle
    
     # Allocate Tracking Data Space (Stim)
    if social:       
        fxS_s = np.zeros((numFrames,6))           # Fish X
        fyS_s = np.zeros((numFrames,6))           # Fish Y
        bxS_s = np.zeros((numFrames,6))           # Body X
        byS_s = np.zeros((numFrames,6))           # Body Y
        exS_s = np.zeros((numFrames,6))           # Eye X
        eyS_s = np.zeros((numFrames,6))           # Eye Y
        areaS_s = np.zeros((numFrames,6))         # area (-1 if error)
        ortS_s = np.zeros((numFrames,6))          # heading/orientation (angle from body to eyes)
        motS_s = np.zeros((numFrames,6))          # frame-by-frame change in segmented particle
    
    # Toggle Display
    display = True
    if display:       
        plt.figure()  
    
    # Track within each ROI (test and, if available, stim)
    for f in range(0,numFrames):
        #vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, f)        
        
        # Read next frame        
        ret, im = vid.read()
        
        # Convert to grayscale (uint8)
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # Subtract Background (not absDiff, fish are always darker!)
        subtraction = cv2.subtract(background, current)
                
        # Process each ROI (Test and Stim (if social condition))
        for i in range(0,6):
            
            # Extract Crop Region
            crop, xOff, yOff = get_ROI_crop(subtraction, test_ROIs, i)
            
            # Threshold            
            level, threshold = cv2.threshold(crop,test_thresholds[i],255,cv2.THRESH_BINARY)    
            
            #print(test_ROIs)        
            #print(np.shape(threshold))

            
            # Binary Close
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
            # Find Binary Contours            
#            contours,hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#            contours,hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            
            
            # If there are NO contours, then skip tracking
            if len(contours) == 0:
                if f!= 0:
                    area = -1.0
                    fX = fxS[f-1, i] - xOff
                    fY = fyS[f-1, i] - yOff
                    bX = bxS[f-1, i] - xOff
                    bY = byS[f-1, i] - yOff
                    eX = exS[f-1, i] - xOff
                    eY = eyS[f-1, i] - yOff
                    heading = ortS[f-1, i]
                    motion = -1.0
                else:
                    area = -1.0
                    fX = xOff
                    fY = yOff
                    bX = xOff
                    bY = yOff
                    eX = xOff
                    eY = yOff
                    heading = -181.0
                    motion = -1.0
            
            else:
                # Get Largest Contour (fish, ideally)
                largest_cnt, area = get_largest_contour(contours)
                
                # If the particle to too small to consider, skip frame
                if area == 0.0:
                    if f!= 0:
                        fX = fxS[f-1, i] - xOff
                        fY = fyS[f-1, i] - yOff
                        bX = bxS[f-1, i] - xOff
                        bY = byS[f-1, i] - yOff
                        eX = exS[f-1, i] - xOff
                        eY = eyS[f-1, i] - yOff
                        heading = ortS[f-1, i]
                        motion = -1.0
                    else:
                        area = -1.0
                        fX = xOff
                        fY = yOff
                        bX = xOff
                        bY = yOff
                        eX = xOff
                        eY = yOff
                        heading = -181.0
                        motion = -1.0
                        
                else:
                    # Create Binary Mask Image (1 for Fish, 0 for Background)
                    mask = np.zeros(crop.shape,np.uint8)
                    cv2.drawContours(mask,[largest_cnt],0,1,-1) # -1 draw the contour filled
                    pixelpoints = np.transpose(np.nonzero(mask))
                    
                    # Get Area (again)
                    area = np.size(pixelpoints, 0)
                    
                    # ---------------------------------------------------------------------------------
                    # Compute Frame-by-Frame Motion of Maksed Particle (ignore first frame)
                    currentMasked = mask * crop
                    if (f != 0):
                        absDiff = cv2.absdiff(masked_TestROIs[i], currentMasked)
                        level, threshold = cv2.threshold(absDiff,test_thresholds[i],255,cv2.THRESH_TOZERO)
                        motion = np.mean(threshold)
                    else:
                        motion = 0
                    
                    # Save Masked Fish Image from ROI (for subsequent frames motion calculation)
                    masked_TestROIs[i] = currentMasked
                    
                    # ---------------------------------------------------------------------------------
                    # Find Body and Eye Centroids
                    area = np.float(area)
                    
                    # Highlight 50% of the birghtest pixels (body + eyes)                    
                    numBodyPixels = np.ceil(area/2)
                    
                    # Highlight 10% of the birghtest pixels (mostly eyes)     
                    numEyePixels = np.ceil(area/10)
                    
                    # Fish Pixel Values
                    fishValues = crop[pixelpoints[:,0], pixelpoints[:,1]]
                    sortedFishValues = np.sort(fishValues)
                    
                    bodyThreshold = sortedFishValues[-numBodyPixels]                    
                    eyeThreshold = sortedFishValues[-numEyePixels]

                    # Compute Binary/Weighted Centroids
                    r = pixelpoints[:,0]
                    c = pixelpoints[:,1]
                    all_values = crop[r,c]
                    all_values = all_values.astype(float)
                    r = r.astype(float)
                    c = c.astype(float)
                    
                    # Fish Centroid
                    values = np.copy(all_values)
                    values = (values-test_thresholds[i]+1)
                    acc = np.sum(values)
                    fX = np.float(np.sum(c*values))/acc
                    fY = np.float(np.sum(r*values))/acc
                    
                    # Eye Centroid (a weighted centorid)
                    values = np.copy(all_values)                   
                    values = (values-eyeThreshold+1)
                    values[values < 0] = 0
                    acc = np.sum(values)
                    eX = np.float(np.sum(c*values))/acc
                    eY = np.float(np.sum(r*values))/acc
    
                    # Body Centroid (a binary centroid, excluding "eye" pixels)
                    values = np.copy(all_values)                   
                    values[values < bodyThreshold] = 0
                    values[values >= bodyThreshold] = 1                                                            
                    values[values > eyeThreshold] = 0                                                            
                    acc = np.sum(values)
                    bX = np.float(np.sum(c*values))/acc
                    bY = np.float(np.sum(r*values))/acc
                    
                    # ---------------------------------------------------------------------------------
                    # Heading (0 deg to right, 90 deg up)
                    if (bY != eY) or (eX != bX):
                        heading = math.atan2((bY-eY), (eX-bX)) * (360.0/(2*np.pi))
                    else:
                        heading = -181.00
            
            # ---------------------------------------------------------------------------------
            # Store data in arrays
            
            # Shift X,Y Values by ROI offset and store in Matrix
            fxS[f, i] = fX + xOff
            fyS[f, i] = fY + yOff
            bxS[f, i] = bX + xOff
            byS[f, i] = bY + yOff
            exS[f, i] = eX + xOff
            eyS[f, i] = eY + yOff
            areaS[f, i] = area
            ortS[f, i] = heading
            motS[f, i] = motion
            
            
# - - - - - -             
            
            # ---------------------------------------------------------------------------------
            # Now repeat for social tracking!
            if social:

                # Extract Crop Region
                crop, xOff, yOff = get_ROI_crop(subtraction, stim_ROIs, i)
                
                # Threshold
                level, threshold = cv2.threshold(crop,stim_thresholds[i],255,cv2.THRESH_BINARY)
                
#                print(stim_ROIs)
#                print("Thresh Channels:" + str(np.shape(threshold)))
                
                # Binary Close
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
                closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
                # Find Binary Contours            
                contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            
                # If there are NO contours, then skip tracking
                if len(contours) == 0:
                    if f!= 0:
                        area = -1.0
                        fX = fxS_s[f-1, i] - xOff
                        fY = fyS_s[f-1, i] - yOff
                        bX = bxS_s[f-1, i] - xOff
                        bY = byS_s[f-1, i] - yOff
                        eX = exS_s[f-1, i] - xOff
                        eY = eyS_s[f-1, i] - yOff
                        heading = ortS_s[f-1, i]
                        motion = -1.0               
                    else:
                        area = -1.0
                        fX = xOff
                        fY = yOff
                        bX = xOff
                        bY = yOff
                        eX = xOff
                        eY = yOff
                        heading = -181.0
                        motion = -1.0
                    
                else:
                    # Get Largest Contour (fish, ideally)
                    largest_cnt, area = get_largest_contour(contours)
                    
                    # If the particle to too small to consider, skip frame
                    if area == 0:
                        if f!= 0:
                            fX = fxS_s[f-1, i] - xOff
                            fY = fyS_s[f-1, i] - yOff
                            bX = bxS_s[f-1, i] - xOff
                            bY = byS_s[f-1, i] - yOff
                            eX = exS_s[f-1, i] - xOff
                            eY = eyS_s[f-1, i] - yOff
                            heading = ortS_s[f-1, i]
                            motion = -1.0
                        else:
                            area = -1.0
                            fX = xOff
                            fY = yOff
                            bX = xOff
                            bY = yOff
                            eX = xOff
                            eY = yOff
                            heading = -181.0
                            motion = -1.0
                            
                    else:
                        # Create Binary Mask Image (1 for Fish, 0 for Background)
                        mask = np.zeros(crop.shape,np.uint8)
                        cv2.drawContours(mask,[largest_cnt],0,1,-1) # -1 draw the contour filled
                        pixelpoints = np.transpose(np.nonzero(mask))
                        
                        # If MULTIPLE (i.e. multiple fish experiment as Stimulus)
                        if multiple:
                            # Create Binary Mask Image (1 for ALL Fish, 0 for Background)
                            mask = np.zeros(crop.shape,np.uint8)
                            level, mask = cv2.threshold(crop,stim_thresholds[i],255,cv2.THRESH_BINARY)
                            pixelpoints = np.transpose(np.nonzero(mask))
                                                      
                        # Get Area (again)
                        area = np.size(pixelpoints, 0)
                        
                        # ---------------------------------------------------------------------------------
                        # Compute Frame-by-Frame Motion of Maksed Particle (ignore first frame)
                        currentMasked = mask * crop
                        if (f != 0):
                            absDiff = cv2.absdiff(masked_StimROIs[i], currentMasked)
                            level, threshold = cv2.threshold(absDiff,stim_thresholds[i],255,cv2.THRESH_TOZERO)
                            motion = np.mean(threshold)
                        else:
                            motion = 0
                        
                        # Save Masked Fish Image from ROI (for subsequent frames motion calculation)
                        masked_StimROIs[i] = currentMasked
                        
                        # ---------------------------------------------------------------------------------
                        # Find Body and Eye Centroids
                        area = np.float(area)
                                            
                        # Highlight 50% of the birghtest pixels (body + eyes)                    
                        numBodyPixels = np.ceil(area/2)
                        
                        # Highlight 10% of the birghtest pixels (mostly eyes)     
                        numEyePixels = np.ceil(area/10)
                        
                        # Fish Pixel Values
                        fishValues = crop[pixelpoints[:,0], pixelpoints[:,1]]
                        sortedFishValues = np.sort(fishValues)
                        
                        bodyThreshold = sortedFishValues[-numBodyPixels]                   
                        eyeThreshold = sortedFishValues[-numEyePixels]
    
                        # Compute Binary/Weighted Centroids
                        r = pixelpoints[:,0]
                        c = pixelpoints[:,1]
                        all_values = crop[r,c]
                        all_values = all_values.astype(float)
                        r = r.astype(float)
                        c = c.astype(float)
                        
                        # Fish Centroid
                        values = np.copy(all_values)
                        values = (values-test_thresholds[i]+1)
                        acc = np.sum(values)
                        fX = np.float(np.sum(c*values))/acc
                        fY = np.float(np.sum(r*values))/acc
                        
                        # Eye Centroid (a weighted centorid)
                        values = np.copy(all_values)                   
                        values = (values-eyeThreshold+1)
                        values[values < 0] = 0
                        acc = np.sum(values)
                        eX = np.float(np.sum(c*values))/acc
                        eY = np.float(np.sum(r*values))/acc
        
                        # Body Centroid (a binary centroid, excluding "eye" pixels)
                        values = np.copy(all_values)                   
                        values[values < bodyThreshold] = 0
                        values[values >= bodyThreshold] = 1                                                            
                        values[values > eyeThreshold] = 0                                                            
                        acc = np.sum(values)
                        bX = np.float(np.sum(c*values))/acc
                        bY = np.float(np.sum(r*values))/acc
                        
                        # ---------------------------------------------------------------------------------
                        # Heading (0 deg to right, 90 deg up)
                        if (bY != eY) or (eX != bX):
                            heading = math.atan2((bY-eY), (eX-bX)) * (360.0/(2*np.pi))
                        else:
                            heading = -181.00
                
                # ---------------------------------------------------------------------------------
                # Store data in arrays
                
                # Shift X,Y Values by ROI offset and store in Matrix
                fxS_s[f, i] = fX + xOff
                fyS_s[f, i] = fY + yOff
                bxS_s[f, i] = bX + xOff
                byS_s[f, i] = bY + yOff
                exS_s[f, i] = eX + xOff
                eyS_s[f, i] = eY + yOff
                areaS_s[f, i] = area
                ortS_s[f, i] = heading
                motS_s[f, i] = motion 
                
       
       # ---------------------------------------------------------------------------------
        # Recompute Background every ~2.5 seconds (exclude first frame) using a 25 second history
        if (f%stepFrames == 0) and (f != 0):
            backgroundStack[:,:,bCount] = current
            bCount = (bCount + 1)%bFrames
            
            # Compute Background Frame (median or mode)
            background = np.median(backgroundStack, axis = 2)
            background = background.astype(np.uint8)
            
        # ---------------------------------------------------------------------------------
        # Display Tracking
        plotType = 0
        
#        # Report Values
#        if display:
#            print fxS[f, 5], fyS[f, 5], exS[f, 5], eyS[f, 5], bxS[f, 5], byS[f, 5], areaS[f, 5], acc
#        
        # Plot All Fish in Movie with Tracking Overlay
        if display and (plotType == 0) and (f%stepFrames == 0):
            plt.clf()
            enhanced = cv2.multiply(subtraction, 5)
            color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            plt.imshow(color)
            plt.axis('image')
            for i in range(0,6):
                plt.plot(fxS[f, i],fyS[f, i],'m.')
                plt.plot(exS[f, i],eyS[f, i],'r.')
                plt.plot(bxS[f, i],byS[f, i],'g.')
                plt.text(bxS[f, i]+10,byS[f, i]+10,  '{0:.1f}'.format(ortS[f, i]), color = [1.0, 1.0, 0.0, 0.5])
                plt.text(bxS[f, i]+10,byS[f, i]+30,  '{0:.0f}'.format(areaS[f, i]), color = [1.0, 0.5, 0.0, 0.5])
                
            if social:
                for i in range(0,6):
                    plt.plot(fxS_s[f, i],fyS_s[f, i],'m.')
                    plt.plot(exS_s[f, i],eyS_s[f, i],'r.')
                    plt.plot(bxS_s[f, i],byS_s[f, i],'c.')
                    plt.text(bxS_s[f, i]+10,byS_s[f, i]+10,  '{0:.1f}'.format(ortS_s[f, i]), color = [0.0, 1.0, 1.0, 0.5])
                    plt.text(bxS_s[f, i]+10,byS_s[f, i]+30,  '{0:.0f}'.format(areaS_s[f, i]), color = [0.0, 0.5, 1.0, 0.5])
            plt.draw()
            plt.pause(0.001)
            
        # Plot only 1 fish in Movie, cropped
        if display and (plotType == 1) and (f%stepFrames == 0):
            # Extract Crop Region
            i = 0
            # Crop
            crop, xOff, yOff = get_ROI_crop(subtraction, test_ROIs, i)
            # Threshold
            level, threshold = cv2.threshold(crop,test_thresholds[i],255,cv2.THRESH_BINARY)
            # Binary Close
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            plt.clf()
            plt.subplot(1,2,1)
            color = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
            plt.imshow(color)
            plt.axis('image')
 
            plt.subplot(1,2,2)
            color = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
            plt.imshow(color)
            plt.axis('image')
            plt.plot(exS[f, i]-xOff,eyS[f, i]-yOff,'r.')
            plt.plot(bxS[f, i]-xOff,byS[f, i]-yOff,'c.')
            plt.text(bxS[f, i]-xOff+10,byS[f, i]-yOff+10,  '{0:.1f}'.format(ortS[f, i]), color = [0.0, 1.0, 1.0, 0.5])
            plt.text(bxS[f, i]-xOff+10,byS[f, i]-yOff+30,  '{0:.0f}'.format(areaS[f, i]), color = [0.0, 0.5, 1.0, 0.5])
            plt.draw()
            plt.pause(0.001)

        # Report Progress
        if (f%stepFrames) == 0:
            print (numFrames-f)
    
    # ---------------------------------------------------------------------------------
    # Close Video File
    vid.release()
    
    # Return Tracking Data!
    if social:
        return fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS, fxS_s, fyS_s, bxS_s, byS_s, exS_s, eyS_s, areaS_s, ortS_s, motS_s
    else:
        return fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS
#------------------------------------------------------------------------------

# Compute the initial background for each ROI
def compute_intial_backgrounds(folder, ROIs):

    # Load Video
    aviFiles = glob.glob(folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Allocate space for all ROI backgrounds
    background_ROIs = []
    for i in range(0,6):
        w, h = get_ROI_size(ROIs, i)
        background_ROIs.append(np.zeros((h, w), dtype = np.float32))
    
    # Find initial background for each ROI
    for i in range(0,6):

        # Allocate space for background stack
        crop_width, crop_height = get_ROI_size(ROIs, i)
        stepFrames = 1000 # Check background frame every 10 seconds
        bFrames = 20
        backgroundStack = np.zeros((crop_height, crop_width, bFrames), dtype = np.float32)
        background = np.zeros((crop_height, crop_width), dtype = np.float32)
        previous = np.zeros((crop_height, crop_width), dtype = np.float32)
        
        # Store first frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, im = vid.read()
        current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        crop, xOff, yOff = get_ROI_crop(current, ROIs, i)
        backgroundStack[:,:,0] = np.copy(crop)
        previous = np.copy(crop)
        bCount = 1
        
        # Search for useful background frames (significantly different than previous)
        changes = []
        for f in range(stepFrames, numFrames, stepFrames):

            # Read frame
            vid.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, im = vid.read()
            current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            crop, xOff, yOff = get_ROI_crop(current, ROIs, i)
        
            # Measure change from current to previous frame
            absdiff = np.abs(previous-crop)
            level = np.median(crop)/7
            change = np.mean(absdiff > level)
            changes.append(change)
            previous = np.copy(crop)
            
            # If significant, add to stack...possible finish
            if(change > 0.0075):
                backgroundStack[:,:,bCount] = np.copy(crop)
                bCount = bCount + 1
                if(bCount == bFrames):
                    print("Background for ROI(" + str(i) + ") found on frame " + str(f))
                    break
        
        # Compute background
        backgroundStack = backgroundStack[:,:, 0:bCount]
        background_ROIs[i] = np.median(backgroundStack, axis=2)
                        
    # Return initial background
    return background_ROIs
#------------------------------------------------------------------------------


# Process Video : Track fish in AVI
def improved_fish_tracking(input_folder, output_folder, ROIs):

    # Compute a "Starting" Background
    # - Median value of 20 frames with significant difference between them
    background_ROIs = compute_intial_backgrounds(input_folder, ROIs)
    
    # Algorithm
    # 1. Find initial background guess for each ROI
    # 2. Extract Crop regions from ROIs
    # 3. Threshold ROI using median/7 of each crop region, Binary Close image using 5 rad disc
    # 4. Find largest particle (Contour)
    # 5. - Compute Weighted Centroid (X,Y) for Eye Region (10% of brightest pixels)
    # 6. - Compute Binary Centroid of Body Region (50% of brightest pixels - eyeRegion)
    # 7. - Compute Heading
    
    # Load Video
    aviFiles = glob.glob(input_folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    # Allocate ROI (crop region) space
    previous_ROIs = []
    for i in range(0,6):
        w, h = get_ROI_size(ROIs, i)
        previous_ROIs.append(np.zeros((h, w), dtype = np.uint8))
    
    # Allocate Tracking Data Space
    fxS = np.zeros((numFrames,6))           # Fish X
    fyS = np.zeros((numFrames,6))           # Fish Y
    bxS = np.zeros((numFrames,6))           # Body X
    byS = np.zeros((numFrames,6))           # Body Y
    exS = np.zeros((numFrames,6))           # Eye X
    eyS = np.zeros((numFrames,6))           # Eye Y
    areaS = np.zeros((numFrames,6))         # area (-1 if error)
    ortS = np.zeros((numFrames,6))          # heading/orientation (angle from body to eyes)
    motS = np.zeros((numFrames,6))          # frame-by-frame change in segmented particle
        
    # Track within each ROI
    plt.figure(figsize=(8,6))
    for f in range(0,numFrames):
        
        # Read next frame        
        ret, im = vid.read()
        
        # Convert to grayscale (uint8)
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        
        # Process each ROI
        for i in range(0,6):
            
            print('Processing ROI ' + str(i+1))            
            # Extract Crop Region
            crop, xOff, yOff = get_ROI_crop(current, ROIs, i)
            crop_height, crop_width = np.shape(crop)

            # Difference from current background
            diff = background_ROIs[i] - crop
            
            # Determine current threshold
            threshold_level = np.median(background_ROIs[i])/7            
   
            # Threshold            
            level, threshold = cv2.threshold(diff,threshold_level,255,cv2.THRESH_BINARY)
            
            # Convert to uint8
            threshold = np.uint8(threshold)
            
            # Binary Close
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
            # Find Binary Contours            
            contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            
            # Create Binary Mask Image
            mask = np.zeros(crop.shape,np.uint8)
                       
            # If there are NO contours, then skip tracking
            if len(contours) == 0:
                if f!= 0:
                    area = -1.0
                    fX = fxS[f-1, i] - xOff
                    fY = fyS[f-1, i] - yOff
                    bX = bxS[f-1, i] - xOff
                    bY = byS[f-1, i] - yOff
                    eX = exS[f-1, i] - xOff
                    eY = eyS[f-1, i] - yOff
                    heading = ortS[f-1, i]
                    motion = -1.0
                else:
                    area = -1.0
                    fX = xOff
                    fY = yOff
                    bX = xOff
                    bY = yOff
                    eX = xOff
                    eY = yOff
                    heading = -181.0
                    motion = -1.0
            
            else:
                # Get Largest Contour (fish, ideally)
                largest_cnt, area = get_largest_contour(contours)
                
                # If the particle to too small to consider, skip frame
                if area == 0.0:
                    if f!= 0:
                        fX = fxS[f-1, i] - xOff
                        fY = fyS[f-1, i] - yOff
                        bX = bxS[f-1, i] - xOff
                        bY = byS[f-1, i] - yOff
                        eX = exS[f-1, i] - xOff
                        eY = eyS[f-1, i] - yOff
                        heading = ortS[f-1, i]
                        motion = -1.0
                    else:
                        area = -1.0
                        fX = xOff
                        fY = yOff
                        bX = xOff
                        bY = yOff
                        eX = xOff
                        eY = yOff
                        heading = -181.0
                        motion = -1.0
                        
                else:
                    # Draw contours into Mask Image (1 for Fish, 0 for Background)
                    cv2.drawContours(mask,[largest_cnt],0,1,-1) # -1 draw the contour filled
                    pixelpoints = np.transpose(np.nonzero(mask))
                    
                    # Get Area (again)
                    area = np.size(pixelpoints, 0)
                    
                    # ---------------------------------------------------------------------------------
                    # Compute Frame-by-Frame Motion (absolute changes above threshold)
                    # - Normalize by total absdiff from background
                    if (f != 0):
                        absdiff = np.abs(diff)
                        absdiff[absdiff < threshold_level] = 0
                        totalAbsDiff = np.sum(np.abs(absdiff))
                        frame_by_frame_absdiff = np.abs(np.float32(previous_ROIs[i]) - np.float32(crop)) / 2 # Adjust for increases and decreases across frames
                        frame_by_frame_absdiff[frame_by_frame_absdiff < threshold_level] = 0
                        motion = np.sum(np.abs(frame_by_frame_absdiff))/totalAbsDiff
                    else:
                        motion = 0
                    
                    # Save Masked Fish Image from ROI (for subsequent frames motion calculation)
                    previous_ROIs[i] = np.copy(crop)
                    
                    # ---------------------------------------------------------------------------------
                    # Find Body and Eye Centroids
                    area = np.float(area)
                    
                    # Highlight 50% of the birghtest pixels (body + eyes)                    
                    numBodyPixels = np.int(np.ceil(area/2))
                    
                    # Highlight 10% of the birghtest pixels (mostly eyes)     
                    numEyePixels = np.int(np.ceil(area/10))
                    
                    # Fish Pixel Values (difference from background)
                    fishValues = diff[pixelpoints[:,0], pixelpoints[:,1]]
                    sortedFishValues = np.sort(fishValues)
                    
                    bodyThreshold = sortedFishValues[-numBodyPixels]                    
                    eyeThreshold = sortedFishValues[-numEyePixels]

                    # Compute Binary/Weighted Centroids
                    r = pixelpoints[:,0]
                    c = pixelpoints[:,1]
                    all_values = diff[r,c]
                    all_values = all_values.astype(float)
                    r = r.astype(float)
                    c = c.astype(float)
                    
                    # Fish Centroid
                    values = np.copy(all_values)
                    values = (values-threshold_level+1)
                    acc = np.sum(values)
                    fX = np.float(np.sum(c*values))/acc
                    fY = np.float(np.sum(r*values))/acc
                    
                    # Eye Centroid (a weighted centorid)
                    values = np.copy(all_values)                   
                    values = (values-eyeThreshold+1)
                    values[values < 0] = 0
                    acc = np.sum(values)
                    eX = np.float(np.sum(c*values))/acc
                    eY = np.float(np.sum(r*values))/acc
    
                    # Body Centroid (a binary centroid, excluding "eye" pixels)
                    values = np.copy(all_values)                   
                    values[values < bodyThreshold] = 0
                    values[values >= bodyThreshold] = 1                                                            
                    values[values > eyeThreshold] = 0                                                            
                    acc = np.sum(values)
                    bX = np.float(np.sum(c*values))/acc
                    bY = np.float(np.sum(r*values))/acc
                    
                    # ---------------------------------------------------------------------------------
                    # Heading (0 deg to right, 90 deg up)
                    if (bY != eY) or (eX != bX):
                        heading = math.atan2((bY-eY), (eX-bX)) * (360.0/(2*np.pi))
                    else:
                        heading = -181.00
            
            # ---------------------------------------------------------------------------------
            # Store data in arrays
            
            # Shift X,Y Values by ROI offset and store in Matrix
            fxS[f, i] = fX + xOff
            fyS[f, i] = fY + yOff
            bxS[f, i] = bX + xOff
            byS[f, i] = bY + yOff
            exS[f, i] = eX + xOff
            eyS[f, i] = eY + yOff
            areaS[f, i] = area
            ortS[f, i] = heading
            motS[f, i] = motion
            
            # -----------------------------------------------------------------
            # Update this ROIs background estimate (everywhere except the (dilated) Fish)
            current_background = np.copy(background_ROIs[i])            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
            dilated_fish = cv2.dilate(mask, kernel, iterations = 2)           
            updated_background = (np.float32(crop) * 0.01) + (current_background * 0.99)
            updated_background[dilated_fish==1] = current_background[dilated_fish==1]            
            background_ROIs[i] = np.copy(updated_background)
            
            
        # ---------------------------------------------------------------------------------
        # Plot All Fish in Movie with Tracking Overlay
        if (f % 100 == 0):
            plt.clf()
            enhanced = cv2.multiply(current, 1)
            color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            plt.imshow(color)
            plt.axis('image')
            for i in range(0,6):
                plt.plot(fxS[f, i],fyS[f, i],'b.', MarkerSize = 1)
                plt.plot(exS[f, i],eyS[f, i],'r.', MarkerSize = 3)
                plt.plot(bxS[f, i],byS[f, i],'co', MarkerSize = 3)
                plt.text(bxS[f, i]+10,byS[f, i]+10,  '{0:.1f}'.format(ortS[f, i]), color = [1.0, 1.0, 0.0, 0.5])
                plt.text(bxS[f, i]+10,byS[f, i]+30,  '{0:.0f}'.format(areaS[f, i]), color = [1.0, 0.5, 0.0, 0.5])
            plt.draw()
            plt.pause(0.001)
            
        # ---------------------------------------------------------------------------------
        # Save Tracking Summary
        if(f == 0):
            plt.savefig(output_folder+'/initial_tracking.png', dpi=300)
            plt.figure('backgrounds')
            for i in range(0,6):
                plt.subplot(2,3,i+1)
                plt.imshow(background_ROIs[i])
            plt.savefig(output_folder+'/initial_backgrounds.png', dpi=300)
            plt.close('backgrounds')
        if(f == numFrames-1):
            plt.savefig(output_folder+'/final_tracking.png', dpi=300)
            plt.figure('backgrounds')
            for i in range(0,6):
                plt.subplot(2,3,i+1)
                plt.imshow(background_ROIs[i])
            plt.savefig(output_folder+'/final_backgrounds.png', dpi=300)
            plt.close('backgrounds')

        # Report Progress
        if (f%100) == 0:
            bs = '\b' * 1000            # The backspace
            print(bs)
            print (numFrames-f)
    
    # -------------------------------------------------------------------------
    # Close Video File
    vid.release()
    
    # Return tracking data
    return fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS
#------------------------------------------------------------------------------
 
# Return cropped image from ROI list
def get_ROI_crop(image, ROIs, numROi):
    r1 = np.int(ROIs[numROi, 1])
    r2 = np.int(r1+ROIs[numROi, 3])
    c1 = np.int(ROIs[numROi, 0])
    c2 = np.int(c1+ROIs[numROi, 2])
    crop = image[r1:r2, c1:c2]
    
    return crop, c1, r1
    
# Return ROI size from ROI list
def get_ROI_size(ROIs, numROi):
    width = np.int(ROIs[numROi, 2])
    height = np.int(ROIs[numROi, 3])
    
    return width, height

# Return largest (area) cotour from contour list
def get_largest_contour(contours):
    # Find contour with maximum area and store it as best_cnt
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
    if max_area > 0:
        return best_cnt, max_area
    else:
        return cnt, max_area

# Compare Old (scaled) and New (non-scaled) background images
def compare_backgrounds(folder):

    # Load -Initial- Background Frame (histogram from first 50 seconds)
    backgroundFile = folder + r'/background_old.png'
    background_old = misc.imread(backgroundFile, False)
    
    backgroundFile = folder + r'/background.png'
    background = misc.imread(backgroundFile, False)
    absDiff = cv2.absdiff(background_old, background)

    return np.mean(absDiff)

# FIN
