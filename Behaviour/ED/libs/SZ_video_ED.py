# -*- coding: utf-8 -*-
"""
Created on Nov 10 2024

@author: dreostilab (Elena Dreosti)
"""
# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import glob
import cv2
import BONSAI_ARK_ED
import math
import matplotlib.patches as patches

# Utilities for processing videos of Social Experiments

# INDEX:
# 1. process_video_summary_images: Process Video - Make Summary Images
# 2. improved_fish_tracking: Process Video - Track fish in AVI
# 3. compute_intial_backgrounds: Compute the initial background for each ROI
# 4. get_ROI_size: Return ROI size from ROI list
# 5. get_ROI_crop: Return cropped image from ROI list
# 6. get_largest_contour: return largest (area) cotour from contour list



# 1. Process Video : Make Summary Images
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
    # plt.figure()    # make a figure
    # plt.imshow(background, cmap = plt.cm.gray, vmin = 0, vmax = 255) # plot the image using gray scale from cmap, set min and max
    # plt.figure()
    #plt.imshow(equ, cmap = plt.cm.gray, vmin = 0, vmax = 255) 
    #ax = plt.gca()
    #plt.show()
    
    # plt.draw() # This is to update the figure
    # plt.pause(0.0001) #This tells how often to update the figure in sec?
    
    # Show Crop Regions

    # Load Test Crop Regions
    bonsaiFiles = glob.glob(folder+'/*.bonsai') # find the path of the file
    bonsaiFiles = bonsaiFiles[0] # Get the first path name. It assumes there is only 1 per folder
    test_ROIs = BONSAI_ARK_ED.read_bonsai_crop_rois(bonsaiFiles)
    #test_ROIs = test_ROIs[:, :]
    croppedTest = np.copy(background) # Copy the background image
    #print(test_ROIs)
    
    # Load Stim Crop Regions
    if social:
        bonsaiFiles = glob.glob(folder+'/Social_Fish/*.bonsai')
        bonsaiFiles = bonsaiFiles[0]
        stim_ROIs = BONSAI_ARK_ED.read_bonsai_crop_rois(bonsaiFiles)
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
            r3 = int(stim_ROIs[i, 3])
            c1 = int(stim_ROIs[i, 0])
            c2 = int(c1+stim_ROIs[i, 2])
            c3 = int(stim_ROIs[i, 2])
            croppedStim[r1:r2, c1:c2] = 0
    
#    
    summary = np.zeros((height, width, 3), dtype = float) # make a 3D arrays that has all the r1

    if social:
        summary[:,:, 0] = croppedStim;
    
    summary[:,:, 1] = equ;
    summary[:,:, 2] = croppedTest;

    summary = summary / np.max(summary)
   
    # Save images
    saveFolder = folder
    plt.imsave(saveFolder + '/background2.png', background, cmap='gray')
    cv2.imwrite(saveFolder + '/background.png', background) # Used to do more analysis later
    plt.imsave(saveFolder + '/difference.png', equ, cmap='cividis')
    plt.imsave(saveFolder + r'/summary.png', summary)
    #plt.imsave(saveFolder + r'/summary2.png', summaryEqu)
    #plt.imsave(saveFolder + r'/summary3.png', summaryTest)
    
    
    # Be aware that **plt.imsave** Automatically normalizes data if the input image array has values outside the [0, 1] or [0, 255] range. 
    # For example, if you provide float data, plt.imsave will scale it to fit the image intensity range.
    # **cv2.imwrite** Expects the data to be in a specific range: [0, 255] for 8-bit images. If the data isn't in this range, you'll need 
    # to normalize or scale it manually, or cv2.imwrite may not save it correctly.

    return 0
#END 

####============

# 1B. Process Video : Make Summary Images
def process_video_summary_images_modified(folder, social=False):
    
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
    # plt.figure()    # make a figure
    # plt.imshow(background, cmap = plt.cm.gray, vmin = 0, vmax = 255) # plot the image using gray scale from cmap, set min and max


    plt.imshow(equ, cmap = plt.cm.gray, vmin = 0, vmax = 255) 
  
    ax = plt.gca() #get the current axis
    #plt.show()
    
    # plt.draw() # This is to update the figure
    # plt.pause(0.0001) #This tells how often to update the figure in sec?
    
    # Show Crop Regions

    # Load Test Crop Regions
    bonsaiFiles = glob.glob(folder+'/*.bonsai') # find the path of the file
    bonsaiFiles = bonsaiFiles[0] # Get the first path name. It assumes there is only 1 per folder
    test_ROIs = BONSAI_ARK_ED.read_bonsai_crop_rois(bonsaiFiles)
    #test_ROIs = test_ROIs[:, :]
    croppedTest = np.copy(background) # Copy the background image
    #print(test_ROIs)
    
    # Load Stim Crop Regions
    if social:
        bonsaiFiles = glob.glob(folder+'/Social_Fish/*.bonsai')
        bonsaiFiles = bonsaiFiles[0]
        stim_ROIs = BONSAI_ARK_ED.read_bonsai_crop_rois(bonsaiFiles)
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
            r3 = int(stim_ROIs[i, 3])
            c1 = int(stim_ROIs[i, 0])
            c2 = int(c1+stim_ROIs[i, 2])
            c3 = int(stim_ROIs[i, 2])
            croppedStim[r1:r2, c1:c2] = 0

            # Extract parameters for the rectangle
            x1 = int(stim_ROIs[i, 0])  # X-coordinate
            y1 = int(stim_ROIs[i, 1])  # Y-coordinate
            width1 = int(stim_ROIs[i, 2])  # Width
            height1 = int(stim_ROIs[i, 3])  # Height


            for i in stim_ROIs[i]:
                rect = patches.Rectangle((x1, y1), width1, height1, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
        plt.draw() 
    summary = np.zeros((height, width, 3), dtype = float) # make a 3D arrays that has all the r1

    if social:
        summary[:,:, 0] = croppedStim;
    
    summary[:,:, 1] = equ;
    summary[:,:, 2] = croppedTest;

    summary = summary / np.max(summary)
   
    # Save images
    saveFolder = folder
    plt.savefig(saveFolder + '/Stim_ROIS.png', dpi=300)
    #plt.show() 
    plt.imsave(saveFolder + '/background2.png', background, cmap='gray')
    cv2.imwrite(saveFolder + '/background.png', background) # Used to do more analysis later
    plt.imsave(saveFolder + '/difference.png', equ, cmap='cividis')
    plt.imsave(saveFolder + r'/summary.png', summary)
    #plt.imsave(saveFolder + r'/summary2.png', summaryEqu)
    #plt.imsave(saveFolder + r'/summary3.png', summaryTest)
    plt.close() 
    
    # Be aware that **plt.imsave** Automatically normalizes data if the input image array has values outside the [0, 1] or [0, 255] range. 
    # For example, if you provide float data, plt.imsave will scale it to fit the image intensity range.
    # **cv2.imwrite** Expects the data to be in a specific range: [0, 255] for 8-bit images. If the data isn't in this range, you'll need 
    # to normalize or scale it manually, or cv2.imwrite may not save it correctly.

    return 0
#END 






####============================================

# 2. improved_fish_tracking: Process Video - Track fish in AVI
def improved_fish_tracking(input_folder, output_folder, ROIs, debug=False):

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
        w, h = get_ROI_size(ROIs, i) # function that makes two list of all 6 widths and heights 
        previous_ROIs.append(np.zeros((h, w), dtype = np.uint8)) #make a single array that has heigth and width after
    
    # Allocate Tracking Data Space. Male all empty arrays 
    fxS = np.zeros((numFrames,6))           # Fish X pixel position
    fyS = np.zeros((numFrames,6))           # Fish Y
    bxS = np.zeros((numFrames,6))           # Body X
    byS = np.zeros((numFrames,6))           # Body Y
    exS = np.zeros((numFrames,6))           # Eye X
    eyS = np.zeros((numFrames,6))           # Eye Y
    areaS = np.zeros((numFrames,6))         # area (-1 if error)
    ortS = np.zeros((numFrames,6))          # heading/orientation (angle from body to eyes)
    motS = np.zeros((numFrames,6))          # frame-by-frame change in segmented particle
        
    # Track within each ROI
    plt.figure(figsize=(8,6)) # this sets the figure size of the plot (8 inches by 6 inches)
    for f in range(0, numFrames):
        
        # Read next frame        
        ret, im = vid.read() # get one frame of the movie

        # Convert to grayscale (uint8)
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Process each ROI
        for i in range(0,6):
            
            # print('Processing ROI ' + str(i+1))   
                     
            # Extract Crop Region
            crop, xOff, yOff = get_ROI_crop(current, ROIs, i) # e.g. xOff are X+width coordinate
            crop_height, crop_width = np.shape(crop)
            
            # Difference from current background
            diff = background_ROIs[i] - crop # Calculate the difference between a frame and the bkg
            #print(diff)

            # if debug:
                # plt.figure()
                # plt.subplot(1,3,2)
                # plt.imshow(crop)
                # plt.subplot(1,3,3)
                # plt.imshow(diff)
                # plt.subplot(1,3,1)
                # plt.imshow(background_ROIs[i])
                # plt.show()

            # Determine current threshold
            threshold_level = np.median(background_ROIs[i])/7  # Calculate the thershold number by diving by 7           
            #print(threshold_level)

            # Threshold            
            level, threshold = cv2.threshold(diff,threshold_level,255,cv2.THRESH_BINARY)
            #print("Level:", level)

            # Convert to uint8
            threshold = np.uint8(threshold)
            
            # Binary Close
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
            # Find Binary Contours            
            contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            # cv2.RETR_LIST = only the contour of the outside figure is considered see https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
            #CHAIN_APPROX_SIMPLE: It only showsend points of a line
            
            
            # Create Binary Mask Image
            mask = np.zeros(crop.shape,np.uint8)
        
            # If there are NO contours, then skip tracking
            if len(contours) == 0:
                if f!= 0:  #If it is NOT the first frame, then assigns fxS values of the previous frame (f-1)
                    area = -1.0
                    fX = fxS[f-1, i] - xOff 
                    fY = fyS[f-1, i] - yOff
                    bX = bxS[f-1, i] - xOff
                    bY = byS[f-1, i] - yOff
                    eX = exS[f-1, i] - xOff
                    eY = eyS[f-1, i] - yOff
                    heading = ortS[f-1, i]
                    motion = -1.0
                else:   #If it IS the first frame, then assigns values of the previous frame
                    area = -1.0
                    fX = xOff
                    fY = yOff
                    bX = xOff
                    bY = yOff
                    eX = xOff
                    eY = yOff
                    heading = -181.0
                    motion = -1.0
            
            else:   # If there are countors (particles)
                # Get Largest Contour (fish, ideally)
                largest_cnt, area = get_largest_contour(contours)
                #print(largest_cnt)

                # If the particle is too small to consider, skip frame
                if area == 0.0:  
                    if f!= 0: #If it is NOT the first frame 
                        fX = fxS[f-1, i] - xOff
                        fY = fyS[f-1, i] - yOff
                        bX = bxS[f-1, i] - xOff
                        bY = byS[f-1, i] - yOff
                        eX = exS[f-1, i] - xOff
                        eY = eyS[f-1, i] - yOff
                        heading = ortS[f-1, i]
                        motion = -1.0
                    else:  #If it IS the first frame 
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
                    # plt.figure()
                    # plt.imshow(mask, cmap='gray')
                    # plt.show() 
                    
                    # Get Area (again)
                    area = np.size(pixelpoints, 0)
                    
                    # ---------------------------------------------------------------------------------
                    # Compute Frame-by-Frame Motion (absolute changes above threshold)
                    # - Normalize by total absdiff from background
                    if (f != 0):
                        absdiff = np.abs(diff)
                        absdiff[absdiff < threshold_level] = 0
                        totalAbsDiff = np.sum(np.abs(absdiff))
                        # plt.figure()
                        # plt.imshow(absdiff, cmap='gray')
                        # plt.show() 
                        frame_by_frame_absdiff = np.abs(np.float32(previous_ROIs[i]) - np.float32(crop)) / 2 # Adjust for increases and decreases across frames
                        # plt.figure()
                        # plt.imshow(frame_by_frame_absdiff, cmap='gray')
                        # plt.show() 
                        frame_by_frame_absdiff[frame_by_frame_absdiff < threshold_level] = 0
                        # plt.figure()
                        # plt.imshow(frame_by_frame_absdiff, cmap='gray')
                        # plt.show() 
                        motion = np.sum(np.abs(frame_by_frame_absdiff))/totalAbsDiff


                    else:
                        motion = 0
                    
                    # Save Masked Fish Image from ROI (for subsequent frames motion calculation)
                    previous_ROIs[i] = np.copy(crop)
                    
                    # ---------------------------------------------------------------------------------
                    # Find Body and Eye Centroids
                    area = float(area)
                    
                    # Highlight 50% of the birghtest pixels (body + eyes)                    
                    numBodyPixels = int(np.ceil(area/2))
                    
                    # Highlight 10% of the birghtest pixels (mostly eyes)     
                    numEyePixels = int(np.ceil(area/10))
                    
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
                    fX = float(np.sum(c*values))/acc
                    fY = float(np.sum(r*values))/acc
                    
                    # Eye Centroid (a weighted centorid)
                    values = np.copy(all_values)                   
                    values = (values-eyeThreshold+1)
                    values[values < 0] = 0
                    acc = np.sum(values)
                    eX = float(np.sum(c*values))/acc
                    eY = float(np.sum(r*values))/acc
    
                    # Body Centroid (a binary centroid, excluding "eye" pixels)
                    values = np.copy(all_values)                   
                    values[values < bodyThreshold] = 0
                    values[values >= bodyThreshold] = 1                                                            
                    values[values > eyeThreshold] = 0                                                            
                    acc = np.sum(values)
                    bX = float(np.sum(c*values))/acc
                    bY = float(np.sum(r*values))/acc
                    
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
            # print(fxS)
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
            # plt.imshow(dilated_fish, cmap='gray')
            # plt.show()       
            updated_background = (np.float32(crop) * 0.01) + (current_background * 0.99)
            # plt.imshow(updated_background, cmap='gray')
            # plt.show() 
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
                plt.plot(fxS[f, i],fyS[f, i],'b.', markersize = 1)
                plt.plot(exS[f, i],eyS[f, i],'r.', markersize = 3)
                plt.plot(bxS[f, i],byS[f, i],'co', markersize = 3)
                plt.text(bxS[f, i]+10,byS[f, i]+10,  '{0:.1f}'.format(ortS[f, i]), color = [1.0, 1.0, 0.0, 0.5])
                plt.text(bxS[f, i]+10,byS[f, i]+30,  '{0:.0f}'.format(areaS[f, i]), color = [1.0, 0.5, 0.0, 0.5])
            plt.draw()
            plt.pause(0.001)
            
        # ---------------------------------------------------------------------------------
        # Save Tracking Summary
        if(f == 100):
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
    
    # Close Video File
    vid.release()
    
    # Return tracking data
    return fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS
#------------------------------------------------------------------------------





# 3. compute_intial_backgrounds: Compute the initial background for each ROI
def compute_intial_backgrounds(folder, ROIs):  #Folder is NS_Fodler, or S_Folder or Social_Cue folder

    # Load Video
    aviFiles = glob.glob(folder+'/*.avi') #get all the .avi files and makle a list of their path names
    aviFile = aviFiles[0] #Get the first folder path. We ssume there is only one .avi per fodler
    vid = cv2.VideoCapture(aviFile) # The cv2 fuction gets the videocapture object. 
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) # Get width of movie 
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get wheught of movie 
    
    # Allocate space for all ROI backgrounds
    background_ROIs = [] 
    for i in range(0,6):
        w, h = get_ROI_size(ROIs, i) #this function gets you a separate list for heigth and weight of eah single fish [0, x1][1,x1][2,x2]..
        background_ROIs.append(np.zeros((h, w), dtype = np.float32)) # This creates 1 array with both height and width 
    
    # Find initial background for each ROI
    for i in range(0,6):

        # Allocate space for background ROI stack
        crop_width, crop_height = get_ROI_size(ROIs, i)
        stepFrames = 1000 # Check background frame every 10 seconds
        bFrames = 20  # Calculate background for the first 5 min (10sec * 20 = 200sec = 3.3 minutes)
        backgroundStack = np.zeros((crop_height, crop_width, bFrames), dtype = np.float32)
        background = np.zeros((crop_height, crop_width), dtype = np.float32)
        previous = np.zeros((crop_height, crop_width), dtype = np.float32)
        
        # Store first frame ZERO
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0) #The .set() allows us to set the video on the frame we specifcy, which is  frame ZERO
        ret, im = vid.read()
        current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        crop, xOff, yOff = get_ROI_crop(current, ROIs, i) # Get image ROI cropped and the X and Y values
        print("backgroundStack shape:", backgroundStack.shape)
        print("crop shape:", crop.shape)
        print("xOff:", xOff)
        print("yOff:", yOff)
        print("crop_width:", crop_width)
        print("crop_height:", crop_height)
        print(width, height)
        backgroundStack[:,:,0] = np.copy(crop) #make a background ROI stack
        previous = np.copy(crop)
        bCount = 1
        
        # Search for useful background frames (significantly different than previous)
        changes = []
        for f in range(stepFrames, numFrames, stepFrames):

            # Read frame
            vid.set(cv2.CAP_PROP_POS_FRAMES, f) #The .set() allows us to set the video on the frame we specifcy "f", which is our current frame
            ret, im = vid.read() #ret = is a boolean value that is True if the frame exist, im = is the variable storing the image frame
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





# 4. get_ROI_size : Return ROI size from ROI list
def get_ROI_size(ROIs, numROi):
    width = int(ROIs[numROi, 2])
    height = int(ROIs[numROi, 3])
    
    return width, height


# 5. get_ROI_crop: Return cropped image from ROI list
def get_ROI_crop(image, ROIs, numROi):
    r1 = int(ROIs[numROi, 1])
    r2 = int(r1+ROIs[numROi, 3])
    c1 = int(ROIs[numROi, 0])
    c2 = int(c1+ROIs[numROi, 2])
    crop = image[r1:r2, c1:c2] # First it is heigth the width 
    return crop, c1, r1

###=============================

# 6. get_largest_contour: return largest (area) cotour from contour list
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