# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:19:59 2019

@author: gonca
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from skimage import img_as_ubyte
from skimage.morphology import skeletonize

fps = 100.
frame_interval = 1 # skip X number of frames before updating window
frame_offsetx = 100 # optional offset for cropping leftside of image
tailbase_kernel_size = 11 # size of morphological operator for removing noise
tail_kernel_size = 11 # size of morphological operator for removing noise
thresh_stim = 150 # laserstim pixel thresh (brighter image => higher threshold)
thresh_base = 33 # bladder pixel thresh (darker image => lower threshold)
thresh_tail = 140 # tail pixel thresh (darker image => lower threshold)
pixelthresh_stim = 20000 # threshold to decide if laser is ON/OFF (total pxs)
pixelthresh_max = 10 # scale factor for plottign stimulus
degree_fit = 13 # degree of polynomial for fitting curve
draw_segment = False # plot raw segmentations

import warnings
warnings.simplefilter('ignore', np.RankWarning)

def morphopen(image,size):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
  return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


#fname = 'Z:/D R E O S T I   L A B/Elisa/experiments/Behavioural_Setup/Experiments/Pilots/2019_07_25/fish_3/round1_2019-07-25T15_26_14.avi'
if not 'fname' in locals():
    fname = input('filename? ').strip("'")

def whitepixels(frame):
    ### PREPROCESS - FIND STIMULUS STATE
    _,segment = cv2.threshold(frame, thresh_stim, 1, cv2.THRESH_BINARY)
    return np.sum(segment)

def tailbase(frame):
    ### PREPROCESS - FIND BASE OF TAIL
    _,segment = cv2.threshold(frame, thresh_base, 255, cv2.THRESH_BINARY_INV)
    segment = morphopen(segment,tailbase_kernel_size)
    bladderys,bladderxs = np.nonzero(segment) # coordinates of bladder pixels
    baseindex = np.argmin(bladderxs) # tail base is leftmosyt bladder pixel
    return bladderys[baseindex],bladderxs[baseindex]

def tailsegment(frame,tailbase,canvas=None):
    ### PREPROCESS - SEGMENT TAIL
    # frame format (rows, columns)
    xtailbase = tailbase[1]
    roi = frame[:,frame_offsetx:xtailbase] # crop a region of interest
    if canvas is not None:
        cv2.line(canvas,(xtailbase,0),(xtailbase,canvas.shape[0]),(255,0,0),2)
        roicanvas = canvas[:,frame_offsetx:xtailbase]
      
    # threshold image
    _,segment = cv2.threshold(roi, thresh_tail, 255, cv2.THRESH_BINARY_INV)
    if canvas is not None and draw_segment:
        cv2.add(cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR),roicanvas,roicanvas)
    
    # find tail pixels
    segment = morphopen(segment,tail_kernel_size) # remove small pixel noise
    tailys,tailxs = np.nonzero(segment) # get coordinates of all tail pixels
      
    ## POLYNOMIAL FIT
    fit = np.polyfit(tailxs, tailys, degree_fit) # fit a 3-degree polynomial to the coords
    p = np.poly1d(fit) # create polynomial object for interpolation
    minx,maxx = (np.min(tailxs), np.max(tailxs)) # get extremes of X-coordinates
    curvexs = np.linspace(minx, maxx, 10) # create equi-distant points
    curveys = p(curvexs) # interpolate coordinates on the curve for those points
    
    ## PLOT FITTED CURVE
    if canvas is not None:
        pts = np.vstack((curvexs+frame_offsetx,curveys)).astype(np.int32).T
        cv2.polylines(canvas,[pts],
                      isClosed=False,color=(0,0,255),thickness=2)
        cv2.imshow('feedback', canvas)
    return np.mean(np.diff(np.diff(curveys)))

def askpermission(message):
    proceed = input(message)
    if proceed == 'n':
        raise Exception("Process aborted!")
    return proceed == 'y'

def waitforfigure():
    while True:
        if plt.waitforbuttonpress():
            break
    plt.close('all')

## USER INPUT - ASK FOR PERMISSION TO PROCESS VIDEO
if askpermission(str.format("Processing file '{0}'... Proceed [y/n/s]? ",fname)):

    # create video capture object and make sure we release it in the end
    #with cv2.VideoCapture() as capture:
    capture = cv2.VideoCapture()
    capture.open(fname) # open video file
    
    # create interactive figure
    cv2.namedWindow('feedback', cv2.WINDOW_GUI_NORMAL) # turn on plots
    
    #capture.set(cv2.CAP_PROP_POS_FRAMES, 40) # go to specific frame
    #result,frame = capture.retrieve() # read a frame
    tailstart = None
    curvature = []
    stimulus = []
    while True:
        print(len(curvature)) # print number of processed frames (length of list)
        _,canvas = capture.read()
        if canvas is None:
            break
        
        frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) # convert to grayscale
        if tailstart is None:
            tailstart = tailbase(frame)
        
        stimulus.append(whitepixels(frame))
        try:
            # if we do not need to draw the current frame, reset the canvas
            if len(curvature) % frame_interval != 0:
                canvas = None
            curvature.append(tailsegment(frame,tailstart,canvas))
            key = cv2.waitKey(frame_interval)
            if key == 27: # Interrupt when pressing ESC
                break
            elif key == 113: # Increase interval when pressing Q
                frame_interval += 10
        except:
            curvature.append(np.nan)
    
    # check stimulus intensity
    cv2.destroyAllWindows()
    capture.release()

plt.plot(stimulus)
plt.show()
waitforfigure()

## USER INPUT - ASK FOR PERMISSION TO CONTINUE ANALYSIS
text = input(str.format("Check stimulus threshold {0}... ? ",pixelthresh_stim))
if len(text) > 0:
    pixelthresh_stim = float(text)

# draw curvature vs stimulus in time
time = np.arange(len(curvature)) / fps
stimval = np.int8(np.array(stimulus) > pixelthresh_stim) * pixelthresh_max
plt.plot(time,curvature)
plt.plot(time,stimval)
plt.xlabel('time (s)')
waitforfigure()

# find stimonset frames
stimon = np.flatnonzero(np.diff(stimval)>0)
trials_curvature = np.array([np.abs(curvature[stim-100:stim+250]) for stim in stimon]).T
trials_stim = np.array([np.abs(stimval[stim-100:stim+250]) for stim in stimon]).T
plt.plot(trials_curvature,'k',alpha=0.1)
plt.plot(trials_stim,'orange',alpha=0.8)
plt.plot(np.mean(trials_curvature,axis=1),'b')
waitforfigure()

## USER INPUT - ASK FOR PERMISSION TO SAVE RESULTS
if askpermission(str.format("Saving results... Proceed [y/n/s]? ",fname)):
    
    # save data to file
    data = np.array([time,stimulus,curvature]).T
    datapath = os.path.splitext(fname)[0]+'.csv'
    datapath,namefile = os.path.split(datapath)
    datapath,namesubject = os.path.split(datapath)
    datapath,namesession = os.path.split(datapath)
    datapath,nameexperiment = os.path.split(datapath)
    datapath,_ = os.path.split(datapath)
    datafname = os.path.join(datapath,
                             'Analysis',
                             nameexperiment,
                             namesession,
                             namesubject,
                             namefile)
    directory = os.path.split(datafname)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(datafname,data,header='time,stimulus,curvature',delimiter=',')
