# -*- coding: utf-8 -*-
"""
Bonsai Python Utilities

@author: Elena Dreosti
"""

import numpy as np
import xml.etree.ElementTree as ET

# Parse Bonsai "Crop Regions" as ROIs
def read_bonsai_crop_rois(filename):

    # Load Bonsai Workflow File (*.bonsai)
    tree = ET.parse(filename) # This function reads and returns the XML element object. It loads into memory
    root = tree.getroot() # This function gets the root(memory location of the object) so it can be manipulated

    # Extract all "Region of Interest Tags" and save parameters (X,Y,Width,Height)
    X = []
    Y = []
    Width = []
    Height = []
    for ROI in root[0].iter('{clr-namespace:Bonsai.Vision;assembly=Bonsai.Vision}RegionOfInterest'):
        for param in ROI.iter('{clr-namespace:Bonsai.Vision;assembly=Bonsai.Vision}X'):
            X.append(int(param.text))
        for param in ROI.iter('{clr-namespace:Bonsai.Vision;assembly=Bonsai.Vision}Y'):
            Y.append(int(param.text))
        for param in ROI.iter('{clr-namespace:Bonsai.Vision;assembly=Bonsai.Vision}Width'):
            Width.append(int(param.text))
        for param in ROI.iter('{clr-namespace:Bonsai.Vision;assembly=Bonsai.Vision}Height'):
            Height.append(int(param.text))
    
    ROIs = np.zeros((len(X), 4))    
    ROIs[:, 0] = np.array(X)
    ROIs[:, 1] = np.array(Y)
    ROIs[:, 2] = np.array(Width)
    ROIs[:, 3] = np.array(Height)
    return ROIs


## Test Script
#filename = r'C:/Users/kampff/Desktop/TrackFile - 6 Fish and Social Tracker_dynamic_Social.bonsai'
#ROIs = read_bonsai_crop_rois(filename)
#print np.shape(ROIs)

# FIN