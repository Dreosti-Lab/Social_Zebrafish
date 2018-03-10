# -*- coding: utf-8 -*-
"""
Created on Sun May 11 14:01:46 2014

@author: kampff
"""

# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'
#-----------------------------------------------------------------------------
# Set "Base Path" for this analysis session
base_path = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Python_Analysis_Adam'

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats

# Import local modules
import SZ_utilities as SZU
import SZ_macros as SZM
import SZ_video as SZV
import SZ_analysis as SZA
import SZ_summary as SZS
import BONSAI_ARK
import glob
import pylab as pl


# -----------------------------------------------------------------------------
# Function to load all summary statistics and make a summary figure
# -----------------------------------------------------------------------------

# Set Analysis Folder Path where all the npz files you want to load are saved
analysisFolder = base_path + r'\Analysis_Folder\Isolated_Summary'
#analysisFolder = base_path + r'\Analysis_Folder\Controls_Summary'
#analysisFolder = base_path + r'\Analysis_Folder\Summary'

# Find all the npz files saved for each group and fish with all the information
npzFiles = glob.glob(analysisFolder+'\*.npz')

#CAlculate how many files
numFiles = np.size(npzFiles, 0)

SPI_NS_ALL = np.zeros(numFiles)
SPI_S_ALL = np.zeros(numFiles)

BPS_NS_ALL = np.zeros(numFiles)
BPS_S_ALL = np.zeros(numFiles)

IBI_NS_ALL = np.zeros(0)
IBI_S_ALL = np.zeros(0)

Pauses_NS_ALL = np.zeros(numFiles)
Pauses_S_ALL = np.zeros(numFiles)

OrtHist_NS_NSS_ALL = np.zeros((numFiles,36))
OrtHist_NS_SS_ALL = np.zeros((numFiles,36))
OrtHist_S_NSS_ALL = np.zeros((numFiles,36))
OrtHist_S_SS_ALL = np.zeros((numFiles,36))


#Go through al the files contained in the analysis folder
for f, filename in enumerate(npzFiles):

    #Load each npz file
    dataobject = np.load(filename)
    
    #Extract from the npz file
    SPI_NS = dataobject['SPI_NS']    
    SPI_S = dataobject['SPI_S']   
    BPS_NS = dataobject['BPS_NS']   
    BPS_S = dataobject['BPS_S']
    IBI_NS = dataobject['IBI_NS']   
    IBI_S = dataobject['IBI_S']
    Pauses_NS = dataobject['Pauses_NS']   
    Pauses_S = dataobject['Pauses_S']
    OrtHist_ns_NonSocialSide = dataobject['OrtHist_NS_NonSocialSide']
    OrtHist_ns_SocialSide = dataobject['OrtHist_NS_SocialSide']
    OrtHist_s_NonSocialSide = dataobject['OrtHist_S_NonSocialSide']
    OrtHist_s_SocialSide = dataobject['OrtHist_S_SocialSide']
    
    #Make an array with all summary stats
    SPI_NS_ALL[f] = SPI_NS
    SPI_S_ALL[f] = SPI_S
    BPS_NS_ALL[f] = BPS_NS
    BPS_S_ALL[f] = BPS_S
    IBI_NS_ALL = np.concatenate([IBI_NS_ALL, IBI_NS])
    IBI_S_ALL = np.concatenate([IBI_S_ALL, IBI_S])
    Pauses_NS_ALL[f] = Pauses_NS
    Pauses_S_ALL[f] = Pauses_S
    OrtHist_NS_NSS_ALL[f,:] = OrtHist_ns_NonSocialSide
    OrtHist_NS_SS_ALL[f,:] = OrtHist_ns_SocialSide
    OrtHist_S_NSS_ALL[f,:] = OrtHist_s_NonSocialSide
    OrtHist_S_SS_ALL[f,:] = OrtHist_s_SocialSide   

# ----------------
# SPI Summary Plot

#Make histogram and plot it with lines 
a_ns,c=np.histogram(SPI_NS_ALL,  bins=8, range=(-1,1))
a_s,c=np.histogram(SPI_S_ALL,  bins=8, range=(-1,1))
centers = (c[:-1]+c[1:])/2

#Normalize by tot number of fish
Tot_Fish_NS=numFiles

a_ns_float = np.float32(a_ns)
a_s_float = np.float32(a_s)

a_ns_nor_medium=a_ns_float/Tot_Fish_NS
a_s_nor_medium=a_s_float/Tot_Fish_NS 
 
plt.figure()
plt.plot(centers, a_ns_nor_medium, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
plt.plot(centers, a_s_nor_medium, color=[1.0,0.0,0.0,0.5], linewidth=4.0)
plt.title('Non Social/Social SPI', fontsize=12)
plt.xlabel('Preference Index (PI_)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)
plt.axis([-1.1, 1.1, 0, 0.5])
pl.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=12)

bar_width=0.25
plt.figure()
plt.bar(centers, a_ns_nor_medium, width=0.25, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
plt.title('Non Social SPI', fontsize=12)
plt.xlabel('Preference Index (PI_)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)
plt.axis([-1.1, 1.1, 0, 0.5])
pl.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=12)

plt.figure()
plt.bar(centers, a_s_nor_medium, width=0.25, color=[1.0,0.0,0.0,1.0], linewidth=4.0)
plt.title('Social SPI', fontsize=12)
plt.xlabel('Preference Index (PI_)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)
plt.axis([-1.1, 1.1, 0, 0.5])
pl.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=12)


# ----------------
# BPS Summary Plot

# Make histogram and plot it with lines 
a_ns,c=np.histogram(BPS_NS_ALL,  bins=16, range=(0,10))
a_s,c=np.histogram(BPS_S_ALL,  bins=16, range=(0,10))
centers = (c[:-1]+c[1:])/2

#Normalize by tot number of fish
Tot_Fish_NS=numFiles

a_ns_float = np.float32(a_ns)
a_s_float = np.float32(a_s)

a_ns_nor_medium=a_ns_float/Tot_Fish_NS
a_s_nor_medium=a_s_float/Tot_Fish_NS 
 
plt.figure()
plt.plot(centers, a_ns_nor_medium, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
plt.plot(centers, a_s_nor_medium, color=[1.0,0.0,0.0,0.5], linewidth=4.0)
plt.title('Non Social/Social BPS', fontsize=12)
plt.xlabel('Bouts per Second (BPS)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)
plt.axis([-0.1, 10.1, 0, 0.5])
pl.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
pl.xticks([0, 5, 10], fontsize=12)

bar_width=0.5
plt.figure()
plt.bar(centers, a_ns_nor_medium, width=bar_width, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
plt.title('Non Social BPS', fontsize=12)
plt.xlabel('Bouts per Second (BPS)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)
plt.axis([-0.1, 10.1, 0, 0.5])
pl.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
pl.xticks([0, 5, 10], fontsize=12)

plt.figure()
plt.bar(centers, a_s_nor_medium, width=bar_width, color=[1.0,0.0,0.0,1.0], linewidth=4.0)
plt.title('Social BPS', fontsize=12)
plt.xlabel('Bouts per Second (BPS)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)
plt.axis([-0.1, 10.1, 0, 0.5])
pl.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
pl.xticks([0, 5, 10], fontsize=12)

# ----------------
# IBI/Pause Summary Plot
mean_long_pauses_NS = np.mean(Pauses_NS_ALL) 
mean_long_pauses_S = np.mean(Pauses_S_ALL) 

plt.figure()

plt.subplot(221)
plt.plot(Pauses_NS_ALL)
plt.title('Long Pauses (NS):' + str(mean_long_pauses_NS), fontsize=12)
plt.xlabel('Fish', fontsize=12)
plt.ylabel('Pause Count', fontsize=12)
plt.subplot(222)
plt.plot(Pauses_S_ALL)
plt.title('Long Pauses (S):' + str(mean_long_pauses_S), fontsize=12)
plt.xlabel('Fish', fontsize=12)
plt.ylabel('Pause Count', fontsize=12)
plt.subplot(223)
plt.plot(SPI_NS_ALL, Pauses_NS_ALL, 'ko')
plt.title('SPI vs Long Pauses (NS)' )
plt.xlabel('SPI', fontsize=12)
plt.ylabel('Pause Count', fontsize=12)
plt.subplot(224)
plt.plot(SPI_S_ALL, Pauses_S_ALL, 'ko')
plt.title('SPI vs Long Pauses (S)')
plt.xlabel('SPI', fontsize=12)
plt.ylabel('Pause Count', fontsize=12)


# ----------------
# ORT_HIST Summary Plot

# Accumulate all histogram values and normalize
Accum_OrtHist_NS_NSS_ALL = np.sum(OrtHist_NS_NSS_ALL, axis=0)
Accum_OrtHist_NS_SS_ALL = np.sum(OrtHist_NS_SS_ALL, axis=0)
Accum_OrtHist_S_NSS_ALL = np.sum(OrtHist_S_NSS_ALL, axis=0)
Accum_OrtHist_S_SS_ALL= np.sum(OrtHist_S_SS_ALL, axis=0)

Norm_OrtHist_NS_NSS_ALL = Accum_OrtHist_NS_NSS_ALL/np.sum(Accum_OrtHist_NS_NSS_ALL)
Norm_OrtHist_NS_SS_ALL = Accum_OrtHist_NS_SS_ALL/np.sum(Accum_OrtHist_NS_SS_ALL)
Norm_OrtHist_S_NSS_ALL = Accum_OrtHist_S_NSS_ALL/np.sum(Accum_OrtHist_S_NSS_ALL)
Norm_OrtHist_S_SS_ALL = Accum_OrtHist_S_SS_ALL/np.sum(Accum_OrtHist_S_SS_ALL)

# Plot Summary
xAxis = np.arange(-np.pi,np.pi+np.pi/18.0, np.pi/18.0)
#plt.figure('Summary: Orientation Histograms')
plt.figure()

ax = plt.subplot(221, polar=True)
plt.title('NS - Non Social Side')
plt.plot(xAxis, np.hstack((Norm_OrtHist_NS_NSS_ALL, Norm_OrtHist_NS_NSS_ALL[0])), linewidth = 3)

ax = plt.subplot(222, polar=True)
plt.title('NS - Social Side')
plt.plot(xAxis, np.hstack((Norm_OrtHist_NS_SS_ALL, Norm_OrtHist_NS_SS_ALL[0])), linewidth = 3)

ax = plt.subplot(223, polar=True)
plt.title('S - Non Social Side')
plt.plot(xAxis, np.hstack((Norm_OrtHist_S_NSS_ALL, Norm_OrtHist_S_NSS_ALL[0])), linewidth = 3)

ax = plt.subplot(224, polar=True)
plt.title('S - Social Side')
plt.plot(xAxis, np.hstack((Norm_OrtHist_S_SS_ALL, Norm_OrtHist_S_SS_ALL[0])), linewidth = 3)


# FIN
