# -*- coding: utf-8 -*-
"""
Created on Sun May 11 14:01:46 2014

@author: kampff
"""

# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'

# Set Library Paths
import sys
sys.path.append(lib_path)

# -----------------------------------------------------------------------------
# Set Base Path Aas the shared Dropbox Folder (unique to each computer)
base_path = dropbox_path 
# Use this in case the dropbox doesn't work (base_path=r'C:\Users\elenadreosti
#\Dropbox (Personal)'
#-----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#Function to load all the SPI_NS and SPI_S and to make summary figure
# -----------------------------------------------------------------------------


#Import libraries
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats
import glob
import pylab as pl


NS_SPI_All=[]
S_SPI_All=[]

# Set Analysis Folder Path where all the npz files you want to load are saved
analysisFolder = (base_path + r'\Python_ED\Social _Behaviour_Setup\Analysis_Folder\Shank3aE2_3bE5\Homozygous')
#analysisFolder = (base_path + r'\Python_ED\Social _Behaviour_Setup\Analysis_Folder\Shank3aE2_3bE5\Wt')







# TO ANALYZE ONLY ONE EEXaFOLDER
#analysisFolder   = analysisFolder + r'\Example Plots'

# Find all the npz files saved for each group and fish with all the information
npzFiles = glob.glob(analysisFolder+'\*.npz')

#CAlculate how many files
numFiles = np.size(npzFiles, 0)

#Go through al the files contained in the analysis folder
for filename in npzFiles:

    #Load each npz file
    dataobject = np.load(filename)
    
    #Extract from teh nz file the SPI_NS and the SPI_S
    NS_SPI = dataobject['SPI_NS']    
    S_SPI = dataobject['SPI_S']
    
    #MAke an array with all the SPI NS and S
    NS_SPI_All=np.append(NS_SPI_All,NS_SPI)
    S_SPI_All=np.append(S_SPI_All,S_SPI)


#Transform the list into a matlab-like array
AllPref_NS = np.array(NS_SPI_All)
AllPref_S = np.array(S_SPI_All)


#Make histogram and plot it with lines 

a_ns,c=np.histogram(AllPref_NS,  bins=8, range=(-1,1))
a_s,c=np.histogram(AllPref_S,  bins=8, range=(-1,1))
centers = (c[:-1]+c[1:])/2

#Normalize by tot number of fish
Tot_Fish_NS=len(AllPref_NS)

a_ns_float = np.float32(a_ns)
a_s_float = np.float32(a_s)

a_ns_nor_medium=a_ns_float/Tot_Fish_NS
a_s_nor_medium=a_s_float/Tot_Fish_NS 
 
plt.figure()
plt.plot(centers, a_ns_nor_medium, color=[0.5,0.5,0.5,0.6], linewidth=4.0)
plt.plot(centers, a_s_nor_medium, color=[1.0,0.0,0.0,1.0], linewidth=4.0)
plt.title('Non Social/Social PI', fontsize=20)
plt.xlabel('Preference Index (PI_)', fontsize=20)
plt.ylabel('Rel. Frequency', fontsize=20)
plt.axis([-1.1, 1.1, 0, 1.0])
pl.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=20)
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=20)

bar_width=0.25
plt.figure()
plt.bar(centers-bar_width/2, a_ns_nor_medium, width=0.25, color=[0.5,0.5,0.5,0.6], linewidth=4.0)
#plt.title('Non Social/Social PI', fontsize=20)
#plt.xlabel('Preference Index (PI_)', fontsize=20)
#plt.ylabel('Rel. Frequency', fontsize=20)
plt.axis([-1.1, 1.1, 0, 1.0])
pl.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=20)
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=20)

plt.figure()
plt.bar(centers-bar_width/2, a_s_nor_medium, width=0.25, color=[1.0,0.0,0.0,1.0], linewidth=4.0)
#plt.title('Non Social/Social PI', fontsize=20)
#plt.xlabel('Preference Index (PI_)', fontsize=20)
#plt.ylabel('Rel. Frequency', fontsize=20)
plt.axis([-1.1, 1.1, 0, 1.0])
pl.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=20)
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=20)




# FIN
