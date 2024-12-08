# -*- coding: utf-8 -*-
"""
Analyze all tracked fish in a social preference experiment
Created on Nov 18 2024
@author: dreostilab (Elena Dreosti)
"""

# Load environment file and variables
import os
from dotenv import load_dotenv
load_dotenv()
libs_path = os.getenv('LIBS_PATH') + "/../Behaviour/ED/libs"
base_path = os.getenv('BASE_PATH')


# Set Library Paths
import sys
sys.path.append(libs_path)

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats
import seaborn as sns
import pandas as pd
import seaborn as sns

# Import local modules
import SZ_utilities_ED as SZU
import SZ_macros_ED as SZM
import SZ_video_ED as SZV
import SZ_analysis_ED as SZA
import SZ_summary_ED as SZS
import BONSAI_ARK_ED
import glob
import pylab as pl
import seaborn as sns
import pandas as pd

# Specify Analysis folder 

# analysisFolder = base_path + r'/Akap11/Analysis_TPI' 

# analysisFolder = base_path + r'/cacnag1g/Analysis_TPI'

# analysisFolder = base_path + r'/gria3/Analysis_TPI'

# analysisFolder = base_path + r'/grin2a/Analysis_TPI'

#analysisFolder = base_path + r'/hcn4/Analysis_TPI'

#analysisFolder = base_path + r'/herc1/Analysis_TPI'

#analysisFolder = base_path + r'/nr3c2/Analysis_TPI'

analysisFolder = base_path + r'/Scrambled/Analysis_TPI'

#analysisFolder = base_path + r'/Sp4/Analysis_TPI'

#analysisFolder = base_path + r'/trio/Analysis_TPI'

#analysisFolder = base_path + r'/xpo7/Analysis_TPI'





# Set freeze time threshold
freeze_threshold = 600 #frames 1 minute
long_freeze_threshold = 2400 # 4 minutes

# Find all the npz files saved for each group and fish with all the information
npzFiles = glob.glob(analysisFolder+'/*.npz')

# Calculate how many files
numFiles = np.size(npzFiles, 0)

# Allocate space for summary data
VPI_NS_ALL = np.zeros(numFiles)
VPI_S_ALL = np.zeros(numFiles)
VPI_NS_BINS_ALL = np.zeros((numFiles, 15))
VPI_S_BINS_ALL = np.zeros((numFiles, 15))
SPI_NS_ALL = np.zeros(numFiles)
SPI_S_ALL = np.zeros(numFiles)
BPS_NS_ALL = np.zeros(numFiles)
BPS_S_ALL = np.zeros(numFiles)
Distance_NS_ALL = np.zeros(numFiles)
Distance_S_ALL = np.zeros(numFiles)
Freezes_NS_ALL = np.zeros(numFiles)
Freezes_S_ALL = np.zeros(numFiles)
Long_Freezes_NS_ALL = np.zeros(numFiles)
Long_Freezes_S_ALL = np.zeros(numFiles)
Percent_Moving_NS_ALL = np.zeros(numFiles)
Percent_Moving_S_ALL = np.zeros(numFiles)
OrtHist_NS_NSS_ALL = np.zeros((numFiles,36))
OrtHist_NS_SS_ALL = np.zeros((numFiles,36))
OrtHist_S_NSS_ALL = np.zeros((numFiles,36))
OrtHist_S_SS_ALL = np.zeros((numFiles,36))
Bouts_NS_ALL = np.zeros((0,10))
Bouts_S_ALL = np.zeros((0,10))
Pauses_NS_ALL = np.zeros((0,10))   
Pauses_S_ALL = np.zeros((0,10))
SPI_S_BINS_ALL = np.zeros((numFiles, 15))
SPI_NS_BINS_ALL = np.zeros((numFiles, 15))

# # Create report file and write on it "r"
reportFilename = analysisFolder + r'/report.txt'
reportFile = open(reportFilename, 'w')

#Go through all the files contained in the analysis folder
for f, filename in enumerate(npzFiles):

    # Load each npz file
    dataobject = np.load(filename) # dataobject is a name used to store the result of np.load(filename)
    print(filename)
    #print(dataobject.keys())
    
    # Extract from the npz file
    VPI_NS = dataobject['VPI_NS']    
    VPI_S = dataobject['VPI_S']
    VPI_NS_BINS = dataobject['VPI_NS_BINS']    
    VPI_S_BINS = dataobject['VPI_S_BINS']
    SPI_NS = dataobject['SPI_NS']
    SPI_S = dataobject['SPI_S']   
    BPS_NS = dataobject['BPS_NS']   
    BPS_S = dataobject['BPS_S']
    Distance_NS = dataobject['Distance_NS']   
    Distance_S = dataobject['Distance_S']   
    OrtHist_ns_NonSocialSide = dataobject['OrtHist_NS_NonSocialSide']
    OrtHist_ns_SocialSide = dataobject['OrtHist_NS_SocialSide']
    OrtHist_s_NonSocialSide = dataobject['OrtHist_S_NonSocialSide']
    OrtHist_s_SocialSide = dataobject['OrtHist_S_SocialSide']
    Bouts_NS = dataobject['Bouts_NS']   
    Bouts_S = dataobject['Bouts_S']
    Pauses_NS = dataobject['Pauses_NS']   
    Pauses_S = dataobject['Pauses_S']
    Percent_Moving_NS = dataobject['Percent_Moving_NS']   
    Percent_Moving_S = dataobject['Percent_Moving_S']
    SPI_NS_BINS = dataobject['SPI_NS_BINS']
    SPI_S_BINS = dataobject['SPI_S_BINS']
    

    # Count Freezes
    Freezes_NS = np.array(np.sum(Pauses_NS[:,8] > freeze_threshold)) 
    # The 8th element of pauses is the duration of each pause
    Freezes_S = np.array(np.sum(Pauses_S[:,8] > freeze_threshold))
    Freezes_NS_ALL[f] = Freezes_NS
    Freezes_S_ALL[f] = Freezes_S
    
    # Count Long Freezes
    Long_Freezes_NS = np.array(np.sum(Pauses_NS[:,8] > long_freeze_threshold))
    Long_Freezes_S = np.array(np.sum(Pauses_S[:,8] > long_freeze_threshold))
    Long_Freezes_NS_ALL[f] = Long_Freezes_NS
    Long_Freezes_S_ALL[f] = Long_Freezes_S

    # Make arrays with all summary stats
    VPI_NS_ALL[f] = VPI_NS
    VPI_S_ALL[f] = VPI_S
    VPI_NS_BINS_ALL[f,:] = VPI_NS_BINS
    VPI_S_BINS_ALL[f,:] = VPI_S_BINS
    SPI_NS_BINS_ALL[f,:] = SPI_NS_BINS
    SPI_S_BINS_ALL[f,:] = SPI_S_BINS
    SPI_NS_ALL[f] = SPI_NS
    SPI_S_ALL[f] = SPI_S
    BPS_NS_ALL[f] = BPS_NS
    BPS_S_ALL[f] = BPS_S
    Distance_NS_ALL[f] = Distance_NS
    Distance_S_ALL[f] = Distance_S
    OrtHist_NS_NSS_ALL[f,:] = OrtHist_ns_NonSocialSide
    OrtHist_NS_SS_ALL[f,:] = OrtHist_ns_SocialSide
    OrtHist_S_NSS_ALL[f,:] = OrtHist_s_NonSocialSide
    OrtHist_S_SS_ALL[f,:] = OrtHist_s_SocialSide

    # Concat all Pauses/Bouts
    Bouts_NS_ALL = np.vstack([Bouts_NS_ALL, Bouts_NS])
    Bouts_S_ALL = np.vstack([Bouts_S_ALL, Bouts_S])
    Pauses_NS_ALL = np.vstack([Pauses_NS_ALL, Pauses_NS])
    Pauses_S_ALL = np.vstack([Pauses_S_ALL, Pauses_S])

    # Save to report (text file)
    reportFile.write(filename + '\n') # /n go to next line
    reportFile.write('-------------------\n')
    reportFile.write('VPI_NS:\t' + format(np.float64(VPI_NS), '.3f') + '\n')
    reportFile.write('VPI_S:\t' + format(np.float64(VPI_S), '.3f') + '\n')
    reportFile.write('SPI_NS:\t' + format(np.float64(SPI_NS), '.3f') + '\n')
    reportFile.write('SPI_S:\t' + format(np.float64(SPI_S), '.3f') + '\n')
    reportFile.write('BPS_NS:\t' + format(np.float64(BPS_NS), '.3f') + '\n')
    reportFile.write('BPS_S:\t' + format(np.float64(BPS_S), '.3f') + '\n')
    reportFile.write('Distance_NS:\t' + format(np.float64(Distance_NS), '.3f') + '\n')
    reportFile.write('Distance_S:\t' + format(np.float64(Distance_S), '.3f') + '\n')
    reportFile.write('Freezes_NS:\t' + format(np.float64(Freezes_NS), '.3f') + '\n')
    reportFile.write('Freezes_S:\t' + format(np.float64(Freezes_S), '.3f') + '\n')
    reportFile.write('Long_Freezes_NS:\t' + format(np.float64(Long_Freezes_NS), '.3f') + '\n')
    reportFile.write('Long_Freezes_S:\t' + format(np.float64(Long_Freezes_S), '.3f') + '\n')
    reportFile.write('Perc_Moving_NS:\t' + format(np.float64(Percent_Moving_NS), '.3f') + '\n')
    reportFile.write('Perc_Moving_S:\t' + format(np.float64(Percent_Moving_S), '.3f') + '\n')
    reportFile.write('-------------------\n\n')

# Close report
reportFile.close()


# # ==================================================================
# FIGURE 1 ===      VPI Summary Plot 

#Make histogram to get the frequencies of VPI for 8 bins 
a_ns,c=np.histogram(VPI_NS_ALL,  bins=8, range=(-1,1))
a_s,c=np.histogram(VPI_S_ALL,  bins=8, range=(-1,1))
centers = (c[:-1]+c[1:])/2

#Normalize by tot number of fish
Tot_Fish_NS=numFiles

a_ns_float = np.float32(a_ns)
a_s_float = np.float32(a_s)

a_ns_nor_medium=a_ns_float/Tot_Fish_NS
a_s_nor_medium=a_s_float/Tot_Fish_NS 
 
#plt.figure()
fig, ax = plt.subplots(1,3,figsize=(10, 4))

# Plot data in FIRST subplot
sns.lineplot(x=centers, y=a_ns_nor_medium, color=[0.5, 0.5, 0.5, 1.0], linewidth=4.0, ax=ax[0], label='Non Social')
sns.lineplot(x=centers, y=a_s_nor_medium, color=[1.0, 0.0, 0.0, 0.5], linewidth=4.0, ax=ax[0], label='Social')
# By changing this  ax=ax[0] to  ax=ax[1] or  ax=ax[2], you indicate on what subplot to plot it. 

# Customize subplot
ax[0].set_title('Non Social/Social VPI', fontsize=12)
ax[0].set_xlabel('Preference Index (PI_)', fontsize=12)
ax[0].set_ylabel('Rel. Frequency', fontsize=12)
ax[0].set_xlim([-1.1, 1.1])
ax[0].set_ylim([0, 0.5])
ax[0].set_xticks([-1, -0.5, 0, 0.5, 1.0])
ax[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax[0].legend()

# Plot data in SECOND subplot
# You need to make sure that centres is an array to plot the xticks evently. Otehrwise plotting on one tick. 
centers = np.array([-0.9, -0.6, -0.4, -0.1, 0.1, 0.4, 0.6, 0.9])

sns.barplot(x=centers, y=a_ns_nor_medium, color=[0.5, 0.5, 0.5, 1.0],linewidth=4.0, ax=ax[1], label='Non Social')

# Customize subplot
ax[1].set_title('Non Social VPI', fontsize=12)
ax[1].set_xlabel('Preference Index', fontsize=12)
ax[1].set_ylabel('Rel. Frequency', fontsize=12)
ax[1].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax[1].legend()


# Plot data in THIRD subplot

sns.barplot(x=centers, y=a_s_nor_medium, color=[1.0, 0.0, 0.0], alpha=0.9,linewidth=4.0, ax=ax[2], label='Social')

# Customize the second subplot
ax[2].set_title('Social VPi', fontsize=12)
ax[2].set_xlabel('Preference Index', fontsize=12)
ax[2].set_ylabel('Rel. Frequency', fontsize=12)
ax[2].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

filename = analysisFolder + '/Fig1_VPI.png'
plt.savefig(filename, dpi=600)
#plt.show()
plt.close('all')


# # ==================================================================
# FIGURE 2 ===  VPI "Binned" in 1 minute BINS 
 
plt.figure()
fig, ax = plt.subplots(1,2,figsize=(8,6))
fig.suptitle("Temporal VPI (one minute bins)", fontsize=16)

plt.subplot(1,2,1)

# Compute mean, standard deviation, and standard error of NS
mean = np.nanmean(VPI_NS_BINS_ALL, 0)
std = np.nanstd(VPI_NS_BINS_ALL, 0)
valid = (np.logical_not(np.isnan(VPI_NS_BINS_ALL)))
n = np.sum(valid, 0)
se = std/np.sqrt(n) # Standard error
upper_bound = mean + se
lower_bound = mean - se

# Prepare x label values for plotting
x_values = np.arange(VPI_NS_BINS_ALL.shape[1])  # 0 to 14 (15 bins)

# Plot all trajectories in one go
plt.plot(x_values, VPI_NS_BINS_ALL.T, color=[0, 0, 0, 0.1], linewidth=1)

# Plot the mean line
sns.lineplot(x=x_values, y=mean, color="black", linewidth=2,  ax=ax[0],label="Mean")
# Add dots for each data point
plt.scatter(x_values, mean, color="black", s=40, zorder=3)

# Plot the standard deviation bounds
plt.plot(x_values, upper_bound, color="red", linewidth=1, linestyle="-", label="Mean + Std Dev")
plt.plot(x_values, lower_bound, color="red", linewidth=1, linestyle="-", label="Mean - Std Dev")

# Set custom y-ticks (optional based on data range)
plt.yticks(ticks=[-1, -0.5, 0, 0.5, 1], labels=["-1", "-0.5", "0", "0.5", "1"])
plt.xlabel('minutes')
plt.ylabel('VPI')

# Customize the  subplot
ax[0].set_title('Non_Social', fontsize=12)
ax[0].set_xlabel('minutes', fontsize=12)
ax[0].set_ylabel('VPI', fontsize=12)
ax[0].legend()



# =======  Social 
plt.subplot(1,2,2)

# Compute mean, standard deviation, and standard error of NS
mean2 = np.nanmean(VPI_S_BINS_ALL, 0)
std2 = np.nanstd(VPI_S_BINS_ALL, 0)
valid2 = (np.logical_not(np.isnan(VPI_S_BINS_ALL)))
n2 = np.sum(valid2, 0)
se2 = std/np.sqrt(n2) # Standard error
upper_bound2 = mean2 + se2
lower_bound2 = mean2 - se2

# Prepare x label values for plotting
x_values2 = np.arange(VPI_NS_BINS_ALL.shape[1])  # 0 to 14 (15 bins)

# Plot all trajectories in one go
plt.plot(x_values2, VPI_NS_BINS_ALL.T, color=[0, 0, 0, 0.1], linewidth=1)

# Plot the mean line
sns.lineplot(x=x_values2, y=mean2, color="black",  ax=ax[1], linewidth=2, label="Mean")
# Add dots for each data point
plt.scatter(x_values2, mean2, color="black", s=40, zorder=3)

# Plot the standard deviation bounds
plt.plot(x_values2, upper_bound2, color="red", linewidth=1, linestyle="-", label="Mean + Std Dev")
plt.plot(x_values2, lower_bound2, color="red", linewidth=1, linestyle="-", label="Mean - Std Dev")

# Customize the  subplot
ax[1].set_title('Social', fontsize=12)
ax[1].set_xlabel('minutes', fontsize=12)
ax[1].set_ylabel('VPI', fontsize=12)
ax[1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
#plt.show()

filename = analysisFolder + '/Fig2_VPI_1min_BINS.png'
plt.savefig(filename, dpi=600)
plt.close('all')



# # ==================================================================
# FIGURE 3 ===  SPI Summary Plot

#Make histogram to get the frequencies of VPI for 8 bins 
a_ns,c=np.histogram(SPI_NS_ALL,  bins=8, range=(-1,1))
a_s,c=np.histogram(SPI_S_ALL,  bins=8, range=(-1,1))
centers = (c[:-1]+c[1:])/2

#Normalize by tot number of fish
Tot_Fish_NS=numFiles

a_ns_float = np.float32(a_ns)
a_s_float = np.float32(a_s)

a_ns_nor_medium=a_ns_float/Tot_Fish_NS
a_s_nor_medium=a_s_float/Tot_Fish_NS 
 
#plt.figure()
fig, ax = plt.subplots(1,3,figsize=(10, 4))

# Plot data in FIRST subplot
sns.lineplot(x=centers, y=a_ns_nor_medium, color=[0.5, 0.5, 0.5, 1.0], linewidth=4.0, ax=ax[0], label='Non Social')
sns.lineplot(x=centers, y=a_s_nor_medium, color=[1.0, 0.0, 0.0, 0.5], linewidth=4.0, ax=ax[0], label='Social')
# By changing this  ax=ax[0] to  ax=ax[1] or  ax=ax[2], you indicate on what subplot to plot it. 

# Customize subplot
ax[0].set_title('Non Social/Social SPI', fontsize=12)
ax[0].set_xlabel('Preference Index', fontsize=12)
ax[0].set_ylabel('Rel. Frequency', fontsize=12)
ax[0].set_xlim([-1.1, 1.1])
ax[0].set_ylim([0, 0.5])
ax[0].set_xticks([-1, -0.5, 0, 0.5, 1.0])
ax[0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax[0].legend()

# Plot data in SECOND subplot
# You need to make sure that centres is an array to plot the xticks evently. Otehrwise plotting on one tick. 
centers = np.array([-0.9, -0.6, -0.4, -0.1, 0.1, 0.4, 0.6, 0.9])

sns.barplot(x=centers, y=a_ns_nor_medium, color=[0.5, 0.5, 0.5, 1.0],linewidth=4.0, ax=ax[1], label='Non Social')

# Customize subplot
ax[1].set_title('Non Social SPI', fontsize=12)
ax[1].set_xlabel('Preference Index', fontsize=12)
ax[1].set_ylabel('Rel. Frequency', fontsize=12)
ax[1].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax[1].legend()


# Plot data in THIRD subplot

sns.barplot(x=centers, y=a_s_nor_medium, color=[1.0, 0.0, 0.0], alpha=0.9,linewidth=4.0, ax=ax[2], label='Social')

# Customize the second subplot
ax[2].set_title('Social SPI', fontsize=12)
ax[2].set_xlabel('Preference Index', fontsize=12)
ax[2].set_ylabel('Rel. Frequency', fontsize=12)
ax[2].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

filename = analysisFolder + '/Fig3_SPI.png'
plt.savefig(filename, dpi=600)
#plt.show()
plt.close('all')




# # ==================================================================
# FIGURE 4 ===  BPS Summary Plot

# Make histogram and plot it with lines 
a_ns,c=np.histogram(BPS_NS_ALL,  bins=25, range=(0,6))
a_s,c=np.histogram(BPS_S_ALL,  bins=25, range=(0,6))
centers = (c[:-1]+c[1:])/2

#Normalize by tot number of fish
Tot_Fish_NS=numFiles

a_ns_float = np.float32(a_ns)
a_s_float = np.float32(a_s)

a_ns_nor_medium=a_ns_float/Tot_Fish_NS
a_s_nor_medium=a_s_float/Tot_Fish_NS  
 
#plt.figure()
fig, ax = plt.subplots(1,3,figsize=(10, 4))

# Plot data in FIRST subplot
sns.lineplot(x=centers, y=a_ns_nor_medium, color=[0.5, 0.5, 0.5, 1.0], linewidth=4.0, ax=ax[0], label='Non Social')
sns.lineplot(x=centers, y=a_s_nor_medium, color=[1.0, 0.0, 0.0, 0.5], linewidth=4.0, ax=ax[0], label='Social')
# By changing this  ax=ax[0] to  ax=ax[1] or  ax=ax[2], you indicate on what subplot to plot it. 

# Customize subplot
ax[0].set_title('Non Social vs Social BPS', fontsize=12)
ax[0].set_xlabel('BoutsPerseconds', fontsize=12)
ax[0].set_ylabel('Rel. Frequency', fontsize=12)
ax[0].set_yticks([0, 0.1, 0.2])
ax[0].legend()

# Plot data in SECOND subplot
# You need to make sure that centres is an array to plot the xticks evently. Otehrwise plotting on one tick. 
#centers = np.array([0.3, 0.9, 1.5, 2.1, 2.7, 3.3, 3.9, 4.5, 5.1, 5.7])

sns.lineplot(x=centers, y=a_ns_nor_medium, color=[0.5, 0.5, 0.5, 1.0],linewidth=4.0, ax=ax[1], label='Non Social')

# Customize subplot
ax[1].set_title('Non Social BPS', fontsize=12)
ax[1].set_xlabel('BoutsPerseconds', fontsize=12)
ax[1].set_ylabel('Rel. Frequency', fontsize=12)
ax[1].set_yticks([0, 0.1, 0.2])
ax[1].legend()


# Plot data in THIRD subplot

sns.lineplot(x=centers, y=a_s_nor_medium, color=[1.0, 0.0, 0.0], alpha=0.9,linewidth=4.0, ax=ax[2], label='Social')

# Customize the second subplot
ax[2].set_title('Social BPS', fontsize=12)
ax[2].set_xlabel('BoutsPerseconds', fontsize=12)
ax[2].set_ylabel('Rel. Frequency', fontsize=12)
ax[2].set_yticks([0, 0.1, 0.2])
ax[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

filename = analysisFolder + '/Fig4_BPS_FrequencyPerSec.png'
plt.savefig(filename, dpi=600)
#plt.show()
plt.close('all')



# # ==================================================================
# FIGURE 5 ===  BPS Summary Plot

plt.figure()
bar_width=0.005

# NON SOCIAL
visible_bouts = np.where(Bouts_NS_ALL[:,9] == 1)[0]
non_visible_bouts = np.where(Bouts_NS_ALL[:,9] == 0)[0]
#This returns the frame numbers of where bouts ar visible or not, based on teh Treue/False Bouts_NS-ALL [:,9] array 

#first graph Get all durations of bouts NS non visible 
plt.subplot(2,2,1)
bout_durations_ns, c = np.histogram(Bouts_NS_ALL[non_visible_bouts,8], bins=51, range=(0,50))
centers = (c[:-1]+c[1:])/2

plt.bar(centers/100, bout_durations_ns, width=bar_width, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
plt.title('Non Social Bout Durations (non-visible)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)

#second graph 
plt.subplot(2,2,3)
bout_durations_ns, c = np.histogram(Bouts_NS_ALL[visible_bouts,8], bins=51, range=(0,50))
centers = (c[:-1]+c[1:])/2

plt.bar(centers/100, bout_durations_ns, width=bar_width, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
plt.title('Non Social Bout Durations (visible)', fontsize=12)
plt.xlabel('Bout Durations (sec)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)

#SOCIAL
visible_bouts = np.where(Bouts_S_ALL[:,9] == 1)[0]
non_visible_bouts = np.where(Bouts_S_ALL[:,9] == 0)[0]

#Third graph 
plt.subplot(2,2,2)
bout_durations_s, c = np.histogram(Bouts_S_ALL[non_visible_bouts,8], bins=51, range=(0,50))
centers = (c[:-1]+c[1:])/2

plt.bar(centers/100, bout_durations_s, width=bar_width, color=[1.0,0.5,0.5,1.0], linewidth=4.0)
plt.title('Social Bout Durations (non-visible)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)

#Fourth graph
plt.subplot(2,2,4)
bout_durations_s, c = np.histogram(Bouts_S_ALL[visible_bouts,8], bins=51, range=(0,50))
centers = (c[:-1]+c[1:])/2
plt.bar(centers/100, bout_durations_s, width=bar_width, color=[1.0,0.5,0.5,1.0], linewidth=4.0)
plt.title('Social Bout Durations (visible)', fontsize=12)
plt.xlabel('Bout Durations (sec)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)


# Adjust layout to prevent overlap
plt.tight_layout()

filename = analysisFolder + '/Fig5_BoutDurations.png'  
plt.savefig(filename, dpi=600)
plt.close('all')



#Test  not used becasue I couldn't figure out the x axis with sns.barplot!

# # Set Seaborn theme
# sns.set_theme(style="whitegrid")

# # Create a figure for subplots
# fig, ax = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid of subplots
# bar_width = 0.005

# # NON-SOCIAL: Non-Visible Bouts
# visible_bouts = np.where(Bouts_NS_ALL[:, 9] == 1)[0]
# non_visible_bouts = np.where(Bouts_NS_ALL[:, 9] == 0)[0]

# bout_durations_ns, c = np.histogram(Bouts_NS_ALL[non_visible_bouts, 8], bins=51, range=(0, 50))
# centers = (c[:-1] + c[1:]) / 2
# sns.barplot(x=centers / 100, y=bout_durations_ns, ax=ax[0, 0], color="gray")
# ax[0, 0].set_title("Non-Social Bout Durations (Non-Visible)", fontsize=14)
# ax[0, 0].set_ylabel("Rel. Frequency", fontsize=12)
# ax[0, 0].set_xlabel("Bout Duration (sec)", fontsize=12)

# # Set x-ticks
# x_ticks = np.arange(0, max(centers) / 100 + 0.5, 0.5)  # Set ticks every 0.5 seconds
# x_ticks= np.array(x_ticks)
# ax[0, 0].set_xticks(x_ticks)
# #labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
# ax[0, 0].set_xticklabels([f"{tick:.0f}" for tick in x_ticks], fontsize=10)
# #ax[0, 0].set_xticklabels(labels)

# # NON-SOCIAL: Visible Bouts
# bout_durations_ns, c = np.histogram(Bouts_NS_ALL[visible_bouts, 8], bins=51, range=(0, 50))
# centers = (c[:-1] + c[1:]) / 2
# sns.barplot(x=centers / 100, y=bout_durations_ns, ax=ax[1, 0], color="gray")
# ax[1, 0].set_title("Non-Social Bout Durations (Visible)", fontsize=14)
# ax[1, 0].set_ylabel("Rel. Frequency", fontsize=12)
# ax[1, 0].set_xlabel("Bout Duration (sec)", fontsize=12)

# # Set x-ticks
# x_ticks = np.arange(0, max(centers) / 100 + 0.5, 0.5)  # Set ticks every 0.5 seconds
# ax[1, 0].set_xticks(x_ticks)
# ax[1, 0].set_xticklabels([f"{tick:.2f}" for tick in x_ticks], fontsize=10)

# # SOCIAL: Non-Visible Bouts
# visible_bouts = np.where(Bouts_S_ALL[:, 9] == 1)[0]
# non_visible_bouts = np.where(Bouts_S_ALL[:, 9] == 0)[0]

# bout_durations_s, c = np.histogram(Bouts_S_ALL[non_visible_bouts, 8], bins=51, range=(0, 50))
# centers = (c[:-1] + c[1:]) / 2
# sns.barplot(x=centers / 100, y=bout_durations_s, ax=ax[0, 1], color="salmon")
# ax[0, 1].set_title("Social Bout Durations (Non-Visible)", fontsize=14)
# ax[0, 1].set_ylabel("Rel. Frequency", fontsize=12)
# ax[0, 1].set_xlabel("Bout Duration (sec)", fontsize=12)

# # Set x-ticks
# x_ticks = np.arange(0, max(centers) / 100 + 0.5, 0.5)  # Set ticks every 0.5 seconds
# ax[0, 1].set_xticks(x_ticks)
# ax[0, 1].set_xticklabels([f"{tick:.2f}" for tick in x_ticks], fontsize=10)

# # SOCIAL: Visible Bouts
# bout_durations_s, c = np.histogram(Bouts_S_ALL[visible_bouts, 8], bins=51, range=(0, 50))
# centers = (c[:-1] + c[1:]) / 2
# sns.barplot(x=centers / 100, y=bout_durations_s, ax=ax[1, 1], color="salmon")
# ax[1, 1].set_title("Social Bout Durations (Visible)", fontsize=14)
# ax[1, 1].set_ylabel("Rel. Frequency", fontsize=12)
# ax[1, 1].set_xlabel("Bout Duration (sec)", fontsize=12)

# # Set x-ticks
# x_ticks = np.arange(0, max(centers) / 100 + 0.5, 0.5)  # Set ticks every 0.5 seconds
# ax[1, 1].set_xticks(x_ticks)
# ax[1, 1].set_xticklabels([f"{tick:.2f}" for tick in x_ticks], fontsize=10)

# # Adjust layout
# plt.tight_layout()

# # Save the figure
# filename = analysisFolder + '/Improved_Fig5_BoutDurations.png'
# plt.savefig(filename, dpi=600)
# plt.show()



# # ==================================================================
# FIGURE 6 ===    All Bouts Distribution Summary Plot

# A. B&W image
#plt.figure(figsize=(10, 4))


plt.subplot(3,2,1)
#fig.suptitle("Position Plots", fontsize=16)
plt.plot(Bouts_NS_ALL[:, 1], Bouts_NS_ALL[:, 2], '.', color=[0.0, 0.0, 0.0, 0.002])
plt.title('Non-Social All')
plt.axis([0, 17, 0, 42])
plt.gca().invert_yaxis()
    
plt.subplot(3,2,2)
plt.plot(Bouts_S_ALL[:, 1], Bouts_S_ALL[:, 2], '.', color=[0.0, 0.0, 0.0, 0.002])
plt.title('Social All')
plt.axis([0, 17, 0, 42])
plt.gca().invert_yaxis()


# filename = analysisFolder + '/Fig6_AllBoutsDistribution.png'  
# plt.savefig(filename, dpi=600)
# plt.close('all')

# B. Coloured image
df_ns = pd.DataFrame(Bouts_NS_ALL, columns=['Bout', 'X', 'Y', 'ort', 'Stop', 'XS', 'YS', '4', '4', '4'])
df_s = pd.DataFrame(Bouts_S_ALL, columns=['Bout', 'X', 'Y', 'ort', 'Stop', 'XS', 'YS', '4', '4', '4'])

# All Bouts Summary Plot
#plt.figure(figsize=(10, 4))

# Non-Social Bouts
plt.subplot(3, 2, 3)
sns.histplot(data=df_ns, x='X', y='Y', bins=80, pmax=1, cmap='viridis')
plt.title('Non-Social All')
plt.xlabel('X Position')
plt.ylabel('Y Position')
#plt.gca().invert_yaxis()
plt.axis([0, 17, 42, 0])

# Non-Social Bouts
plt.subplot(3, 2, 4)
sns.histplot(data=df_s, x='X', y='Y', bins=80, pmax=1, cmap='viridis')
plt.title('Social All')
plt.xlabel('X Position')
plt.ylabel('Y Position')
#plt.gca().invert_yaxis()
plt.axis([0, 17, 42, 0])
# filename = analysisFolder + '/Fig7_BoutsPosition_Visible.png'  
# plt.savefig(filename, dpi=600)
# #plt.show()
# plt.close('all')

# C ONLY VISIBLE BOUTS
# Assuming Bouts_NS_ALL and Bouts_S_ALL are your numpy arrays
# Convert numpy arrays to DataFrames. I only need the first three columns, but i need to specify all of them.
df_ns = pd.DataFrame(Bouts_NS_ALL, columns=['Bout', 'X', 'Y', 'ort', 'Stop', 'XS', 'YS', '4', '4', 'TF'])
df_s = pd.DataFrame(Bouts_S_ALL, columns=['Bout', 'X', 'Y', 'ort', 'Stop', 'XS', 'YS', '4', '4', 'TF'])

# Filter the DataFrame for rows where 'TF' is True
filtered_df_ns = df_ns[df_ns['TF'] == True]
filtered_df_s = df_s[df_s['TF'] == True]

# All Bouts Summary Plot
#plt.figure(figsize=(10, 4))

# Non-Social Bouts
plt.subplot(3, 2, 5)
#sns.histplot(data=df_ns, x='XS', y='YS', bins=80, pmax=1, cmap='viridis')
sns.histplot(data=filtered_df_ns, x='X', y='Y', bins=80, pmax=1, cmap='viridis')

plt.title('Non social visible')
plt.xlabel('X Position')
plt.ylabel('Y Position')
#plt.gca().invert_yaxis()
plt.axis([0, 17, 42, 0])

# Non-Social Bouts
plt.subplot(3, 2, 6)
#sns.histplot(data=df_s, x='XS', y='YS', bins=80, pmax=1, cmap='viridis')
sns.histplot(data=filtered_df_s, x='X', y='Y', bins=80, pmax=1, cmap='viridis')
plt.title('Social visible')
plt.xlabel('X Position')
plt.ylabel('Y Position')
#plt.gca().invert_yaxis()
plt.axis([0, 17, 42, 0])

# Adjust layout to prevent overlap
plt.tight_layout()

filename = analysisFolder + '/Fig6_BoutsPosition_Summary.png'  
plt.savefig(filename, dpi=600)
#plt.show()
plt.close('all')



# # ==================================================================
# # 
# # FIGURE 7  == PAUSES DURATION EVERYWHERE at at the Visible side

plt.figure()
bar_width=0.005

# NON SOCIAL
visible_pauses = np.where(Pauses_NS_ALL[:,9] == 1)[0]
non_visible_pauses = np.where(Pauses_NS_ALL[:,9] == 0)[0]
#This returns the frame numbers of where pauses ar visible or not, based on the Treue/False Bouts_NS-ALL [:,9] array 

#first graph Get all durations of bouts NS non visible 
plt.subplot(2,2,1)
pauses_durations_ns, c = np.histogram(Pauses_NS_ALL[non_visible_bouts,8], bins=51, range=(0,50))
centers = (c[:-1]+c[1:])/2

plt.bar(centers/100, pauses_durations_ns, width=bar_width, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
plt.title('Non Social Pauses Durations (non-visible)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)

#second graph 
plt.subplot(2,2,3)
pauses_durations_ns, c = np.histogram(Pauses_NS_ALL[visible_bouts,8], bins=51, range=(0,50))
centers = (c[:-1]+c[1:])/2

plt.bar(centers/100, pauses_durations_ns, width=bar_width, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
plt.title('Non Social Pauses Durations (visible)', fontsize=12)
plt.xlabel('Pauses Durations (sec)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)

#SOCIAL
visible_pauses = np.where(Pauses_S_ALL[:,9] == 1)[0]
non_visible_pauses = np.where(Pauses_S_ALL[:,9] == 0)[0]

#Third graph 
plt.subplot(2,2,2)
pauses_durations_s, c = np.histogram(Pauses_S_ALL[non_visible_pauses,8], bins=51, range=(0,50))
centers = (c[:-1]+c[1:])/2

plt.bar(centers/100, pauses_durations_s, width=bar_width, color=[1.0,0.5,0.5,1.0], linewidth=4.0)
plt.title('Social Pauses Durations (non-visible)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)

#Fourth graph
plt.subplot(2,2,4)
pauses_durations_s, c = np.histogram(Pauses_S_ALL[visible_pauses,8], bins=51, range=(0,50))
centers = (c[:-1]+c[1:])/2
plt.bar(centers/100, pauses_durations_s, width=bar_width, color=[1.0,0.5,0.5,1.0], linewidth=4.0)
plt.title('Social Bout Durations (visible)', fontsize=12)
plt.xlabel('Bout Durations (sec)', fontsize=12)
plt.ylabel('Rel. Frequency', fontsize=12)


# Adjust layout to prevent overlap
plt.tight_layout()

filename = analysisFolder + '/Fig7_PausesDurations.png'  
plt.savefig(filename, dpi=600)
plt.close('all')



# # ==================================================================

# FIGURE 8  === PAUSES Summary Plot (End of bout position)

plt.subplot(3,2,1)
#fig.suptitle("Position Plots", fontsize=16)
plt.plot(Bouts_NS_ALL[:, 5], Bouts_NS_ALL[:, 6], '.', color=[0.0, 0.0, 0.0, 0.002])
plt.title('Non-Social All')
plt.axis([0, 17, 0, 42])
plt.gca().invert_yaxis()
    
plt.subplot(3,2,2)
plt.plot(Bouts_S_ALL[:, 5], Bouts_S_ALL[:, 6], '.', color=[0.0, 0.0, 0.0, 0.002])
plt.title('Social All')
plt.axis([0, 17, 0, 42])
plt.gca().invert_yaxis()


# filename = analysisFolder + '/Fig6_AllBoutsDistribution.png'  
# plt.savefig(filename, dpi=600)
# plt.close('all')

# B. Coloured image
df_ns = pd.DataFrame(Bouts_NS_ALL, columns=['Bout', 'X', 'Y', 'ort', 'Stop', 'XS', 'YS', '4', '4', 'TF'])
df_s = pd.DataFrame(Bouts_S_ALL, columns=['Bout', 'X', 'Y', 'ort', 'Stop', 'XS', 'YS', '4', '4', 'TF'])

# All Bouts Summary Plot
#plt.figure(figsize=(10, 4))

# Non-Social Bouts
plt.subplot(3, 2, 3)
sns.histplot(data=df_ns, x='XS', y='YS', bins=80, pmax=1, cmap='viridis')
plt.title('Non-Social All')
plt.xlabel('X Position')
plt.ylabel('Y Position')
#plt.gca().invert_yaxis()
plt.axis([0, 17, 42, 0])

# Non-Social Bouts
plt.subplot(3, 2, 4)
sns.histplot(data=df_s, x='XS', y='YS', bins=80, pmax=1, cmap='viridis')
plt.title('Social All')
plt.xlabel('X Position')
plt.ylabel('Y Position')
#plt.gca().invert_yaxis()
plt.axis([0, 17, 42, 0])
# filename = analysisFolder + '/Fig7_BoutsPosition_Visible.png'  
# plt.savefig(filename, dpi=600)
# #plt.show()
# plt.close('all')

# C ONLY VISIBLE BOUTS
# Assuming Bouts_NS_ALL and Bouts_S_ALL are your numpy arrays
# Convert numpy arrays to DataFrames. I only need the first three columns, but i need to specify all of them.
df_ns = pd.DataFrame(Bouts_NS_ALL, columns=['Bout', 'X', 'Y', 'ort', 'Stop', 'XS', 'YS', '4', '4', 'TF'])
df_s = pd.DataFrame(Bouts_S_ALL, columns=['Bout', 'X', 'Y', 'ort', 'Stop', 'XS', 'YS', '4', '4', 'TF'])

# Filter the DataFrame for rows where 'TF' is True
filtered_df_ns = df_ns[df_ns['TF'] == True]
filtered_df_s = df_s[df_s['TF'] == True]

# All Bouts Summary Plot
#plt.figure(figsize=(10, 4))

# Non-Social Bouts
plt.subplot(3, 2, 5)
#sns.histplot(data=df_ns, x='XS', y='YS', bins=80, pmax=1, cmap='viridis')
sns.histplot(data=filtered_df_ns, x='XS', y='YS', bins=80, pmax=1, cmap='viridis')

plt.title('Non social visible')
plt.xlabel('X Position')
plt.ylabel('Y Position')
#plt.gca().invert_yaxis()
plt.axis([0, 17, 42, 0])

# Non-Social Bouts
plt.subplot(3, 2, 6)
#sns.histplot(data=df_s, x='XS', y='YS', bins=80, pmax=1, cmap='viridis')
sns.histplot(data=filtered_df_s, x='XS', y='YS', bins=80, pmax=1, cmap='viridis')
plt.title('Social visible')
plt.xlabel('X Position')
plt.ylabel('Y Position')
#plt.gca().invert_yaxis()
plt.axis([0, 17, 42, 0])

# Adjust layout to prevent overlap
plt.tight_layout()

filename = analysisFolder + '/Fig8_Pauses_PositionSummary.png'  
plt.savefig(filename, dpi=600)
#plt.show()
plt.close('all')



### ================================ 
# FIGIURE 9 POSITION PLOT FOR SELECTED PAUSESbased on figure 7

#Threshold in frames not seconds
threshold1 = 250
threshold2 = 300

plt.figure()

plt.title('Interval of pauses')
# Filter pauses between threshold1 and threshold2
filtered_pauses_ns = np.where((Pauses_NS_ALL[:, 8] > threshold1) & (Pauses_NS_ALL[:, 8] < threshold2))[0]
filtered_pauses_s = np.where((Pauses_S_ALL[:, 8] > threshold1) & (Pauses_S_ALL[:, 8] < threshold2))[0]

# Calculate the number of pauses per fish
num_filtered_pauses_per_fish_ns = len(filtered_pauses_ns) / numFiles
num_filtered_pauses_per_fish_ss = len(filtered_pauses_s) / numFiles

# ----------------
# range Pauses Plot
plt.subplot(3, 2, 1)
#plt.title('NS: #Filtered Pauses per fish = ' + format(num_filtered_pauses_per_fish_ns, '.0f'))
plt.plot(Pauses_NS_ALL[filtered_pauses_ns, 1], Pauses_NS_ALL[filtered_pauses_ns, 2], 'o', color=[0.0, 0.0, 0.0, 0.2])
plt.axis([0, 17, 0, 42])
plt.gca().invert_yaxis()
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.tight_layout()

plt.subplot(3, 2, 2)
#plt.title('NS: #Filtered Pauses per fish = ' + format(num_filtered_pauses_per_fish_ns, '.0f'))
plt.plot(Pauses_S_ALL[filtered_pauses_s, 1], Pauses_S_ALL[filtered_pauses_s, 2], 'o', color=[0.0, 0.0, 0.0, 0.2])
plt.axis([0, 17, 0, 42])
plt.gca().invert_yaxis()
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.tight_layout()

# ----------------
# Short Pauses Plot
plt.subplot(3,2,3)
short_pauses_ns = np.where(Pauses_NS_ALL[:,8] > freeze_threshold)[0]
num_short_pauses_per_fish_ns = len(short_pauses_ns)/numFiles
plt.title('NS: #Short Pauses = ' + format(num_short_pauses_per_fish_ns, '.4f'))
plt.plot(Pauses_NS_ALL[short_pauses_ns, 1], Pauses_NS_ALL[short_pauses_ns, 2], 'o', color=[0.0, 0.0, 0.0, 0.2])
plt.axis([0, 17, 0, 42])
plt.gca().invert_yaxis()
    
plt.subplot(3,2,4)
short_pauses_s = np.where(Pauses_S_ALL[:,8] > freeze_threshold)[0]
num_short_pauses_per_fish_s = len(short_pauses_s)/numFiles
plt.title('S: #Short Pauses = ' + format(num_short_pauses_per_fish_s, '.4f'))
plt.plot(Pauses_S_ALL[short_pauses_s, 1], Pauses_S_ALL[short_pauses_s, 2], 'o', color=[0.0, 0.0, 0.0, 0.2])
plt.axis([0, 17, 0, 42])
plt.gca().invert_yaxis()


# ----------------
# Long Pauses Plot
plt.subplot(3,2,5)
long_pauses_ns = np.where(Pauses_NS_ALL[:,8] > long_freeze_threshold)[0]
num_long_pauses_per_fish_ns = len(long_pauses_ns)/numFiles
plt.title('NS: #Long Pauses = ' + format(num_long_pauses_per_fish_ns, '.4f'))
plt.plot(Pauses_NS_ALL[long_pauses_ns, 1], Pauses_NS_ALL[long_pauses_ns, 2], 'o', color=[0.0, 0.0, 0.0, 0.2])
plt.axis([0, 17, 0, 42])
plt.gca().invert_yaxis()
    
plt.subplot(3,2,6)
long_pauses_s = np.where(Pauses_S_ALL[:,8] > long_freeze_threshold)[0]
num_long_pauses_per_fish_s = len(long_pauses_s)/numFiles
plt.title('S: #Long Pauses = ' + format(num_long_pauses_per_fish_s, '.0f'))
plt.plot(Pauses_S_ALL[long_pauses_s, 1], Pauses_S_ALL[long_pauses_s, 2], 'o', color=[0.0, 0.0, 0.0, 0.2])
plt.axis([0, 17, 0, 42])
plt.gca().invert_yaxis()
plt.tight_layout()
filename = analysisFolder + '/Fig9_PositionFilteredPauses.png'  
plt.savefig(filename, dpi=600)
plt.close('all')

### ================================ 
# FIGIURE 10  Correlation SPI Plots


plt.subplot(3,2,1)
plt.title('SVPI vs Freezes (NS)', fontsize=8)
plt.title('VPI vs BPS (NS)', fontsize=8)
plt.plot(VPI_NS_ALL, BPS_NS_ALL, '.', color=[0.0, 0.0, 0.0, 0.5])
plt.xlabel('Visual Preference Index (VPI)', fontsize=8)
plt.ylabel('Bouts per Second', fontsize=8)
plt.axis([-1.1, 1.1, 0, max(BPS_NS_ALL)+0.5])
pl.xticks([-1, -0.5, 0, 0.5, 1.0])
    
plt.subplot(3,2,2)
plt.title('VPI vs BPS (S)', fontsize=8)
plt.plot(VPI_S_ALL, BPS_S_ALL, '.', color=[0.0, 0.0, 0.0, 0.5])
plt.xlabel('Visual Preference Index (VPI)', fontsize=8)
plt.axis([-1.1, 1.1, 0,  max(BPS_S_ALL)+0.5])
pl.xticks([-1, -0.5, 0, 0.5, 1.0])
plt.tight_layout()

# # ----------------
# VPI vs Distance Traveled Summary Plot 

plt.subplot(3,2,3)
plt.title('VPI vs Freezes (NS)', fontsize=8)
plt.title('VPI vs Distance Traveled (NS)', fontsize=8)
plt.plot(VPI_NS_ALL, Distance_NS_ALL, '.', color=[0.0, 0.0, 0.0, 0.5])
plt.xlabel('Visual Preference Index (VPI)', fontsize=8)
plt.ylabel('Distance (mm)', fontsize=8)
plt.axis([-1.1, 1.1, 0, max(Distance_NS_ALL)+0.5])
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=8)
    
plt.subplot(3,2,4)
plt.title('VPI vs Distance Traveled (S)', fontsize=8)
plt.plot(VPI_S_ALL, Distance_S_ALL, '.', color=[0.0, 0.0, 0.0, 0.5])
plt.xlabel('Visual Preference Index (VPI)', fontsize=8)
plt.axis([-1.1, 1.1, 0,  max(Distance_S_ALL)+0.5])
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=8)
plt.tight_layout()

# # ----------------
# VPI vs Number of Long Pauses Summary Plot 

plt.subplot(3,2,5)
plt.title('VPI vs Freezes (NS)', fontsize=8)
plt.plot(VPI_NS_ALL, Freezes_NS_ALL, '.', color=[0.0, 0.0, 0.0, 0.5])
plt.xlabel('Visual Preference Index (VPI)', fontsize=8)
plt.ylabel('Freezes (count)', fontsize=8)
plt.axis([-1.1, 1.1, -1, max(Freezes_NS_ALL)+0.5])
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=8)
    
plt.subplot(3,2,6)
plt.title('VPI vs Freezes (S)', fontsize=8)
plt.plot(VPI_S_ALL, Freezes_S_ALL, '.', color=[0.0, 0.0, 0.0, 0.5])
plt.xlabel('Visual Preference Index (VPI)', fontsize=8)
plt.axis([-1.1, 1.1, -1, max(Freezes_S_ALL)+0.5])
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=8)
plt.tight_layout()
plt.rcParams.update({'font.size': 8}) 
filename = analysisFolder + '/Fig10_Correlations_VPI.png'
plt.savefig(filename, dpi=600)
plt.close('all')




### ================================ 
# FIGIURE 11  Correlation VPI Plots

plt.subplot(3,2,1)
plt.title('SPI vs Freezes (NS)', fontsize=8)
plt.title('SPI vs BPS (NS)', fontsize=8)
plt.plot(SPI_NS_ALL, BPS_NS_ALL, '.', color=[0.0, 0.0, 0.0, 0.5])
plt.xlabel('Visual Preference Index (SPI)', fontsize=8)
plt.ylabel('Bouts per Second', fontsize=8)
plt.axis([-1.1, 1.1, 0, max(BPS_NS_ALL)+0.5])
pl.xticks([-1, -0.5, 0, 0.5, 1.0])
    
plt.subplot(3,2,2)
plt.title('SPI vs BPS (S)', fontsize=8)
plt.plot(SPI_S_ALL, BPS_S_ALL, '.', color=[0.0, 0.0, 0.0, 0.5])
plt.xlabel('Visual Preference Index (SPI)', fontsize=8)
plt.axis([-1.1, 1.1, 0,  max(BPS_S_ALL)+0.5])
pl.xticks([-1, -0.5, 0, 0.5, 1.0])


# # ----------------
# VPI vs Distance Traveled Summary Plot 

plt.subplot(3,2,3)
plt.title('SPI vs Freezes (NS)', fontsize=8)
plt.title('SPI vs Distance Traveled (NS)', fontsize=8)
plt.plot(SPI_NS_ALL, Distance_NS_ALL, '.', color=[0.0, 0.0, 0.0, 0.5])
plt.xlabel('Visual Preference Index (SPI)', fontsize=8)
plt.ylabel('Distance (mm)', fontsize=8)
plt.axis([-1.1, 1.1, 0, max(Distance_NS_ALL)+0.5])
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=8)
    
plt.subplot(3,2,4)
plt.title('SPI vs Distance Traveled (S)', fontsize=8)
plt.plot(SPI_S_ALL, Distance_S_ALL, '.', color=[0.0, 0.0, 0.0, 0.5])
plt.xlabel('Visual Preference Index (SPI)', fontsize=8)
plt.axis([-1.1, 1.1, 0,  max(Distance_S_ALL)+0.5])
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=8)


# # ----------------
# VPI vs Number of Long Pauses Summary Plot 

plt.subplot(3,2,5)
plt.title('SPI vs Freezes (NS)', fontsize=8)
plt.plot(SPI_NS_ALL, Freezes_NS_ALL, '.', color=[0.0, 0.0, 0.0, 0.5])
plt.xlabel('Visual Preference Index (SPI)', fontsize=8)
plt.ylabel('Freezes (count)', fontsize=8)
plt.axis([-1.1, 1.1, -1, max(Freezes_NS_ALL)+0.5])
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=8)
    
plt.subplot(3,2,6)
plt.title('SPI vs Freezes (S)', fontsize=8)
plt.plot(SPI_S_ALL, Freezes_S_ALL, '.', color=[0.0, 0.0, 0.0, 0.5])
plt.xlabel('Visual Preference Index (SPI)', fontsize=8)
plt.axis([-1.1, 1.1, -1, max(Freezes_S_ALL)+0.5])
pl.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=8)
plt.tight_layout()
plt.rcParams.update({'font.size': 8}) 
filename = analysisFolder + '/Fig11_Correlations_SPI.png'
plt.savefig(filename, dpi=600)
plt.close('all')




### ================================ 
# FIGIURE 12  ORT_HIST Summary Plot

# Accumulate all histogram values and normalize
# OrtHist_NS_NSS_ALL is an array of 36 columns (which are bins of 10 degrees angles0 and each row is how many times the fish was in that bin)
Accum_OrtHist_NS_NSS_ALL = np.sum(OrtHist_NS_NSS_ALL, axis=0) # get an array with all the sum per bin
Accum_OrtHist_NS_SS_ALL = np.sum(OrtHist_NS_SS_ALL, axis=0)
Accum_OrtHist_S_NSS_ALL = np.sum(OrtHist_S_NSS_ALL, axis=0)
Accum_OrtHist_S_SS_ALL= np.sum(OrtHist_S_SS_ALL, axis=0)

#Normalise by tot number of fish positions
Norm_OrtHist_NS_NSS_ALL = Accum_OrtHist_NS_NSS_ALL/np.sum(Accum_OrtHist_NS_NSS_ALL)
Norm_OrtHist_NS_SS_ALL = Accum_OrtHist_NS_SS_ALL/np.sum(Accum_OrtHist_NS_SS_ALL)
Norm_OrtHist_S_NSS_ALL = Accum_OrtHist_S_NSS_ALL/np.sum(Accum_OrtHist_S_NSS_ALL)
Norm_OrtHist_S_SS_ALL = Accum_OrtHist_S_SS_ALL/np.sum(Accum_OrtHist_S_SS_ALL)

# Generates angles evenly spaced around a circle in radians, which is useful for polar plots or orientation histograms.
xAxis = np.arange(-np.pi,np.pi+np.pi/18.0, np.pi/18.0)
#Example of xAxis [-3.14159, -2.96706, -2.79253, ..., 2.96706, 3.14159, 3.31612]

#plt.figure('Summary: Orientation Histograms')
plt.figure()

ax = plt.subplot(2,2,1, polar=True)
plt.title('NS - Non Social Side')
plt.plot(xAxis, np.hstack((Norm_OrtHist_NS_NSS_ALL, Norm_OrtHist_NS_NSS_ALL[0])), linewidth = 3)

ax = plt.subplot(2,2,2, polar=True)
plt.title('NS - Social Side')
plt.plot(xAxis, np.hstack((Norm_OrtHist_NS_SS_ALL, Norm_OrtHist_NS_SS_ALL[0])), linewidth = 3)

ax = plt.subplot(2,2,3, polar=True)
plt.title('S - Non Social Side')
plt.plot(xAxis, np.hstack((Norm_OrtHist_S_NSS_ALL, Norm_OrtHist_S_NSS_ALL[0])), linewidth = 3)

ax = plt.subplot(2,2,4, polar=True)
plt.title('S - Social Side')
plt.plot(xAxis, np.hstack((Norm_OrtHist_S_SS_ALL, Norm_OrtHist_S_SS_ALL[0])), linewidth = 3)
filename = analysisFolder + '/Fig12_ORT_HIST.png'
plt.savefig(filename, dpi=600)
plt.close('all')


# TEST Generate synthetic orientation data
# Prepare data for the heatmap
heatmap_dataq = np.array([
    Norm_OrtHist_NS_NSS_ALL,
    Norm_OrtHist_NS_SS_ALL,
])

heatmap_data2 = np.array([
    Norm_OrtHist_S_NSS_ALL,
    Norm_OrtHist_S_SS_ALL
])

# Heatmap labels
x_labels_degrees = np.round(np.degrees(np.linspace(-np.pi, np.pi, len(Norm_OrtHist_NS_NSS_ALL))), 0)  # Convert radians to degrees
x_labels_degrees = [int(label) for label in x_labels_degrees]  # Convert to integers

y_labels1 = ["NS Non-Social", "NS Social"]  # Y-axis
y_labels2 = ["S Non-Social", "S Social"]  # Y-axis

# Create the heatmap plot
plt.figure(figsize=(10, 6))

plt.subplot(2,1,1)
sns.heatmap(
    heatmap_data2,
    xticklabels=x_labels_degrees,
    yticklabels=y_labels1,
    cmap="viridis",
    cbar_kws={'label': 'Normalized Counts'},
    linewidths=0.5
)
plt.title("Orientation Histogram Heatmap", fontsize=14)
plt.xlabel("Orientation (radians)", fontsize=12)
plt.ylabel("Condition", fontsize=12)
plt.tight_layout()

plt.subplot(2,1,2)
sns.heatmap(
    heatmap_data2,
    xticklabels=x_labels_degrees,
    yticklabels=y_labels2,
    cmap="viridis",
    cbar_kws={'label': 'Normalized Counts'},
    linewidths=0.5
)
plt.title("Orientation Histogram Heatmap", fontsize=14)
plt.xlabel("Orientation (radians)", fontsize=12)
plt.ylabel("Condition", fontsize=12)
plt.tight_layout()


# Save the heatmap
filename = analysisFolder + '/Fig12_ORT_HIST_HEATMAP.png'
plt.savefig(filename, dpi=600)
#plt.show()
plt.close('all')



### ================================ 
# FIGIURE 13  ORT_HIST Summary Plot
plt.figure()

# VPI
plt.subplot(2,2,1)
plt.title('VPI')
s1 = pd.Series(VPI_NS_ALL, name='NS')
s2 = pd.Series(VPI_S_ALL, name='S')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], errorbar=('ci', 95), capsize=0.05, err_kws={'linewidth': 2})
sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="auto")

# BPS
plt.subplot(2,2,2)
plt.title('BPS')
s1 = pd.Series(BPS_NS_ALL, name='NS')
s2 = pd.Series(BPS_S_ALL, name='S')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], errorbar=('ci', 95), capsize=0.05, err_kws={'linewidth': 2})
sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="auto")

plt.subplot(2,2,3)
plt.title('Distance Traveled')
s1 = pd.Series(Distance_NS_ALL, name='NS')
s2 = pd.Series(Distance_S_ALL, name='S')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], errorbar=('ci', 95), capsize=0.05, err_kws={'linewidth': 2})
sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="auto")


# Freezes
plt.subplot(2,2,4)
plt.title('Freezes')
s1 = pd.Series(Freezes_NS_ALL, name='NS')
s2 = pd.Series(Freezes_S_ALL, name='S')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], errorbar=('ci', 95), capsize=0.05, err_kws={'linewidth': 2})
sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="auto")
filename = analysisFolder + '/Fig13_SummaryPlots.png'
plt.savefig(filename, dpi=600)
plt.close('all')

#################

# FIGURE TEST ===  SPI "Binned" in 1 minute BINS 

 
plt.figure()
fig, ax = plt.subplots(1,2,figsize=(8,6))
fig.suptitle("Temporal SPI (one minute bins)", fontsize=16)

plt.subplot(1,2,1)

# Compute mean, standard deviation, and standard error of NS
mean = np.nanmean(SPI_NS_BINS_ALL, 0)
std = np.nanstd(SPI_NS_BINS_ALL, 0)
valid = (np.logical_not(np.isnan(VPI_NS_BINS_ALL)))
n = np.sum(valid, 0)
se = std/np.sqrt(n) # Standard error
upper_bound = mean + se
lower_bound = mean - se

# Prepare x label values for plotting
x_values = np.arange(SPI_NS_BINS_ALL.shape[1])  # 0 to 14 (15 bins)

# Plot all trajectories in one go
plt.plot(x_values, SPI_NS_BINS_ALL.T, color=[0, 0, 0, 0.1], linewidth=1)

# Plot the mean line
sns.lineplot(x=x_values, y=mean, color="black", linewidth=2,  ax=ax[0],label="Mean")
# Add dots for each data point
plt.scatter(x_values, mean, color="black", s=40, zorder=3)

# Plot the standard deviation bounds
plt.plot(x_values, upper_bound, color="red", linewidth=1, linestyle="-", label="Mean + Std Dev")
plt.plot(x_values, lower_bound, color="red", linewidth=1, linestyle="-", label="Mean - Std Dev")

# Set custom y-ticks (optional based on data range)
#plt.yticks(ticks=[-1, -0.5, 0, 0.5, 1], labels=["-1", "-0.5", "0", "0.5", "1"])
plt.xlabel('minutes')
plt.ylabel('VPI')

# Customize the  subplot
ax[0].set_title('Non_Social', fontsize=12)
ax[0].set_xlabel('minutes', fontsize=12)
ax[0].set_ylabel('VPI', fontsize=12)
ax[0].legend()



# =======  Social 
plt.subplot(1,2,2)

# Compute mean, standard deviation, and standard error of NS
mean2 = np.nanmean(SPI_S_BINS_ALL, 0)
std2 = np.nanstd(SPI_S_BINS_ALL, 0)
valid2 = (np.logical_not(np.isnan(VPI_S_BINS_ALL)))
n2 = np.sum(valid2, 0)
se2 = std/np.sqrt(n2) # Standard error
upper_bound2 = mean2 + se2
lower_bound2 = mean2 - se2

# Prepare x label values for plotting
x_values2 = np.arange(SPI_S_BINS_ALL.shape[1])  # 0 to 14 (15 bins)

# Plot all trajectories in one go
plt.plot(x_values2, SPI_S_BINS_ALL.T, color=[0, 0, 0, 0.1], linewidth=1)

# Plot the mean line
sns.lineplot(x=x_values2, y=mean2, color="black",  ax=ax[1], linewidth=2, label="Mean")
# Add dots for each data point
plt.scatter(x_values2, mean2, color="black", s=40, zorder=3)

# Plot the standard deviation bounds
plt.plot(x_values2, upper_bound2, color="red", linewidth=1, linestyle="-", label="Mean + Std Dev")
plt.plot(x_values2, lower_bound2, color="red", linewidth=1, linestyle="-", label="Mean - Std Dev")

# Customize the  subplot
ax[1].set_title('Social', fontsize=12)
ax[1].set_xlabel('minutes', fontsize=12)
ax[1].set_ylabel('VPI', fontsize=12)
ax[1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
#plt.show()

filename = analysisFolder + '/Fig14_SPI_1min_BINS.png'
plt.savefig(filename, dpi=600)
plt.close('all')


# FIN
