# -*- coding: utf-8 -*-
"""
Created on Sun Oct 6 18:09:02 2019

@author: thoma
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'
base_path=r'Z:\D R E O S T I   L A B\Hande\Behaviours\Robustness'
# Analysis folders including the two conditions
analysisFolder = base_path + '\Analysis_folder'
amFolder = analysisFolder + r'\Combined_Morning'
pmFolder = analysisFolder + r'\Combined_Afternoon'

# Set Library Paths and import libraries
import sys
sys.path.append(lib_path)
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob
import random as rd
# -----------------------------------------------------------------------------
# Function to load all summary statistics and make a figure comparing paired 
# morning and afternoon sessions. Analysis includes regression and bootstrap
# -----------------------------------------------------------------------------

# Find all the summary npz files saved for morning and afternoon in step3_code
amNpzFiles = glob.glob(amFolder + '\*.npz')
pmNpzFiles = glob.glob(pmFolder + '\*.npz')
numAmFiles = np.size(amNpzFiles, 0)
numPmFiles = np.size(pmNpzFiles, 0)

# create array to store the data
VPIBeforeAfter=np.zeros((numAmFiles,2))

# extract relevant lists from AM then PM files. store these in the array
for f,filename in enumerate(amNpzFiles):
    #load npz file
    dataobject = np.load(filename)
    
    # extract relevant info from both files
    VPIBeforeAfter[f,0]=dataobject['VPI_S']
    filename=pmNpzFiles[f]
    dataobject = np.load(filename)
    VPIBeforeAfter[f,1]=dataobject['VPI_S']
    
# make easier labels for plotting
xd=VPIBeforeAfter[:,0]
yd=VPIBeforeAfter[:,1]
plt.figure()
plt.scatter(xd, yd)
plt.xlabel('Morning VPI')
plt.ylabel('Afternoon VPI')
plt.title('Linear regression of VPI Morning vs. Afternoon')
# plot line of best fit (trend line)
coeffs = np.polyfit(xd, yd,deg=1) # a polynomial with deg=1 makes a straight line

# may be a better way of doing this, but here I grab the coefficients to actually draw a line
intercept = coeffs[-1]
slope = coeffs[-2]
power = coeffs[0]

#define the range of xdata
minxd = np.min(xd)
maxxd = np.max(xd)

# define the line
xl = np.array([minxd, maxxd])
yl = power * xl ** 2 + slope * xl + intercept

#draw the line
plt.plot(xl, yl)

# found another way of doing it using linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(xd,yd)
print("Regression R-square =",r_value)
print("Regression p-value =",p_value)
print("Regression slope =",slope)

#%%----------------------------------------------------------------------------
# Bootstrap test

# What is the absolute mean difference between morning and afternoon?
actualVPIs = (yd-xd)
actualAbsVPIs = np.abs(yd-xd)
actualVPI = np.mean(np.abs(actualVPIs))

# Randomise the pairings nboot times and find population mean VPI shifts to 
# to obtain our test distribution
nboot = 10000
randVPIs = np.zeros(nboot)
for i in range(nboot):
    randXd=xd
    randYd=yd
    rd.shuffle(randXd)
    rd.shuffle(randYd)
    randVPIs[i] = np.mean(np.abs(randYd - randXd))

# Plot distribution of absolute actual difference in VPI
actualAbsVPIs_Hist, edges = np.histogram(actualAbsVPIs, bins=20, range=[-2,2])
centres = edges[:-1] + 0.1
plt.figure()
plt.bar(centres, actualAbsVPIs_Hist, width=0.2, align='center', color=[0,0,0,1])
plt.title('Robustness of VPI, Morning vs. Afternoon (Absolute)')
plt.show()

# find sensible xlimits from the data
plt.xlim([np.min(actualAbsVPIs)-0.05,np.max(actualAbsVPIs)+0.05])

# Plot distribution of actual difference in VPI
actualVPIs_Hist, edges = np.histogram(actualVPIs, bins=20, range=[-2,2])
centres = edges[:-1] + 0.1
plt.figure()
plt.bar(centres, actualVPIs_Hist, width=0.2, align='center', color=[0,0,0,1])
plt.title('Robustness of VPI, Morning vs. Afternoon (Actual)')
plt.show()
plt.xlim([np.min(actualVPIs)-0.05,np.max(actualVPIs)+0.05])

# Plot random pairing VPIs and show where our actual VPI lies
randVPIs_Hist, edges = np.histogram(randVPIs, bins=1000, range=[-2,2])
centres = edges[:-1] + 0.001
plt.figure()
plt.title('Randomised pairings vs. Actual')
plt.bar(centres, randVPIs_Hist, width=0.02, align='center', color=[0,0,0,1])
plt.vlines(actualVPI, 0, np.max(randVPIs_Hist),'r')
plt.show()

# find sensible xlimits from the data
plt.xlim([np.min(randVPIs)-0.05,np.max(randVPIs)+0.05])

## Create PDF for randVPIs_Hist to read off value of actual VPI
#randVPIs_pdf=randVPIs
#randVPIs_pdf/=np.sum(randVPIs)

# easier to count the number of randVPIs that are lower than the actual 
# if the actual mean is lower than the mean random difference or higher if 
# actual mean is higher.
randMean=np.mean(randVPIs)

if actualVPI<randMean:
    count = np.sum(randVPIs < actualVPI) # performs the sum of the boolean
    Pvalue=1-(count/nboot)
else:
    count = np.sum(randVPIs < actualVPI)
    Pvalue=1-(count/nboot)



print('P-value =' + str(Pvalue))
print('Mean actual VPI difference (absolute) =' + str(actualVPI))
print('Mean random VPI difference (absolute) =' + str(np.mean(randVPIs)))
checkDirection=np.mean(actualVPIs)
if checkDirection<0:
    print('On average, VPIs fall by ' + str(checkDirection))
elif checkDirection>0:
     print('On average, VPIs rise by ' + str(checkDirection))

    