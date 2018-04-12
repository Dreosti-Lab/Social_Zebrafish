# -*- coding: utf-8 -*-
"""
This script loads and processes a cFos folder list: .nii images and behaviour

@author: Dreosti Lab
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import SZ_cfos as SZCFOS
import SZ_summary as SZS
import SZ_analysis as SZA

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Set Summary List
#summaryListFile = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Excel_Summary_List_ARK.xlsx'
#summaryListFile = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Excel_Summary_List_Isolated.xlsx'
summaryListFile = r'\\128.40.155.187\data\D R E O S T I   L A B\Isolation_Experiments\Social_Brain_Areas_Analisys\Excel_Summary_List_Isolated_No_SC.xlsx'

# Set Mask Path
mask_path = r'D:\Anatomical_Segmentation\Diencephalon_Causal_Hypothalamus_Area1.tif'
#mask_path = r'D:\Anatomical_Segmentation\Diencephalon_Area6.tif'

# Set Background Path
background_path = r'D:\Anatomical_Segmentation\Diencephalon_Area_8.labels.tif'

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
npz_files, cfos_files = SZCFOS.read_summarylist(summaryListFile)
num_files = len(cfos_files)

# Load mask(s)
mask_stack = SZCFOS.load_mask(mask_path)
num_mask_voxels = np.sum(np.sum(np.sum(mask_stack)))

background_stack = SZCFOS.load_mask(background_path)
num_background_voxels = np.sum(np.sum(np.sum(background_stack)))

# ------------------------------------------------------------------
# Behaviour Analysis
# ------------------------------------------------------------------

# Freeze time threshold
freeze_threshold = 500

# Load Behaviour Analysis
VPI_NS_ALL = np.zeros(num_files) 
VPI_S_ALL = np.zeros(num_files)
SPI_NS_ALL = np.zeros(num_files)
SPI_S_ALL = np.zeros(num_files)
BPS_NS_ALL = np.zeros(num_files)
BPS_S_ALL = np.zeros(num_files)
Distance_NS_ALL = np.zeros(num_files)
Distance_S_ALL = np.zeros(num_files)
Freezes_NS_ALL = np.zeros(num_files)
Freezes_S_ALL = np.zeros(num_files)
OrtHist_NS_NSS_ALL = np.zeros((num_files,36))
OrtHist_NS_SS_ALL = np.zeros((num_files,36))
OrtHist_S_NSS_ALL = np.zeros((num_files,36))
OrtHist_S_SS_ALL = np.zeros((num_files,36))
Bouts_NS_ALL = np.zeros((0,10))
Bouts_S_ALL = np.zeros((0,10))
Pauses_NS_ALL = np.zeros((0,10))   
Pauses_S_ALL = np.zeros((0,10))

# Go through all the npz files
for f, filename in enumerate(npz_files):

    #Load each npz file
    dataobject = np.load(filename)
    
    #Extract from the npz file
    VPI_NS = dataobject['VPI_NS']    
    VPI_S = dataobject['VPI_S']   
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

    # Count Freezes
    Freezes_NS_ALL[f] = np.sum(Pauses_NS[:,8] > freeze_threshold)
    Freezes_S_ALL[f] = np.sum(Pauses_S[:,8] > freeze_threshold)
    
    #Make an array with all summary stats
    VPI_NS_ALL[f] = VPI_NS
    VPI_S_ALL[f] = VPI_S
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
    
    # Somehow concat all Pauses/Bouts
    Bouts_NS_ALL = np.vstack([Bouts_NS_ALL, Bouts_NS])
    Bouts_S_ALL = np.vstack([Bouts_S_ALL, Bouts_S])
    Pauses_NS_ALL = np.vstack([Pauses_NS_ALL, Pauses_NS])
    Pauses_S_ALL = np.vstack([Pauses_S_ALL, Pauses_S])


# ------------------------------------------------------------------
# Imaging Analysis
# ------------------------------------------------------------------

# Measure cFOS in Mask (normalize to "background")
signal_values = np.zeros(num_files)
background_values = np.zeros(num_files)
normalized_cFos_values = np.zeros(num_files)
for i in range(num_files):
    cfos_data = SZCFOS.load_nii(cfos_files[i])
    signal_value = np.sum(np.sum(np.sum(mask_stack * cfos_data)))/num_mask_voxels
    background_value = np.sum(np.sum(np.sum(background_stack * cfos_data)))/num_background_voxels
                             
    # Append to list
    signal_values[i] = signal_value
    background_values[i] = background_value
    normalized_cFos_values[i] = signal_value/background_value
    print(cfos_files[i])
    print(str(i) + ", cFos = " + format(normalized_cFos_values[i], '.3f') + ", VPI = " + format(VPI_S_ALL[i], '.3f') + ", BPS = " + format(BPS_S_ALL[i], '.3f'))

# Make plots
plt.figure()

# Plot unnormalized data
plt.subplot(2,4,1)
plt.title("BPS vs cFos - UnNormalized")
plt.plot(BPS_S_ALL, signal_values, 'b.')
plt.plot(BPS_S_ALL, background_values, 'r.')

plt.subplot(2,4,2)
plt.title("VPI vs cFos - UnNormalized")
plt.plot(VPI_S_ALL, signal_values, 'b.')
plt.plot(VPI_S_ALL, background_values, 'r.')

plt.subplot(2,4,3)
plt.title("Freezes vs cFos - UnNormalized")
plt.plot(Freezes_S_ALL, signal_values, 'b.')
plt.plot(Freezes_S_ALL, background_values, 'r.')

plt.subplot(2,4,4)
plt.title("Distance (NS) vs cFos - UnNormalized")
plt.plot(Distance_NS_ALL, signal_values, 'b.')
plt.plot(Distance_NS_ALL, background_values, 'r.')

# Plot normalized data
plt.subplot(2,4,5)
plt.title("BPS vs cFos - Normalized by DA8")
plt.plot(BPS_S_ALL, normalized_cFos_values, 'k.')

plt.subplot(2,4,6)
plt.title("VPI vs cFos - Normalized by DA8")
plt.plot(VPI_S_ALL, normalized_cFos_values, 'k.')

plt.subplot(2,4,7)
plt.title("Freezes vs cFos - Normalized by DA8")
plt.plot(Freezes_S_ALL, normalized_cFos_values, 'k.')

plt.subplot(2,4,8)
plt.title("Distance (NS) vs cFos - Normalized by DA8")
plt.plot(Distance_NS_ALL, normalized_cFos_values, 'k.')

# FIN
