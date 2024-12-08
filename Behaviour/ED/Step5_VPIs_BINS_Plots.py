# -*- coding: utf-8 -*-
"""
Compare summaries of analyzed social preference experiments

@author: Elena Dreosti
"""
## This script plots the VPI and SPI Bins over the 15 min
## It takes all the conditions (w/t and 10 mutants) and compares them

# MISSING:
# - add SPI plots. 
# - add statistics


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
from matplotlib.colors import to_rgba

# Import local modules
import BONSAI_ARK_ED
import SZ_statistics_ED as SZSt
import glob
import pylab as pl

# Set analysis folder and label for experiment/condition A
# and set conditions: descriptive labels corresponding to each folder, allowing the script to reference experimental groups 
analysisFolder_A = base_path + r'/Scrambled/Analysis_TPI'
conditionName_A = "w/t"
# Set analysis folder and label for experiment/condition B
analysisFolder_B = base_path + r'/Akap11/Analysis_TPI'
conditionName_B = "A11"
# Set analysis folder and label for experiment/condition C
analysisFolder_C = base_path + r'/cacnag1g/Analysis_TPI'
conditionName_C = "c1g"
# Set analysis folder and label for experiment/condition D
analysisFolder_D = base_path + r'/gria3/Analysis_TPI'
conditionName_D = "G3"
# Set analysis folder and label for experiment/condition E
analysisFolder_E = base_path + r'/grin2a/Analysis_TPI'
conditionName_E = "G2a"
# Set analysis folder and label for experiment/condition F
analysisFolder_F = base_path + r'/hcn4/Analysis_TPI'
conditionName_F = "H4"
# Set analysis folder and label for experiment/condition G
analysisFolder_G = base_path + r'/herc1/Analysis_TPI'
conditionName_G = "H1"
# Set analysis folder and label for experiment/condition H
analysisFolder_H = base_path + r'/nr3c2/Analysis_TPI'
conditionName_H = "n3"
# Set analysis folder and label for experiment/condition i
analysisFolder_I = base_path + r'/Sp4/Analysis_TPI'
conditionName_I = "Sp4"
# Set analysis folder and label for experiment/condition L
analysisFolder_L = base_path + r'/trio/Analysis_TPI'
conditionName_L = "Trio"
# Set analysis folder and label for experiment/condition M
analysisFolder_M = base_path + r'/xpo7/Analysis_TPI'
conditionName_M = "X7"


# Assemble lists
analysisFolders = [analysisFolder_A, analysisFolder_B, analysisFolder_C, analysisFolder_D, analysisFolder_E, analysisFolder_F, analysisFolder_G, analysisFolder_H, analysisFolder_I, analysisFolder_L, analysisFolder_M]

# Createa list of conditions 
conditionNames = [conditionName_A, conditionName_B, conditionName_C, conditionName_D, conditionName_E, conditionName_F, conditionName_G, conditionName_H, conditionName_I, conditionName_L, conditionName_M]

# Summary Containers
VPI_NS_BINS_summary = [] 
VPI_S_BINS_summary = []
# SPI_NS_BINS_summary = [] 
# SPI_S_BINS_summary = []

# Go through each analysis folder (analysis folder)
for i, analysisFolder in enumerate(analysisFolders):
    
    # Find all the npz files saved for each group and fish with all the information
    npzFiles = glob.glob(analysisFolder+'/*.npz')
    
    # Calculate how many files
    numFiles = np.size(npzFiles, 0)

    # Allocate space for summary data       
    VPI_NS_BINS_ALL = np.zeros((numFiles, 15)) 
    VPI_S_BINS_ALL = np.zeros((numFiles, 15))
    # SPI_NS_BINS_ALL = np.zeros((numFiles, 15)) 
    # SPI_S_BINS_ALL = np.zeros((numFiles, 15))

    
    # Go through all the files contained in the analysis folder
    for f, filename in enumerate(npzFiles):
    
        # Load each npz file
        dataobject = np.load(filename)
        
        # Extract from the npz file
        VPI_NS_BINS= dataobject['VPI_NS_BINS'] 
        VPI_S_BINS= dataobject['VPI_S_BINS'] 
        # SPI_NS_BINS= dataobject['SPI_NS_BINS'] 
        # SPI_S_BINS= dataobject['SPI_S_BINS'] 
    
        # Make an array with all npz values of each fish
        VPI_NS_BINS_ALL[f,:] =  VPI_NS_BINS
        VPI_S_BINS_ALL[f,:] = VPI_S_BINS
        # SPI_NS_BINS_ALL[f,:] =  SPI_NS_BINS
        # SPI_S_BINS_ALL[f,:] = SPI_S_BINS

    
    # Add to summary lists
  
    VPI_NS_BINS_summary.append(VPI_NS_BINS_ALL) 
    VPI_S_BINS_summary.append(VPI_S_BINS_ALL)
    # SPI_NS_BINS_summary.append(SPI_NS_BINS_ALL) 
    # SPI_S_BINS_summary.append(SPI_S_BINS_ALL)


# #------------------------
# Figure1 - PLOT all VPI BINS in one graph 

fig = plt.figure(figsize=(4, 4))

# Use a distinct Seaborn color palette
num_conditions = len(conditionNames)

# Give a specific seaborn palette to all conditions apart the w/t (-1)
# The second enty of the color_palette is hte number of colours you need e.g. sns.color_palette("husl", 8)
palette = sns.color_palette("tab20", num_conditions - 1)  

# Define black as colour for the w/t
colors = ["black"] + list(palette)

for condition_idx, (condition_name, VPI_S_BINS_condition) in enumerate(zip(conditionNames, VPI_S_BINS_summary)):
    # We convert inside the "for loop" the single VPI_S_BINS_condition (idx) into a np.array so we can use numpy properties
    # We convert each idx VPI_S_BINS_Summary inside the loop because the whole VPI_S_BINS_Summary may not be convertable to np.array due to different sizes of its idx arrays
    # BUT its single arrays coudl be made np.arrays
    VPI_S_BINS_condition = np.array(VPI_S_BINS_condition)
    #print(f"VPI_S_BINS_condition: {VPI_S_BINS_condition}")
    
    # Calculate mean and standard deviation across fish for the current condition
    mean_VPI = np.mean(VPI_S_BINS_condition, axis=0)
    #print(f"Mean: {mean_VPI}")
    std_dev_VPI = np.std(VPI_S_BINS_condition, axis=0)
    
    # Plot mean with error bands one at a time as calculated
    plt.plot(
        range(1, 16), mean_VPI, label=f"{condition_name} (mean)", color=colors[condition_idx], linewidth=3
    )
    plt.fill_between(
        range(1, 16), mean_VPI - std_dev_VPI, mean_VPI + std_dev_VPI, color=colors[condition_idx], alpha=0.01
    )

# Customize the plot
plt.title("VPI_S_BINS_ALL Time Series: Means and Standard Deviations", fontsize=14)
plt.xlabel("Time (Minutes)", fontsize=12)
plt.ylabel("VPI", fontsize=12)
plt.xticks(range(1, 16))  # Set x-axis ticks to match the bins (1-15)
plt.grid(alpha=0.3)
#plt.legend(fontsize=6, title="Conditions", loc="upper right")
plt.legend(fontsize=6, loc="upper right")
plt.tight_layout()

# Add legend outside the graph
plt.legend(
    loc="center left",          # Anchor the legend on the left center of the plot
    bbox_to_anchor=(0, 0.1),    # Position it fully outside the right edge
    borderaxespad=0,            # Padding between the axes and the legend
    fontsize=6,
    #title="Conditions"          # Optional title for the legend
)

# Save the plot
analysisFolder_figures = base_path + r'/Final_Figures/VPI_BINS'
filename = analysisFolder_figures + '/VPI_S_BINS_ALL_Mean_Std.png'
plt.savefig(filename, dpi=300)
#plt.show()


# ------------------------
# Figure2 - PLOT all VPI BINS in different graph to compare to w/t

fig, axes = plt.subplots(4, 3, figsize=(8, 8))  # Create a grid of 3x3 plots
axes = axes.flatten()  # Flatten the 3x3 grid into a 1D array for easier access

# Define the first condition (w/t) data
w_t_condition = np.array(VPI_S_BINS_summary[0])  # Always the first condition
mean_w_t = np.mean(w_t_condition, axis=0)
std_w_t = np.std(w_t_condition, axis=0)

# Number of conditions to plot
num_conditions = len(conditionNames) - 1  # Skip the first condition ("w/t")

# Iterate through the remaining 9 conditions
for idx in range(num_conditions):
    condition_name = conditionNames[idx + 1]
    VPI_S_BINS_condition = np.array(VPI_S_BINS_summary[idx + 1])

    # Get the current axis
    ax = axes[idx]

    # Calculate mean and std for the current condition
    mean_condition = np.mean(VPI_S_BINS_condition, axis=0)
    std_condition = np.std(VPI_S_BINS_condition, axis=0)

    # Plot w/t (black line)
    ax.plot(range(1, 16), mean_w_t, label="w/t (mean)", color="black", linewidth=3)
    ax.fill_between(
        range(1, 16),
        mean_w_t - std_w_t,
        mean_w_t + std_w_t,
        color="black",
        alpha=0.2,
    )

    # Plot the current condition (blue line)
    ax.plot(
        range(1, 16),
        mean_condition,
        label=f"{condition_name} (mean)",
        color="tab:blue",
        linewidth=3,
    )
    ax.fill_between(
        range(1, 16),
        mean_condition - std_condition,
        mean_condition + std_condition,
        color="tab:blue",
        alpha=0.2,
    )

    # Customize the plot
    ax.set_title(f"w/t vs {condition_name}", fontsize=12)
    #ax.set_xlabel("Time (Minutes)")
    #ax.set_ylabel("VPI")
    ax.set_xticks(range(1, 16))
    ax.grid(alpha=0.3)
    
    #  # Add a smaller legend at the bottom
    # ax.legend(
    #     fontsize=8,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.2),  # Place the legend below the plot
    #     ncol=2,  # Arrange legend items in two columns
    # )


# Hide unused subplots (if any)
for idx in range(num_conditions, len(axes)):
    axes[idx].axis("off")

# Adjust layout and
plt.tight_layout()


analysisFolder_figures = base_path + r'/Final_Figures/VPI_BINS'
filename = analysisFolder_figures + '/VPI_S_BINS_Comparison.png'
plt.savefig(filename, dpi=300)
#plt.show()


