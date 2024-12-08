# -*- coding: utf-8 -*-
"""
Compare summaries of analyzed social preference experiments

@author: Elena Dreosti
"""

## This script plots the VPI and SPI (1/4 of the Y chamber width)
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
from matplotlib.ticker import FixedLocator

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
VPI_NS_summary = []
VPI_S_summary = []
SPI_NS_summary = []
SPI_S_summary = []   
VPI_NS_BINS_summary = [] 
VPI_S_BINS_summary = []

# Go through each analysis folder (analysis folder)
for i, analysisFolder in enumerate(analysisFolders):
    
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
    
    # Go through all the files contained in the analysis folder
    for f, filename in enumerate(npzFiles):
    
        # Load each npz file
        dataobject = np.load(filename)
        
        # Extract from the npz file
        VPI_NS = dataobject['VPI_NS']    
        VPI_S = dataobject['VPI_S'] 
        SPI_NS = dataobject['SPI_NS']    
        SPI_S = dataobject['SPI_S'] 
        VPI_NS_BINS= dataobject['VPI_NS_BINS'] 
        VPI_S_BINS= dataobject['VPI_S_BINS'] 
    
        # Make an array with all npz values of each fish
        VPI_NS_ALL[f] = VPI_NS
        VPI_S_ALL[f] = VPI_S
        SPI_NS_ALL[f] = SPI_NS
        SPI_S_ALL[f] = SPI_S 
        VPI_NS_BINS_ALL[f,:] =  VPI_NS_BINS
        VPI_S_BINS_ALL[f,:] = VPI_S_BINS

    
    # Add to summary lists
    VPI_NS_summary.append(VPI_NS_ALL)
    VPI_S_summary.append(VPI_S_ALL)  
    SPI_NS_summary.append(SPI_NS_ALL)
    SPI_S_summary.append(SPI_S_ALL)     
    VPI_NS_BINS_summary.append(VPI_NS_BINS_ALL) 
    VPI_S_BINS_summary.append(VPI_S_BINS_ALL)


# #------------------------
# # FIGURE1 SPI VIOLIN PLOT

# SPI
fig = plt.figure(figsize=(8, 6))

fig.suptitle("SOCIAL SIDE", fontsize=10)
plt.title('SPI')

series_list = []


for i, name in enumerate(conditionNames):
    s = pd.Series(SPI_S_summary[i], name="S: " + name)
    series_list.append(s)
df = pd.concat(series_list, axis=1)


# # prepare data for violin plot.
# df = pd.concat(series_list, axis=1)
df_long = df.melt(var_name="Condition", value_name="Value")

#sns.violinplot(x="Condition", y="Value", data=df_long, inner="quartile", palette="Set3", cut=0)
plt.xticks(rotation=45)
plt.tight_layout()



colors = ['#d9d9d9','#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#ccebc5', '#ffed6f']

# Create a violin plot of VPI 

sns.violinplot(x="Condition", y="Value", data=df_long, inner=None, palette=colors,hue="Condition", alpha=0.8, cut=0, linewidth=1, legend=False )
sns.stripplot(x="Condition", y="Value", data=df_long, color="black", alpha=0.3, jitter=True)
plt.xticks(rotation=45)

# Add median lines manually, ensuring they fit within the violin
for i, condition in enumerate(df_long["Condition"].unique()):
    median = df_long[df_long["Condition"] == condition]["Value"].median()
    
    # Dynamically calculate the width of the violin for scaling the median line
    violin_width = 0.3  # Adjust this value as needed for better alignment
    plt.plot(
        [i - violin_width, i + violin_width],  # X-range for the median line (centered within the violin)
        [median, median],  # Y-range for the median line
        color="black", 
        linewidth=4  # Thickness of the median line
    )

plt.tight_layout

#plt.show(block=False)

analysisFolder_figures = base_path + r'/Final_Figures/SPI'

filename = analysisFolder_figures + '/SPI_ViolinPlot.png'
plt.savefig(filename, dpi=600)

# ###############################################

# FIGURE 2 ===  SPI BOX PLOT

fig = plt.figure(figsize=(8, 6))
#fig = plt.figure(figsize=(8, 6))
#fig, ax = plt.subplots(2,3,figsize=(10, 4))
#plt.subplot(2,3,1)
plt.title("SPI")

series_list = []

# Assemble the data into a series list
for i, name in enumerate(conditionNames):
    s = pd.Series(SPI_S_summary[i], name="S: " + name)
    series_list.append(s)


# Combine series into a DataFrame
df = pd.concat(series_list, axis=1)

# Melt the DataFrame for compatibility with Seaborn
df_long = df.melt(var_name="Condition", value_name="Value")

# Define custom colors: first column gray, others custom hex colors
box_colours_rgba3 = ['#d9d9d9','#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#ccebc5', '#ffed6f']


# Create a boxplot
sns.boxplot(
    x="Condition",  # Set the x-axis to the conditions
    y="Value",      # Set the y-axis to the values
    data=df_long,   # Use the melted long DataFrame
    palette=box_colours_rgba3, # Choose a color palette (optional)
    #palette="Set3", # Choose a color palette (optional)
    linewidth=1,    # Thickness of boxplot lines
    #showmeans=True, # Optionally show mean markers 
    hue="Condition",
    meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black", "markersize": 3}, # Customize mean marker
    flierprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black", "markersize": 3}
)


# Overlay individual data points
sns.stripplot(
    x="Condition",  # Align dots with conditions
    y="Value",      # Align dots with values
    data=df_long,   # Use the melted DataFrame
    color="#404040",  # Set dot color
    #color=box_colours_rgba2,  # Set dot color
    alpha=0.7,      # Adjust dot transparency
    jitter=True,    # Add jitter to avoid overlap of dots
    size=5        # Adjust dot size
)

# Add titles and labels if needed

plt.xticks(rotation=20)  # Rotate x-axis labels for better readability
plt.tight_layout()

# set3_palette = sns.color_palette("Set3", 12)

# # Convert the colors to hex format
# set3_hex = set3_palette.as_hex()

# Display the hex values
#print(set3_hex)

plt.xlabel('')
# Display the plot
#plt.show(block=False)

analysisFolder_figures = base_path + r'/Final_Figures/SPI'

filename = analysisFolder_figures + '/SPI_BoxPlot.png'
plt.savefig(filename, dpi=600)
#plt.show()


# ###############################################
# Figure 3 SPI SWARM PLOT

series_list = []

# Assemble the data into a series list
for i, name in enumerate(conditionNames):
    s = pd.Series(SPI_S_summary[i], name="S: " + name)
    series_list.append(s)


# Combine series into a DataFrame
df = pd.concat(series_list, axis=1)

# Melt the DataFrame for compatibility with Seaborn
df_long = df.melt(var_name="Condition", value_name="Value")


# Define colors for each condition
colors = ['#989898', '#63948b', '#b2b27d', '#858299', '#b05a50', 
 '#5a7c94', '#b17e45', '#7d9b4a', '#b090a0', '#8fa48a', '#b2a64e']



# Start the plot
fig = plt.figure(figsize=(10, 6))
plt.title("SPI with Mean and SD", fontsize=14)

# Ensure the conditions are plotted in the correct order
condition_order = conditionNames

# Strip "S: " prefix from Condition column if needed
df_long["Condition"] = df_long["Condition"].str.replace("S: ", "", regex=False)

# Ensure Condition column is categorical and matches condition_order
df_long["Condition"] = pd.Categorical(df_long["Condition"], categories=condition_order, ordered=True)



# Swarmplot with custom colors
sns.swarmplot(
    x="Condition", 
    y="Value", 
    hue="Condition",
    data=df_long, 
    palette=colors[:len(df_long["Condition"].unique())], 
    alpha=1, 
    size=5,
    order=condition_order, 
)

# Calculate mean and standard deviation for each condition
means = df_long.groupby("Condition")["Value"].mean()
stds = df_long.groupby("Condition")["Value"].std()
ses = df_long.groupby("Condition")["Value"].sem()  # Calculate standard error


# Overlay mean and standard deviation lines
for i, condition in enumerate(means.index):
    mean = means[condition]
    se = ses[condition]

    # Plot vertical line connecting mean to standard deviation
    plt.vlines(
        x=i, ymin=mean - se, ymax=mean + se, colors="gray",linewidth=1.5, linestyle="-"
    )
    
    # Plot mean as a thicker horizontal line
    plt.hlines(
        y=mean, xmin=i - 0.4, xmax=i + 0.4, colors="black", linewidth=5
    )

    # Plot standard deviation as thinner horizontal lines
    plt.hlines(
        y=mean - se, xmin=i - 0.2, xmax=i + 0.2, colors="black", linewidth=2.5, linestyle="-"
    )
    plt.hlines(
        y=mean + se, xmin=i - 0.2, xmax=i + 0.2, colors="black", linewidth=2.5, linestyle="-"
    )

# Customize x-axis labels
plt.xticks(ticks=range(len(means)), labels=means.index, rotation=20)

# Add grid for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout
plt.tight_layout()

# Display the plot
#plt.show()
analysisFolder_figures = base_path + r'/Final_Figures/SPI'

filename = analysisFolder_figures + '/SPI_SwarmPlot.png'
plt.savefig(filename, dpi=600)



######################################

# #------------------------
# # FIGURE4 VPI VIOLIN PLOT

# SPI
fig = plt.figure(figsize=(8, 6))

fig.suptitle("SOCIAL SIDE", fontsize=10)
plt.title('VPI')

series_list = []


for i, name in enumerate(conditionNames):
    s = pd.Series(VPI_S_summary[i], name="S: " + name)
    series_list.append(s)
df = pd.concat(series_list, axis=1)


# # prepare data for violin plot.
# df = pd.concat(series_list, axis=1)
df_long = df.melt(var_name="Condition", value_name="Value")

#sns.violinplot(x="Condition", y="Value", data=df_long, inner="quartile", palette="Set3", cut=0)
plt.xticks(rotation=45)
plt.tight_layout()


# Define custom colors: gray for all mutants, distinct color (e.g., blue) for control
#colors = ["gray"] * (len(conditionNames) - 1) + ["blue"] # to have the last colour blue
#colors = ["blue"] + ["gray"] * (len(conditionNames) - 1)  # To get first colour blue

# # Or Define a dictionary mapping each condition to a custom color 
# colors = {
#     "w/t": "#1f77b4",  # Example: blue
#     "A11": "#ff7f0e",  # Example: orange
#     "c1g": "#2ca02c",  # Example: green
#     "G3": "#d62728",   # Example: red
#     "G2a": "#9467bd",  # Example: purple
#     # Add as many as needed
# }


colors = ['#d9d9d9','#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#ccebc5', '#ffed6f']

if len(colors) != len(df_long["Condition"].unique()):
    print("Palette length does not match the number of unique conditions!")


# Create a violin plot of VPI 

sns.violinplot(
    x="Condition", 
    y="Value", 
    data=df_long, 
    hue="Condition", 
    inner=None, 
    palette=colors,
    alpha=0.8, 
    cut=0, 
    linewidth=1, 
    legend=False )

colors = ['#d9d9d9','#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#ccebc5', '#ffed6f']


sns.stripplot(
    x="Condition", 
    y="Value", 
    data=df_long, 
    hue="Condition", 
    palette='dark:#404040', 
    alpha=0.3, 
    jitter=True)
plt.xticks(rotation=45)

# Add median lines manually, ensuring they fit within the violin
for i, condition in enumerate(df_long["Condition"].unique()):
    median = df_long[df_long["Condition"] == condition]["Value"].median()
    
    # Dynamically calculate the width of the violin for scaling the median line
    violin_width = 0.3  # Adjust this value as needed for better alignment
    plt.plot(
        [i - violin_width, i + violin_width],  # X-range for the median line (centered within the violin)
        [median, median],  # Y-range for the median line
        color="black", 
        linewidth=4  # Thickness of the median line
    )

plt.tight_layout

#plt.show(block=False)

analysisFolder_figures = base_path + r'/Final_Figures/VPI'

filename = analysisFolder_figures + '/VPI_ViolinPlot.png'
plt.savefig(filename, dpi=600)

# ###############################################

# FIGURE 5 ===  VPI BOX PLOT

fig = plt.figure(figsize=(8, 6))
#fig = plt.figure(figsize=(8, 6))
#fig, ax = plt.subplots(2,3,figsize=(10, 4))
#plt.subplot(2,3,1)
plt.title("VPI")

series_list = []

# Assemble the data into a series list
for i, name in enumerate(conditionNames):
    s = pd.Series(VPI_S_summary[i], name="S: " + name)
    series_list.append(s)


# Combine series into a DataFrame
df = pd.concat(series_list, axis=1)

# Melt the DataFrame for compatibility with Seaborn
df_long = df.melt(var_name="Condition", value_name="Value")

# Define custom colors: first column gray, others custom hex colors
# box_colours = ["#808080"] + ["#FF5733", "#33FF57", "#3357FF", "#F1C40F", "#9B59B6", "#1ABC9C", "#E74C3C", "#34495E", "#2ECC71"]
# #To add alpha transparency
# box_colours_rgba = ["#D9D9D9"] + ["#FFCCC2", "#C2FFC2", "#C2CCFF", "#FBEEC6", "#E1CCE9", "#BAEBCD", "#F8C9C4", "#C2C9CF", "#C0F0D4"]
# box_colours_rgba2 = ['#d9d9d9','#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#bc80bd', '#ccebc5', '#ffed6f']
box_colours_rgba3 = ['#d9d9d9','#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#ccebc5', '#ffed6f']


# Create a boxplot
sns.boxplot(
    x="Condition",  # Set the x-axis to the conditions
    y="Value",      # Set the y-axis to the values
    data=df_long,   # Use the melted long DataFrame
    palette=box_colours_rgba3, # Choose a color palette (optional)
    #palette="Set3", # Choose a color palette (optional)
    linewidth=1,    # Thickness of boxplot lines
    hue="Condition",
    #showmeans=True, # Optionally show mean markers 
    meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black", "markersize": 3}, # Customize mean marker
    flierprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black", "markersize": 3}
)


# Overlay individual data points
sns.stripplot(
    x="Condition",  # Align dots with conditions
    y="Value",      # Align dots with values
    data=df_long,   # Use the melted DataFrame
    color="#404040",  # Set dot color
    #color=box_colours_rgba2,  # Set dot color
    alpha=0.7,      # Adjust dot transparency
    jitter=True,    # Add jitter to avoid overlap of dots
    size=5         # Adjust dot size
)

# Add titles and labels if needed

plt.xticks(rotation=20)  # Rotate x-axis labels for better readability
plt.tight_layout()

# set3_palette = sns.color_palette("Set3", 12)

# # Convert the colors to hex format
# set3_hex = set3_palette.as_hex()

# Display the hex values
#print(set3_hex)

plt.xlabel('')
# Display the plot
#plt.show(block=False)

analysisFolder_figures = base_path + r'/Final_Figures/VPI'

filename = analysisFolder_figures + '/VPI_BoxPlot.png'
plt.savefig(filename, dpi=600)
#plt.show()

######################################################

# Figure 6 - VPI SWARM PLOT

# Define colors for each condition
colors = ['#989898', '#63948b', '#b2b27d', '#858299', '#b05a50', 
 '#5a7c94', '#b17e45', '#7d9b4a', '#b090a0', '#8fa48a', '#b2a64e']



# Start the plot
fig = plt.figure(figsize=(10, 6))
plt.title("VPI with Mean and SD", fontsize=14)

# Strip "S: " prefix from Condition column if needed
df_long["Condition"] = df_long["Condition"].str.replace("S: ", "", regex=False)

# Ensure Condition column is categorical and matches condition_order
df_long["Condition"] = pd.Categorical(df_long["Condition"], categories=condition_order, ordered=True)


condition_order = conditionNames

# Swarmplot with custom colors
sns.swarmplot(
    x="Condition", 
    y="Value", 
    data=df_long, 
    hue="Condition",
    palette=colors[:len(df_long["Condition"].unique())], 
    alpha=1, 
    size=5,
    order=condition_order, 
)

# Calculate mean and standard deviation for each condition
means = df_long.groupby("Condition")["Value"].mean()
stds = df_long.groupby("Condition")["Value"].std()

# Overlay mean and standard deviation lines
for i, condition in enumerate(means.index):
    mean = means[condition]
    #std = stds[condition]
    se = ses[condition]

    # Plot vertical line connecting mean to standard deviation
    plt.vlines(
        x=i, ymin=mean - se, ymax=mean + se, colors="gray", linewidth=1.5, linestyle="-"
    )
    
    # Plot mean as a thicker horizontal line
    plt.hlines(
        y=mean, xmin=i - 0.4, xmax=i + 0.4, colors="black", linewidth=5
    )

    # Plot standard deviation as thinner horizontal lines
    plt.hlines(
        y=mean - se, xmin=i - 0.2, xmax=i + 0.2, colors="black", linewidth=2.5, linestyle="-"
    )
    plt.hlines(
        y=mean + se, xmin=i - 0.2, xmax=i + 0.2, colors="black", linewidth=2.5, linestyle="-"
    )

# Customize x-axis labels
plt.xticks(ticks=range(len(means)), labels=means.index, rotation=20)

# Add grid for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout
plt.tight_layout()

# Display the plot
#plt.show()
analysisFolder_figures = base_path + r'/Final_Figures/VPI'

filename = analysisFolder_figures + '/VPI_SwarmPlot.png'
plt.savefig(filename, dpi=600)


###############

# Figure 4 - Summary plots
plt.figure()
plt.title('VPI')

series_list = []
    
for i, name in enumerate(conditionNames):
    s = pd.Series(VPI_S_summary[i], name="S: " + name)
    series_list.append(s)
df = pd.concat(series_list, axis=1)

colors = ['#333333', '#333333', '#333333' , '#333333', '#333333' , '#333333', '#333333', '#333333', '#333333', '#333333', '#333333']

palette = ['#989898', '#63948b', '#b2b27d', '#858299', '#b05a50', '#5a7c94', '#b17e45', '#7d9b4a', '#b090a0', '#8fa48a', '#b2a64e']
#box_colours_rgba3 = ['#d9d9d9','#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#ccebc5', '#ffed6f']


#palette = sns.color_palette(palette='tab20')
ax=sns.pointplot(data=df, marker=".", markersize=18, orient="v", errorbar=("se", 1),capsize=0.2, err_kws={'linewidth': 2.5}, linestyle='none', palette=palette, linewidth=1)
ax.spines[['right', 'top']].set_visible(False)
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.tick_params(width=2)

# Fix x-tick labels
ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize = 14, fontname = 'Arial', fontweight = 'bold')

# Fix y-tick labels
yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels(ax.get_yticklabels(),fontsize = 12, fontname = 'Arial', fontweight = 'bold')
#ax=sns.stripplot(data=df, orient="v", size=5, jitter=True, edgecolor="none", alpha=1, palette=colors, zorder=0,  linewidth=2, legend=False)
ax.set(ylim=(-0.3, 0.6))
#plt.show()

###############

# Figure 5 - Summary plots
plt.figure()
plt.title('SPI')

series_list = []
    
for i, name in enumerate(conditionNames):
    s = pd.Series(SPI_S_summary[i], name="S: " + name)
    series_list.append(s)
df = pd.concat(series_list, axis=1)

colors = ['#333333', '#333333', '#333333' , '#333333', '#333333' , '#333333', '#333333', '#333333', '#333333', '#333333', '#333333']

palette = ['#989898', '#63948b', '#b2b27d', '#858299', '#b05a50', '#5a7c94', '#b17e45', '#7d9b4a', '#b090a0', '#8fa48a', '#b2a64e']
#box_colours_rgba3 = ['#d9d9d9','#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#ccebc5', '#ffed6f']


#palette = sns.color_palette(palette='tab20')
ax=sns.pointplot(data=df, marker=".", markersize=18, orient="v", errorbar=("se", 1),capsize=0.2, err_kws={'linewidth': 2.5}, linestyle='none', palette=palette, linewidth=1)
ax.spines[['right', 'top']].set_visible(False)
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.tick_params(width=2)

# Fix x-tick labels
ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize = 14, fontname = 'Arial', fontweight = 'bold')

# Fix y-tick labels
yticks = ax.get_yticks()
ax.set_yticks(yticks)
ax.set_yticklabels(ax.get_yticklabels(),fontsize = 12, fontname = 'Arial', fontweight = 'bold')
#ax=sns.stripplot(data=df, orient="v", size=5, jitter=True, edgecolor="none", alpha=1, palette=colors, zorder=0,  linewidth=2, legend=False)
ax.set(ylim=(0, 0.6))
plt.show()