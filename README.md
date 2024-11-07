# Social_Zebrafish
A set of Python analysis scripts for analysing social behaviour of zebrafish

# Create and activate the virtual environment
1. Create a temporary folder _tmp (use underscore so that it appearsa at the top)
2. Type in the terminal python3 -m venv SZ (for Social zebrafish). This will create the folder
3. Activate virtual environmnent. You run the activate script contained in the "bin" folder.


```bash
# Before you run the comands make sure you are in the correct folder "Social_zebrafish" repository
mkdir _tmp
cd _tmp
python3 -m venv SZ
source ./SZ/bin/activate
```

# Install packages

```bash
pip install numpy matplotlib scipy opencv-python seaborn

```
## Environment File
You must create a ".env" file in the root directory of the Repository with equivalent content to the following, obviously with **paths** that make sense for your computer:
```txt
# Location of Python libraries
LIBS_PATH="/Users/elenadreo/Repos/Dreosti-Lab/social_Zebrafish/libs"
# Location of data
BASE_PATH="/Volumes/DreostiLab1/"
```

# Run code
for example 

```bash
cd ..
python3 Behaviour/Step2_analysis.py
```

## Analysis Steps
Pre-Analysis:
1. Produce background and projection images : Stored in same data folder
2. Re-run Tracking (Create Tracking#.npz for each #fish - Contains X,Y area, etc. values relative to ROI) : Stored in same data folder

Tracking#.npz

Analysis:
3. Use the same FolderList File to make a figure and do some analysis for each fish:
	- #_#.PNG (Figure) and #_#.npz (Measurements) - The .npz contains just SPI non-social and social

Plotting:
4. Plot SPI summary (go through all the #-#.npz files)

Statistics:
5. Select Data Folder to compare (social vs. non-social conditions)

