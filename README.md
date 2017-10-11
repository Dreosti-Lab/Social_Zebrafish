# Social_Zebrafish
A set of Python analysis scripts for analysing social behaviour of zebrafish

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