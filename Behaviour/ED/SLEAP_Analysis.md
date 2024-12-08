# SLEAP_Analysis
SLEAP Analysis to track stimulus cues when more than 1 fish 

# The FIRST TIME you run SLEAP

# Create and activate the virtual environment
1. Create a temporary folder _tmp (use underscore so that it appearsa at the top)
2. Type in the terminal python3 -m venv SLEAP (for SLEAP). This will create the folder
3. Activate virtual environmnent. You run the activate script contained in the "bin" folder.


```bash
# Before you run the comands make sure you are in the correct folder "Social_zebrafish" repository
mkdir _tmp
cd _tmp
python3 -m venv SLEAP
source ./SLEAP/bin/activate
```

# Run Sleap 
```bash
# If you run this command you shoudl see a window that opens. It may take a while
sleap-label
```

# Install otehr packages
- You may get many errors saying that SLEAP can't find certain libraries. 
- You need to install all missing library

```bash
# For eaxample...
pip install ...

```
## SECOND TIME you run SLEAP

1. Be in the folder where you created the environment _tmp 

2. Activate the environment:

```bash
source ./SLEAP/bin/activate
```
3. Run sleap

```bash
sleap-label
```