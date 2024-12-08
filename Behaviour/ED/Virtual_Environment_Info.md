


# Create and activate the virtual environment first time
The first time you are creating teh environment:
1. Create a temporary folder _tmp (use underscore so that it appearsa at the top). And go inside 
that folder. 

- check if you are in the folder where you want to create your environment. 
- mkdir _tmp
- cd _tmp

2. Create the virtual environment inside tmp. You can name it as you want to. Ours is "SZ" that stands 
for Social zebrafish. This command will create a virtual environment folder that contains already 
some useful files like the "activate " script. 

python3 -m venv SZ

3. To activate the virtual environmnent, you need to run the "activate" script that is contained 
in the "bin" folder. This means that first you need to add a path SZ(name of the environment)/bin/ 
as a path. Or you coudl aslo navigate in the "bin" folder and type  "source activate"

- source ./SZ/bin/activate

# Run the virtual environment the subsequent times.

1. Be in the folder where you created the environment _tmp 

2. Activate the environment:
- source ./SZ/bin/activate

source ../../_tmp/SZ/bin/activate


# Installation instructions SLEAP

pip install "sleap[pypi]"
sleap-label

# Installation instructions DLC
1. Insatll Homebrew, which you need for H
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"


2.  Run these commands in your terminal to add Homebrew to your PATH:
    echo >> /Users/elenadreo/.zprofile
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/elenadreo/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"\

3. brew install hdf5 

4. brew install c-blosc lzo 

brew install lzo 
  export LDFLAGS="-L/opt/homebrew/opt/lzo/lib"
  export CPPFLAGS="-I/opt/homebrew/opt/lzo/include"

5. brew install bzip2

    echo 'export PATH="/opt/homebrew/opt/bzip2/bin:$PATH"' >> /Users/elenadreo/.zshrc

  export LDFLAGS="-L/opt/homebrew/opt/bzip2/lib"
  export CPPFLAGS="-I/opt/homebrew/opt/bzip2/include"


5. pip install "git+https://github.com/DeepLabCut/DeepLabCut.git@pytorch_dlc#egg=deeplabcut[gui,modelzoo,wandb]"

6. python -m deeplabcut



Using miniconda

mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh





PySide6
matplotlib
oauthlib
idna
charset-normalizer
pyasn1
Keras_Applications
shiboken6














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

# Import local modules
import SZ_utilities_ED as SZU
# import SZ_macros as SZM
import SZ_video_ED as SZV