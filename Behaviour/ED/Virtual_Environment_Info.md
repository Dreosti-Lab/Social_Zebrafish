


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