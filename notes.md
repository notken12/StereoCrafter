make sure to install git lfs before cloning weights (brew install git-lfs)

create huggingface account and use personal access token with read perms to clone weights

use python 3.10
requirements.txt is not working, you should install each of the packages manually

decord doesn't work for M4 macs, use evo-decord instead

# Check the installed version (e.g., gcc-13)
brew list gcc 

# Install xformers using the specific GCC version
CC=gcc-[version] CXX=g++-[version] pip install xformers
# Example: CC=gcc-13 CXX=g++-13 pip install xformers
