make sure to install git lfs before cloning weights (brew install git-lfs)

create huggingface account and use personal access token with read perms to clone weights

use python 3.10
requirements.txt is not working, you should install each of the packages manually

decord doesn't work for M4 macs, use eva-decord instead

# Check the installed version (e.g., gcc-13)
brew list gcc 

# Install xformers using the specific GCC version
CC=gcc-[version] CXX=g++-[version] pip install xformers
# Example: CC=gcc-13 CXX=g++-13 pip install xformers

## Inference ijob
ijob -A shrew-crew -p gpu --gres=gpu:a100:1 --mem=48G -c 8 -t 4:00:00

## Run stereocrafter
module load apptainer
apptainer exec --nv --env MAX_DISP=3.80 stereocrafter.sif sh run_inference.sh 

## Download results
rsync -avz --progress jjq7qj@login.hpc.virginia.edu:/scratch/jjq7qj/stereocrafter/outputs/ ~/Downloads/stereocrafter-outputs/
