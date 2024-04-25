#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2023b

# Activating the conda environment
source activate genesis

# Run the script
python pytorch/Genesis_Finetune.py --config pytorch/configs/fine_tune_config.yaml