#!/bin/bash

# Loading the required module
# source /etc/profile
source /home/gridsan/nchutisilp/.bashrc
module load anaconda/2023a-pytorch

# Activating the conda environment
source activate genesis
which python
# Run the script
python pytorch/Genesis_Finetune.py --config pytorch/configs/fine_tune_config.yaml
