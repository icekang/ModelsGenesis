#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2023a-pytorch

# Activating the conda environment
source activate genesis
conda activate genesis
which python

# Setup env variables nn_UNet
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"

# Run the script
python pytorch/Genesis_Finetune.py --config pytorch/configs/fine_tune_config-fold2.yaml