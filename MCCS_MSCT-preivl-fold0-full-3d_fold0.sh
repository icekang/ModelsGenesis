#!/bin/bash

# Activating the conda environment
source activate genesis
which python

# Setup env variables nn_UNet
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"

# Run the script
python pytorch/nnUNet_RegressionFullRun.py --config pytorch/configs/fine_tune_config-regression-MCCS_MSCT-3d.yaml
