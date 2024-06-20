#!/bin/bash

source activate genesis

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

# LLsub ./convert_unlabeled_image_folders_to_nifiti_and_mp4.sh [2,24,2]
python -u convert_unlabeled_image_folders_to_nifiti_and_mp4.py $LLSUB_RANK $LLSUB_SIZE