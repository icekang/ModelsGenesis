#!/bin/bash

source activate genesis

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

python -u convert_nifti_to_video.py $LLSUB_RANK $LLSUB_SIZE