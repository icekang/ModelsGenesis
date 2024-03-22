readarray -t train_subset_elements < datasets/Unlabeled_OCT/train.txt
readarray -t test_subset_elements < datasets/Unlabeled_OCT/test.txt
combined=("${train_subset_elements[@]}" "${test_subset_elements[@]}")
for subset in "${combined[@]}"
do
    python -W ignore infinite_generator_3D.py \
    --fold $subset \
    --scale 32 \
    --data /storage_bizon/bizon_imagedata/naravich/Unlabeled_OCT/ \
    --save /storage_bizon/naravich/Unlabeled_OCT_cubes/
done