subset="02-005 02-008"
subset_elements=($subset)
for subset in "${subset_elements[@]}"
do
python -W ignore infinite_generator_3D.py \
--fold $subset \
--scale 32 \
--data /storage_bizon/bizon_imagedata/naravich/longitudinal_view/ \
--save generated_cubes
done