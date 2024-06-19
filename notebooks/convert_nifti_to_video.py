from pathlib import Path
import nibabel as nib
import cv2
import numpy as np
import sys

# Grab the arguments that are passed in
my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

nifti_dir = Path('/home/gridsan/nchutisilp/datasets/Unlabeled_OCT_by_CADx/NiFTI')
mp4_dir = Path('/home/gridsan/nchutisilp/datasets/Unlabeled_OCT_by_CADx/MP4')

fnames = list(nifti_dir.glob('*.nii.gz'))
fnames.sort()
my_fnames = fnames[my_task_id:len(fnames):num_tasks]
print(f'Converting {len(my_fnames)} NIfTI files to MP4')

for i, nifti_file in enumerate(my_fnames):
    image_name = nifti_file.stem.replace(".nii", "")
    video_name = mp4_dir / f'{image_name}.mp4'
    img = nib.load(nifti_file)
    img_data = img.get_fdata()
    image_shape = img_data.shape[:2]

    fps = 1
    # Convert to video
    video = cv2.VideoWriter(str(video_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, image_shape, isColor=True)
    min_val, max_val = float('inf'), float('-inf')
    value_127 = 127.0 / 255.0
    for z in range(img_data.shape[2]):
        image = img_data[:, :, z]
        image = image.astype(np.float64) * 255.0 / value_127
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        min_val = min(min_val, image.min())
        max_val = max(max_val, image.max())
        if image.shape[:2] != image_shape:
            raise ValueError(f'Image {z} has shape {image.shape[:2]}, expected {image_shape}')
        video.write(image)
    video.release()
    print(f'[{i+1}/{len(my_fnames)} / {len(fnames)}] {nifti_file} -> {video_name}')