from pathlib import Path
import numpy as np
from typing import List
import cv2
import sys
import nibabel as nib

unlabeled_data_dir = Path('/home/gridsan/nchutisilp/datasets/Unlabeled_OCT_by_CADx/Unlabeled_OCT_by_CADx/') # Yiqing filtered data
output_nifti = unlabeled_data_dir / 'NiFTI'
output_mp4 = unlabeled_data_dir / 'MP4'
output_nifti.mkdir(exist_ok=True)
output_mp4.mkdir(exist_ok=True)

def resolve_image_path(subject_id: str, cadx: str):
    if subject_id.startswith('00'):
        subject_id = subject_id[1:]
    image_dir = unlabeled_data_dir / cadx
    post_stent_image_path = image_dir.glob("{}Final".format(subject_id.replace('-', '')))
    post_ivl_image_path = image_dir.glob("{}Post".format(subject_id.replace('-', '')))
    pre_ivl_image_path = image_dir.glob("{}Pre".format(subject_id.replace('-', '')))

    post_stent_image_path = list(post_stent_image_path) 
    post_ivl_image_path = list(post_ivl_image_path)
    pre_ivl_image_path = list(pre_ivl_image_path)

    return {
        'Pre': pre_ivl_image_path[0].name if pre_ivl_image_path else None,
        'Post': post_ivl_image_path[0].name if post_ivl_image_path else None,
        'Final': post_stent_image_path[0].name if post_stent_image_path else None,
    }

def get_sorted_patient_images(patient_dir: Path, verbose=True):
    images = list(patient_dir.glob('*.png'))
    if verbose and len(images) == 0:
        print(f'No images in {patient_dir}')
    images.sort(key = lambda x: int(x.stem.split('_')[-1]))
    return images

def read_an_image(image_absolute_path: Path):
    image = cv2.imread(str(image_absolute_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def min_max_normalize_image(image: np.array):
    assert image.max() <= 127, f'Image has max value {image.max()} > 127'
    assert 0 <= image.min(), f'Image has min value {image.max()} < 0'
    image = image.astype(np.float32) * 255.0 / 127.0
    image = image.astype(np.uint8)
    return image

def read_images(image_absolute_paths: List[Path]):
    images = [read_an_image(image_absolute_path) for image_absolute_path in image_absolute_paths]
    images = [min_max_normalize_image(image) for image in images]
    return images

def save_image_to_mp4(images: List[np.array], cadx: str, subject_id: str):
    final_output_dir = output_mp4 / cadx
    final_output_dir.mkdir(exist_ok=True)
    video_name = final_output_dir / f'{subject_id}.mp4'

    fps = 1
    sample_image = images[0]
    video = cv2.VideoWriter(str(video_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, sample_image.shape[:2], isColor=True)
    for image in images:
        if image.shape != sample_image.shape:
            raise ValueError(f'Image has shape {image.shape}, expected {sample_image.shape}')
        video.write(image)
    video.release()

def save_image_to_nifit(images: List[np.array], cadx: str, subject_id: str):
    final_output_dir = output_nifti / cadx
    final_output_dir.mkdir(exist_ok=True)
    nifti_name = final_output_dir / f'{subject_id}.nii.gz'

    for image in images:
        assert np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 1] == image[:, :, 2]), 'All channels must have the same value'
    data = np.stack([image[:, :, 0] for image in images], axis=-1)

    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, nifti_name)

# Grab the arguments that are passed in
my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])


# Walk through the unlabeled data directory and get the image paths
jobs = []
for cadx in unlabeled_data_dir.iterdir():
    if cadx.is_dir() and cadx.name.startswith("CAD"):
        for subject_id in cadx.iterdir():
            if subject_id.is_dir():
                if 'Dis' in subject_id.name or 'Prox' in subject_id.name:
                    print('Skipping', subject_id.name)
                    continue
                sorted_patient_images = get_sorted_patient_images(subject_id, verbose=False)
                if len(sorted_patient_images) == 0:
                    print('Skipping', subject_id.name)
                    continue
                jobs.append((cadx, subject_id))
jobs.sort()
my_jobs = jobs[my_task_id:len(jobs):num_tasks]
print(f'Creating {len(my_jobs)} folders into NIfTI and MP4 (task_id {my_task_id}; num_tasks: {num_tasks})')

for cadx, subject_id in my_jobs:
    sorted_patient_images = get_sorted_patient_images(subject_id, verbose=False)
    images = read_images(sorted_patient_images)
    cadx_name = cadx.name
    subject_id_name = subject_id.name
    save_image_to_mp4(images, cadx_name, subject_id_name)
    save_image_to_nifit(images, cadx_name, subject_id_name)
    print(cadx.name, subject_id.name, len(sorted_patient_images))
