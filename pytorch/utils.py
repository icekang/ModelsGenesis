from __future__ import print_function
from functools import partial
import math
import os
import random
import copy
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import scipy
import imageio
import string
import numpy as np
from skimage.transform import resize
import torch
from torch.utils.data import Dataset
import torchio as tio
from typing import Tuple, List, Union
from pathlib import Path
import json
import lightning as L
from torch.utils.data import DataLoader
from torchmetrics import Dice, MetricCollection, MeanSquaredError, F1Score, Accuracy, Precision, Recall, R2Score
from nnunetv2.run.run_training import get_trainer_from_args
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
import wandb.util
import pandas as pd

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[0, noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                               noise_z:noise_z+block_noise_size_z,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y, 
                                 block_noise_size_z))
        image_temp[0, noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y, 
                      noise_z:noise_z+block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y, 
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x

def image_out_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    noise_z = random.randint(3, img_deps-block_noise_size_z-3)
    x[:, 
      noise_x:noise_x+block_noise_size_x, 
      noise_y:noise_y+block_noise_size_y, 
      noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                       noise_y:noise_y+block_noise_size_y, 
                                                       noise_z:noise_z+block_noise_size_z]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                           noise_y:noise_y+block_noise_size_y, 
                                                           noise_z:noise_z+block_noise_size_z]
        cnt -= 1
    return x
                
class PairDataGenerator(Dataset):
    """Re-implementation of a functional pair data generator, to avoid irreproduciable computer freezes, using PyTorch's Dataset class which has a built-in memory management.
    """
    def __init__(self, img_path, config) -> None:
        self.img_paths = img_path
        self.flip_rate = config.flip_rate
        self.local_rate = config.local_rate
        self.nonlinear_rate = config.nonlinear_rate
        self.paint_rate = config.paint_rate
        self.inpaint_rate = config.inpaint_rate
        # self.img_index = []
        # self.generate_index()

    def generate_index(self):
        for image_path in self.img_paths:
            offset = len(self.img_index)
            s = np.load(image_path)
            indices_and_path = [(offset, image_path) for i in range(s.shape[0])]
            self.img_index.extend(indices_and_path)
            del s
        
    def __len__(self):
        return len(self.img_paths)
        # return len(self.img_index)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        # offset, image_path = self.img_index[index]
        # y = np.load(image_path)[index - offset][np.newaxis, ...]
        y = self.img_paths[index]

        y = copy.deepcopy(y)

        # Autoencoder
        x = copy.deepcopy(y)

        # Flip
        x, y = data_augmentation(x, y, self.flip_rate)

        # Local Shuffle Pixel
        x = local_pixel_shuffling(x, prob=self.local_rate)
        
        # Apply non-Linear transformation with an assigned probability
        x = nonlinear_transformation(x, self.nonlinear_rate)
        
        # Inpainting & Outpainting
        if random.random() < self.paint_rate:
            if random.random() < self.inpaint_rate:
                # Inpainting
                x = image_in_painting(x)
            else:
                # Outpainting
                x = image_out_painting(x)

        return torch.Tensor(x.copy()), torch.Tensor(y.copy())


def generate_pair(img, batch_size, config, status="test"):
    img_rows, img_cols, img_deps = img.shape[2], img.shape[3], img.shape[4]
    while True:
        index = [i for i in range(img.shape[0])]
        random.shuffle(index)
        y = img[index[:batch_size]]
        x = copy.deepcopy(y)
        for n in range(batch_size):
            
            # Autoencoder
            x[n] = copy.deepcopy(y[n])
            
            # Flip
            x[n], y[n] = data_augmentation(x[n], y[n], config.flip_rate)

            # Local Shuffle Pixel
            x[n] = local_pixel_shuffling(x[n], prob=config.local_rate)
            
            # Apply non-Linear transformation with an assigned probability
            x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)
            
            # Inpainting & Outpainting
            if random.random() < config.paint_rate:
                if random.random() < config.inpaint_rate:
                    # Inpainting
                    x[n] = image_in_painting(x[n])
                else:
                    # Outpainting
                    x[n] = image_out_painting(x[n])

        # Save sample images module
        if config.save_samples is not None and status == "train" and random.random() < 0.01:
            n_sample = random.choice( [i for i in range(config.batch_size)] )
            sample_1 = np.concatenate((x[n_sample,0,:,:,2*img_deps//6], y[n_sample,0,:,:,2*img_deps//6]), axis=1)
            sample_2 = np.concatenate((x[n_sample,0,:,:,3*img_deps//6], y[n_sample,0,:,:,3*img_deps//6]), axis=1)
            sample_3 = np.concatenate((x[n_sample,0,:,:,4*img_deps//6], y[n_sample,0,:,:,4*img_deps//6]), axis=1)
            sample_4 = np.concatenate((x[n_sample,0,:,:,5*img_deps//6], y[n_sample,0,:,:,5*img_deps//6]), axis=1)
            final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
            final_sample = final_sample * 255.0
            final_sample = final_sample.astype(np.uint8)
            file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
            imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

        yield (x, y)


class KFoldNNUNetSegmentationDataModule(L.LightningDataModule):
    def __init__(self,
                 config: dict) -> None:
        self.config = config
        self.fold = self.config['data']['fold']
        self.dataDir = self.config['data']['data_directory'] # /storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset301_Calcium_OCT
        if isinstance(self.dataDir, str):
            self.dataDir = Path(self.dataDir)

        self.num_workers = self.config['data']['num_workers']
        self.batch_size = self.config['data']['batch_size']

    def setup(self, stage: str) -> None:
        """Define the split and data before putting them into dataloader

        Args:
            stage (str): Torch Lightning stage ('fit', 'validate', 'predict', ...), not used in the LightningDataModule
        """
        #TODO: Make tio.Queue and define the augmentation and preprocessing transformation for this dataset
        self.preprocess = self.getPreprocessTransform()
        self.augment = self.getAugmentationTransform()
        self.transform = tio.Compose([self.preprocess, self.augment])

        if stage == 'fit' or stage is None:
            trainImages, trainLabels = self._getImagesAndLabels('train')
            valImages, valLabels = self._getImagesAndLabels('val')

            trainSubjects = self._filesToSubject(trainImages, trainLabels)
            valSubjects = self._filesToSubject(valImages, valLabels)

            self.trainSet = tio.SubjectsDataset(trainSubjects, transform=self.transform)
            trainSampler = tio.data.UniformSampler(
                patch_size=self.config['data']['patch_size'],
            )
            self.patchesTrainSet = [
                tio.Queue(
                subjects_dataset=self.trainSet,
                max_length=self.config['data']['queue_max_length'],
                samples_per_volume=self.config['data']['samples_per_volume'],
                sampler=trainSampler,
                num_workers=self.num_workers // 2,
                shuffle_subjects=True,
                shuffle_patches=True,),
                tio.Queue(
                subjects_dataset=self.trainSet,
                max_length=self.config['data']['queue_max_length'],
                samples_per_volume=self.config['data']['samples_per_volume'],
                sampler=tio.data.LabelSampler(
                    patch_size=self.config['data']['patch_size'],
                    label_name='label'
                ),
                num_workers=self.num_workers // 2,
                shuffle_subjects=True,
                shuffle_patches=True,),
                 ]
            print('=====================================================================================================================\n')
            print('self.patchesTrainSet.iterations_per_epoch', self.patchesTrainSet[0].iterations_per_epoch)
            print('\n=====================================================================================================================')

            if len(valSubjects) == 0:
                valSubjects = trainSubjects
                print("Warning: Validation set is empty, using training set for validation")
            self.valSet = tio.SubjectsDataset(valSubjects, transform=self.preprocess)
            valSampler = tio.data.UniformSampler(
                patch_size=self.config['data']['patch_size'],
            )
            self.patchesValSet = tio.Queue(
                subjects_dataset=self.valSet,
                max_length=self.config['data']['queue_max_length'],
                samples_per_volume=self.config['data']['samples_per_volume'],
                sampler=valSampler,
                num_workers=self.num_workers,
                shuffle_subjects=False,
                shuffle_patches=False,
            )

        if stage == 'test':
            testImages, testLabels = self._getImagesAndLabels('test')

            testSubjects = self._filesToSubject(testImages, testLabels)
            self.testSubjectGridSamplers = [tio.inference.GridSampler(
                subject=testSubject,
                patch_size=self.config['data']['patch_size'],
                patch_overlap=(s//2 for s in self.config['data']['patch_size'])) for testSubject in testSubjects]
            self.testAggregators = [tio.inference.GridAggregator(gridSampler) for gridSampler in self.testSubjectGridSamplers]

    @staticmethod
    def collate_fn(batch, test=False):
        collated_batch = {
            'image': torch.stack([data['image'][tio.DATA] for data in batch], dim=0),
            'label': torch.stack([data['label'][tio.DATA] for data in batch], dim=0),
        }
        if test:
            collated_batch['location'] = torch.stack([data[tio.LOCATION] for data in batch], dim=0)

        return collated_batch

    def train_dataloader(self):
        concatenated_dataset = torch.utils.data.ConcatDataset(self.patchesTrainSet)
        return DataLoader(concatenated_dataset, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)
        return DataLoader(self.patchesTrainSet, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.patchesValSet, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)

    def test_dataloader(self) -> Tuple[List[DataLoader], List[tio.GridSampler]]:
        return [DataLoader(testSubjectGridSampler, batch_size=self.batch_size, num_workers=0, collate_fn=partial(self.collate_fn, test=True)) for testSubjectGridSampler in self.testSubjectGridSamplers], self.testSubjectGridSamplers

    def getPreprocessTransform(self):
        preprocess = tio.Compose([])
        return preprocess

    def getAugmentationTransform(self):
        augment = tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.1),
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10, isotropic=True, default_pad_value='minimum'),
            tio.RandomNoise(std=(0, 0.1)),
        ])
        return augment
    
    def _filesToSubject(self, imageFiles: List[Path], labelFiles: List[Path]) -> List[tio.Subject]:
        """Convert image and label files to TorchIO subjects

        Args:
            imageFiles (List[Path]): List of image files
            labelFiles (List[Path]): List of label files

        Returns:
            List[tio.Subject]: List of TorchIO subjects
        """
        subjects = []
        for imageFile, labelFile in zip(imageFiles, labelFiles):
            subject = tio.Subject(
                image=tio.ScalarImage(str(imageFile)),
                label=tio.LabelMap(str(labelFile)),
                name=imageFile.stem.split('_')[0]
            )
            subjects.append(subject)
        return subjects

    def _getSplit(self) -> Tuple[List[str], List[str]]:
        """Get the train and validation split for the current fold and split

        Returns:
            Tuple[List[str], List[str]]: List of train and validation unique case IDs
        """
        dataSetName = self.dataDir.stem
        if 'scale_path' in self.config['data']:
            splitPath = Path(self.config['data']['scale_path'])
        else:
            splitPath = self.dataDir / '..' / '..' / 'nnUNet_preprocessed' / dataSetName / 'splits_final.json'
        assert splitPath.exists(), f"Split file {splitPath} does not exist"
        with open(splitPath, 'r') as f:
            splits = json.load(f)
            train = splits[self.fold]['train']
            val = splits[self.fold]['val']

        return train, val

    def _getImagesAndLabels(self, split: str) -> Tuple[List[Path], List[Path]]:
        """Get the image and label files for the current fold and split

        Args:
            split (str): 'train', 'val', or 'test'

        Returns:
            Tuple[List[Path], List[Path]]: List of image and label path files
        """
        train, val = self._getSplit()
        train.sort()
        val.sort()

        if split in {'train', 'val'}:
            imageDir = self.dataDir / 'imagesTr'
            labelDir = self.dataDir / 'labelsTr'
        else:
            imageDir = self.dataDir / 'imagesTs'
            labelDir = self.dataDir / 'labelsTs'
        
        assert imageDir.exists(), f"Image directory {imageDir} does not exist"
        assert labelDir.exists(), f"Label directory {labelDir} does not exist"

        imageFiles = sorted(list(imageDir.glob('*.nii.gz')))
        labelFiles = sorted(list(labelDir.glob('*.nii.gz')))

        assert len(imageFiles) == len(labelFiles), f"Number of images and labels do not match: {len(imageFiles)} != {len(labelFiles)}"

        if split == 'train':
            caseFilter = train
        elif split == 'val':
            caseFilter = val
        else:
            caseFilter = None
        if caseFilter is not None:
            imageFiles = [f for f in imageFiles if f.stem.split('_')[0] in caseFilter]
            labelFiles = [f for f in labelFiles if f.stem.replace('.nii', '') in caseFilter]

        assert len(imageFiles) == len(labelFiles), f"Number of images and labels AFTER filtering do not match: {len(imageFiles)} != {len(labelFiles)}, filter={caseFilter}"

        return imageFiles, labelFiles

def convert_nnUNet_to_Genesis(model):
    model.decoder.seg_layers[-1] = torch.nn.Conv3d(32, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    return model

#Declare the Dice Loss
def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

class GenesisSegmentation(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self.build_network()

        if 'model' in config and 'freeze_encoder' in config['model'] and config['model']['freeze_encoder']:
            self.model.encoder.requires_grad_(False)

        metrics = MetricCollection({
            'dice': Dice(num_classes=1)
        })
        self.train_metrics = metrics.clone('train_')
        self.val_metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')

        self.test_grid_samplers: List[tio.GridSampler] = None
        self.prediction_aggregators: List[tio.GridAggregator] = None
        self.label_aggregators: List[tio.GridAggregator] = None

    def set_test_grid_samplers(self, test_grid_samplers: List[tio.GridSampler]):
        """Set the test grid samplers for the model
        Initialize the prediction and label aggregators for the test grid samplers
        This function should be called before running the test step
        The prediction and label aggregators are used to aggregate the predictions and labels from the test grid samplers
        Then, the aggregated predictions and labels can be used to calculate the test metrics

        Args:
            test_grid_samplers (List[tio.GridSampler]): _description_
        """
        self.test_grid_samplers = test_grid_samplers
        self.prediction_aggregators = [tio.inference.GridAggregator(grid_sampler) for grid_sampler in self.test_grid_samplers]
        self.label_aggregators = [tio.inference.GridAggregator(grid_sampler) for grid_sampler in self.test_grid_samplers]

    def build_network(self) -> torch.nn.Module:
        # prepare the 3D model
        # Initalize the model from nnUNet
        trainer = get_trainer_from_args(
            dataset_name_or_id=self.config['nnUNet']['dataset_name_or_id'],
            configuration=self.config['nnUNet']['configuration'],
            fold=self.config['nnUNet']['fold'],
            trainer_name=self.config['nnUNet']['trainer_name'],
            plans_identifier=self.config['nnUNet']['plans_identifier'],
            device=torch.device('cpu'))
        trainer.initialize()

        model = trainer.network

        load_pretrained_weights(model, self.config['pre_trained_weight_path'], verbose=True)

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total number of parameters in the model: ", pytorch_total_params)

        print("########## Check if the weights are loaded correctly (should ONLY print *.seg_layers.* in the keys) ##########")
        checkpoint = torch.load(self.config['pre_trained_weight_path'], map_location=torch.device('cpu'))
        for key in model.state_dict().keys():
            checkpoint_key = None
            for _checkpoint_key in checkpoint['network_weights'].keys():
                if _checkpoint_key.endswith(key) or key.endswith(_checkpoint_key):
                    checkpoint_key = _checkpoint_key
                    print('_checkpoint_key', checkpoint_key, '<->', key)
                    break
            if checkpoint_key is None:
                print('Cannot find a matching key in the checkpoing, skipping.')
                continue
            if not torch.equal(checkpoint['network_weights'][checkpoint_key], model.state_dict()[key]):
                print('key, checkpoint_key', key, checkpoint_key)
                print((checkpoint['network_weights'][checkpoint_key] - model.state_dict()[key]).sum())
                print('')
                assert '.seg_layers.' in key, 'Weights are not loaded correctly'
        print("########## Weights are loaded correctly ##########")
        return model

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        x = x.float()
        y = y.long()
        y_hat = self.model(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)

        loss = 0.5 * torch_dice_coef_loss(y_hat[:, 1, :, :], y.float()) + 0.5 * torch.nn.functional.cross_entropy(y_hat, torch.squeeze(y, dim=1), reduction="none").mean()
        self.train_metrics.update(y_hat[:, 1, :, :].reshape(-1), y.reshape(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.parameters()), # Make sure to filter the parameters based on `requires_grad`
            lr=self.config['optimizer']['learning_rate'],
            momentum=self.config['optimizer']['momentum'],
            weight_decay=self.config['optimizer']['weight_decay'],
            nesterov=self.config['optimizer']['nesterov'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.config['optimizer']['scheduler_step_size'], 
            gamma=self.config['optimizer']['scheduler_gamma'], 
            verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        x = x.float()
        y = y.long()
        y_hat = self.model(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)

        loss = 0.5 * torch_dice_coef_loss(y_hat[:, 1, :, :], y.float()) + 0.5 * torch.nn.functional.cross_entropy(y_hat, torch.squeeze(y, dim=1), reduction="none").mean()
        self.val_metrics.update(y_hat[:, 1, :, :].reshape(-1), y.reshape(-1))
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

        if self.global_step % 100 == 0:
            import matplotlib.pyplot as plt
            import wandb
            import os

            sum_label = torch.sum(y, dim=(1,2,3,4))
            positive_sample_idx = torch.where(sum_label > 0)[0]
            if len(positive_sample_idx):
                positive_sample_idx = positive_sample_idx[0].item()
            else:
                print('No positive sample found in the validation set, using the first item in the validation batch')
                positive_sample_idx = 0

            negative_sample_idx = torch.where(sum_label == 0)[0]
            if len(negative_sample_idx):
                negative_sample_idx = negative_sample_idx[0].item()
            else:
                print('No negative sample found in the validation set, using the first item in the validation batch')
                negative_sample_idx = 0
            for i, kind in zip([positive_sample_idx, negative_sample_idx], ['Positive', 'Negative']):
                sample = tio.Subject(
                    image=tio.ScalarImage(tensor=x[i].detach().cpu()),
                    label=tio.LabelMap(tensor=y[i].detach().cpu()),
                    prediction=tio.ScalarImage(tensor=y_hat[i].detach().cpu()),
                )
                try:
                    fig = plt.figure(num=1, clear=True, figsize=(10, 10))
                    ax = fig.add_subplot()
                    sample.plot()
                    ax.set_title(f"{kind} Validation Sample Loss {loss.item():.4f}")
                    random_string = wandb.util.generate_id()
                    image_name = f"sample_{random_string}.png"
                    plt.savefig(image_name)
                    sample_image = plt.imread(image_name)
                    os.remove(image_name)
                    plt.close("all")
                    self.logger.experiment.log(
                        {f'val_sample_{kind}': [wandb.Image(sample_image)]}
                    )
                except np.linalg.LinAlgError:
                    print("Error plotting sample")
                    np.save(f"sample_error_{self.validation_step}.npy", sample)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch['image'], batch['label']
        location = batch['location']
        x = x.float()
        y = y.long()
        y_hat = self.model(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        y_hat = torch.argmax(y_hat, dim=1).unsqueeze(dim=1)

        # Don't worry about this being in GPU, the aggregators are will put the data in the CPU
        self.prediction_aggregators[dataloader_idx].add_batch(y_hat, location)
        self.label_aggregators[dataloader_idx].add_batch(y, location)
        return None
    
    def on_test_epoch_end(self):
        for idx, (prediction_aggregator, label_aggregator) in enumerate(zip(self.prediction_aggregators, self.label_aggregators)):
            torch.save(prediction_aggregator.get_output_tensor(), Path(self.trainer.log_dir) / f"fold_{self.config['data']['fold']}" / f'prediction_{idx}.pt')
            torch.save(label_aggregator.get_output_tensor(), Path(self.trainer.log_dir) / f"fold_{self.config['data']['fold']}" / f'label_{idx}.pt')
            self.test_metrics.update(prediction_aggregator.get_output_tensor().view(-1).float(), label_aggregator.get_output_tensor().view(-1))
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

# Define the UNetRegressor
class UNetRegressorHead(torch.nn.Module):
    def __init__(self, in_channels: int, n_classes: int, pooling="avg", dropout=0.2, task: str = 'regression'):
        super(UNetRegressorHead, self).__init__()
        """
        >>> input_sample = torch.randn(1, 1, 512, 512, 384)
        >>> output_sample = encoder(input_sample)
        >>> [feat.shape for feat in output_sample]
        >>> [torch.Size([1, 32, 512, 512, 384]),
             torch.Size([1, 64, 256, 256, 192]),
             torch.Size([1, 128, 128, 128, 96]),
             torch.Size([1, 256, 64, 64, 48]),
             torch.Size([1, 320, 32, 32, 24]),
             torch.Size([1, 320, 32, 16, 12])]
        """
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))

        if task not in ("regression", "classification"):
            raise ValueError("Task should be one of ('regression', 'classification'), got {}.".format(task))

        self.regressor = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool3d(1) if pooling == "avg" else torch.nn.AdaptiveMaxPool3d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=dropout, inplace=True) if dropout > 0 else torch.nn.Identity(),
            torch.nn.Linear(in_channels, n_classes, bias=True),
            torch.nn.ReLU() if task == 'regression' else torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.regressor(x)
        return x

class KFoldNNUNetTabularDataModule(L.LightningDataModule):
    def __init__(self,
                 config: dict) -> None:
        self.config = config
        self.fold = self.config['data']['fold']
        self.dataDir = self.config['data']['data_directory'] # /storage_bizon/naravich/Unlabeled_OCT_by_CADx/NiFTI/
        self.tabularDataDir = Path(self.config['data']['tabular_data_directory'])
        
        self.inputModality = self.config['data']['input_modality'] # ('pre', 'post', 'final')
        self.outputModality = self.config['data']['output_modality'] # ('pre', 'post', 'final')
        self.outputMetrics: List[str] = self.config['data']['output_metrics']
        # Just for now
        self.modalityToDataframePath = {
            'pre': self.tabularDataDir / 'Pre_IVL.csv',
            'post': self.tabularDataDir / 'Post_IVL.csv',
            'final': self.tabularDataDir / 'Post_Stent.csv'
        }
        self.modalityToName = {
            'pre': 'Pre_IVL',
            'post': 'Post_IVL',
            'final': 'Post_Stent'
        }

        if isinstance(self.dataDir, str):
            self.dataDir = Path(self.dataDir)

        self.num_workers = self.config['data']['num_workers']
        self.batch_size = self.config['data']['batch_size']

    def setup(self, stage: str) -> None:
        """Define the split and data before putting them into dataloader

        Args:
            stage (str): Torch Lightning stage ('fit', 'validate', 'predict', ...), not used in the LightningDataModule
        """
        self.preprocess = self.getPreprocessTransform()
        self.augment = self.getAugmentationTransform()
        self.transform = tio.Compose([self.preprocess, self.augment])

        # create subject from the CSV!
        outputModalityDf = pd.read_csv(self.modalityToDataframePath[self.outputModality])
        inputName = self.modalityToName[self.inputModality]
        outputModalityDf = outputModalityDf[['USUBJID'] + self.outputMetrics + [f'{inputName}_image_path']]

        if self.config['data']['nan_handling'] == 'drop':
            outputModalityDf.dropna(inplace=True)
        elif self.config['data']['nan_handling'] == 'mean':
            outputModalityDf.fillna(outputModalityDf.mean(), inplace=True)
        elif self.config['data']['nan_handling'] == 'median':
            outputModalityDf.fillna(outputModalityDf.median(), inplace=True)
        elif self.config['data']['nan_handling'] == 'zero':
            outputModalityDf.fillna(0, inplace=True)


        if self.config['data']['target_normalization'] == 'minmax':
            # Get the train split to normalize the target
            train_subject_ids, _ = self._getSplit('fit', outputModalityDf=outputModalityDf)
            trainDF = outputModalityDf[outputModalityDf['USUBJID'].isin(train_subject_ids)]
            outputModalityDf[self.outputMetrics] = (outputModalityDf[self.outputMetrics] - trainDF[self.outputMetrics].min(axis=0)) / (trainDF[self.outputMetrics].max(axis=0) - trainDF[self.outputMetrics].min(axis=0))
        
        train_subject_ids, val_subject_ids = self._getSplit(stage, outputModalityDf=outputModalityDf)

        if stage == 'test':
            test_subject_ids = train_subject_ids

        # Create the subjects
        if stage == 'fit':
            if self.config['data']['overfit']:
                val_subject_ids = train_subject_ids.copy()
                train_subject_ids = train_subject_ids[:1]
                val_subject_ids = val_subject_ids[:1]
            trainSubjects = []
            for _, row in outputModalityDf[outputModalityDf['USUBJID'].isin(train_subject_ids)].iterrows():
                assert (self.dataDir / row[f'{inputName}_image_path']).exists(), '{} does not exist'.format(self.dataDir / row[f'{inputName}_image_path'])
                subject = tio.Subject(
                    **{metric: row[metric] for metric in self.outputMetrics},
                    image=tio.ScalarImage(self.dataDir / row[f'{inputName}_image_path'])
                )
                trainSubjects.append(subject)
            
            valSubjects = []
            for _, row in outputModalityDf[outputModalityDf['USUBJID'].isin(val_subject_ids)].iterrows():
                assert (self.dataDir / row[f'{inputName}_image_path']).exists(), '{} does not exist'.format(self.dataDir / row[f'{inputName}_image_path'])
                subject = tio.Subject(
                    **{metric: row[metric] for metric in self.outputMetrics},
                    image=tio.ScalarImage(self.dataDir / row[f'{inputName}_image_path'])
                )
                valSubjects.append(subject)

            self.trainSet = tio.SubjectsDataset(subjects=trainSubjects, transform=self.transform)
            self.valSet = tio.SubjectsDataset(subjects=valSubjects, transform=self.transform)
            self.patchesTrainSet = tio.Queue(
                subjects_dataset=self.trainSet,
                max_length=self.config['data']['queue_max_length'],
                samples_per_volume=self.config['data']['samples_per_volume'],
                sampler=tio.UniformSampler(patch_size=self.config['data']['patch_size']),
                num_workers=self.num_workers,
                shuffle_subjects=True,
                shuffle_patches=True,)
            self.patchesValSet = tio.Queue(
                subjects_dataset=self.valSet,
                max_length=self.config['data']['queue_max_length'],
                samples_per_volume=self.config['data']['samples_per_volume'],
                sampler=tio.UniformSampler(patch_size=self.config['data']['patch_size']),
                num_workers=self.num_workers,
                shuffle_subjects=False,
                shuffle_patches=False,)
        elif stage == 'test':
            testSubjects = []
            for _, row in outputModalityDf[outputModalityDf['USUBJID'].isin(test_subject_ids)].iterrows():
                subject = tio.Subject(
                    **{metric: row[metric] for metric in self.outputMetrics},
                    image=tio.ScalarImage(self.dataDir / row[f'{inputName}_image_path'])
                )
                testSubjects.append(subject)
            self.testSet = tio.SubjectsDataset(subjects=testSubjects, transform=self.preprocess)
            sampler = tio.UniformSampler(patch_size=self.config['data']['patch_size'])
            self.patchesTestSet = tio.Queue(
                subjects_dataset=self.testSet,
                max_length=len(testSubjects),
                samples_per_volume=1,
                sampler=sampler,
                num_workers=self.num_workers,
                shuffle_subjects=False,
                shuffle_patches=False,)
        

    def getPreprocessTransform(self):
            preprocess = tio.Compose([
                tio.transforms.Resize(target_shape=self.config['data']['patch_size'], image_interpolation='bspline'),
                # tio.CropOrPad(target_shape=self.config['data']['patch_size']), # patch_size = 512, 512, 384 (i.e. WHOLE VOLUME)
            ])
            return preprocess

    def getAugmentationTransform(self):
        augment = tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.2),
            tio.RandomAffine(scales=(0.9, 1.1), degrees=180, isotropic=True, default_pad_value='minimum', p=0.2),
            tio.RandomAnisotropy(p=0.2),
            tio.RandomNoise(std=(0, 0.1), p=0.2),
            tio.RandomBlur(p=0.2),
            tio.RandomGamma(p=0.2),
        ])
        return augment
    

    def _getSplit(self, stage: str, outputModalityDf: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Get the train and validation split for the current fold and split

        Returns:
            Tuple[List[str], List[str]]: List of train and validation unique case IDs, if stage is 'test' return test case IDs, and None
        """
        if not os.path.exists('tabular_data/splits_final.json'):
            from sklearn.model_selection import train_test_split
            subject_ids = outputModalityDf['USUBJID'].tolist()
            train_subject_ids, test_subject_ids = train_test_split(subject_ids, test_size=0.2, random_state=0)
            with open(self.tabularDataDir / 'test.json', 'w') as f:
                json.dump(test_subject_ids, f, indent=4)
            splits = []
            for fold in range(3):
                fold_train_subject_ids, fold_val_subject_ids = train_test_split(train_subject_ids, test_size=0.2, random_state=0)
                splits.append({
                    'train': fold_train_subject_ids,
                    'val': fold_val_subject_ids,
                })
            with open(self.tabularDataDir / 'splits_final.json', 'w') as f:
                json.dump(splits, f, indent=4)

        if stage == 'fit':
            with open(self.tabularDataDir / 'splits_final.json', 'r') as f:
                splits = json.load(f)
            train_subject_ids = splits[self.fold]['train']
            val_subject_ids = splits[self.fold]['val']
            return train_subject_ids, val_subject_ids

        elif stage == 'test':
            with open(self.tabularDataDir / 'test.json', 'r') as f:
                test_subject_ids = json.load(f)
            return test_subject_ids, None

    @staticmethod
    def collate_fn(batch, test=False):
        collated_batch = {
            metric: torch.tensor([[data[metric]] for data in batch]) for metric in batch[0].keys() if metric != 'image' and metric != 'location'
        }
        collated_batch['image'] = torch.stack([data['image'][tio.DATA] for data in batch], dim=0)

        if test:
            collated_batch['location'] = torch.stack([data[tio.LOCATION] for data in batch], dim=0)

        return collated_batch

    def train_dataloader(self):
        return DataLoader(self.patchesTrainSet, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.patchesValSet, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)

    def test_dataloader(self) -> Tuple[List[DataLoader], List[tio.GridSampler]]:
        return DataLoader(self.patchesTestSet, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)

class nnUNetRegressionClassification(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = self.build_network().encoder
        self.head = UNetRegressorHead(
            in_channels=self.config['model']['head']['in_channels'], 
            n_classes=self.config['data']['num_classes'], 
            pooling=self.config['model']['head']['pooling'], 
            dropout=self.config['model']['head']['dropout'], 
            task=self.config['model']['head']['task'])

        if 'model' in config and 'freeze_encoder' in config['model'] and config['model']['freeze_encoder']:
            self.encoder.requires_grad_(False)

        if self.config['model']['head']['task'] == 'regression':
            metrics = MetricCollection({
                f'rmse_{metric}': MeanSquaredError(squared=True) for metric in config['data']['output_metrics']
            })
            self.criterion = torch.nn.MSELoss()

        else:
            # TODO: handle multiple targets
            metrics = MetricCollection({
                'accuracy': Accuracy(),
                'f1': F1Score(num_classes=self.config['data']['num_classes']),
                'precision': Precision(num_classes=self.config['data']['num_classes']),
                'recall': Recall(num_classes=self.config['data']['num_classes']),
            })
            self.criterion = torch.nn.CrossEntropyLoss()

        self.train_metrics = metrics.clone('train_')
        self.val_metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')

    def build_network(self) -> torch.nn.Module:
        # prepare the 3D model
        # Initalize the model from nnUNet
        trainer = get_trainer_from_args(
            dataset_name_or_id=self.config['nnUNet']['dataset_name_or_id'],
            configuration=self.config['nnUNet']['configuration'],
            fold=self.config['nnUNet']['fold'],
            trainer_name=self.config['nnUNet']['trainer_name'],
            plans_identifier=self.config['nnUNet']['plans_identifier'],
            device=torch.device('cpu'))
        trainer.initialize()
        model = trainer.network
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total number of parameters in the model: ", pytorch_total_params)

        if 'pre_trained_weight_path' in self.config and self.config['pre_trained_weight_path'] is not None:
            #Load pre-trained weights
            weight_dir = self.config['pre_trained_weight_path']
            checkpoint = torch.load(weight_dir, map_location=torch.device('cpu'))
            state_dict = checkpoint['state_dict']
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            
            if '_orig_mod.' in list(model.state_dict().keys())[0]:
                # Since torch.optimized is used we need to add "_orig_mod." from the keys
                print("Adding _orig_mod. to the state_dict")
                unParalled_state_dict = {f"_orig_mod.{k}": v for k, v in unParalled_state_dict.items()}

            model.load_state_dict(unParalled_state_dict)
        return model

    def training_step(self, batch, batch_idx):
        x = batch['image']
        x = x.float()
        # one metric : N, 1
        y = torch.concat([batch[metric] for metric in self.config['data']['output_metrics']], dim=1)
        y = y.float()

        y_hat = self.encoder(x)
        y_hat = self.head(y_hat[-1])

        loss = self.criterion(y_hat, y)
        print(f'loss {loss}; y_hat {y_hat}; y {y}')

        for i, metric in enumerate(self.config['data']['output_metrics']):
            self.train_metrics[f'rmse_{metric}'].update(y_hat[:, i], y[:, i])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['image']
        x = x.float()
        # one metric : N, 1
        y = torch.concat([batch[metric] for metric in self.config['data']['output_metrics']], dim=1)
        y = y.float()

        y_hat = self.encoder(x)
        y_hat = self.head(y_hat[-1])

        loss = self.criterion(y_hat, y)

        for i, metric in enumerate(self.config['data']['output_metrics']):
            self.val_metrics[f'rmse_{metric}'].update(y_hat[:, i], y[:, i])
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['image']
        x = x.float()
        # one metric : N, 1
        y = torch.concat([batch[metric] for metric in self.config['data']['output_metrics']], dim=1)
        y = y.float()

        y_hat = self.encoder(x)
        y_hat = self.head(y_hat[-1])

        for i, metric in enumerate(self.config['data']['output_metrics']):
            self.test_metrics[f'rmse_{metric}'].update(y_hat[:, i], y[:, i])
        self.test_metrics.update(y_hat.view(-1), y.view(-1))
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return None

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     filter(lambda p: p.requires_grad, self.parameters()), # Make sure to filter the parameters based on `requires_grad`
        #     lr=self.config['optimizer']['learning_rate'],
        #     momentum=self.config['optimizer']['momentum'],
        #     weight_decay=self.config['optimizer']['weight_decay'],
        #     nesterov=self.config['optimizer']['nesterov'])
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config['optimizer']['learning_rate'],
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, 
        #     step_size=self.config['optimizer']['scheduler_step_size'], 
        #     gamma=self.config['optimizer']['scheduler_gamma'], 
        #     verbose=True)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': scheduler,
        }