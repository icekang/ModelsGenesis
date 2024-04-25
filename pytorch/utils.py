from __future__ import print_function
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
from torchmetrics import Dice, MetricCollection
from nnunetv2.run.run_training import get_trainer_from_args


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
                 fold: int,
                 dataDir: Union[Path,str]) -> None:
        self.fold = fold
        self.dataDir = dataDir # /storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset301_Calcium_OCT
        if isinstance(dataDir, str):
            self.dataDir = Path(dataDir)

        self.num_workers = 4
        self.batch_size = 4

    def setup(self, stage: str) -> None:
        """Define the split and data before putting them into dataloader

        Args:
            stage (str): Torch Lightning stage ('fit', 'validate', 'predict', ...), not used in the LightningDataModule
        """
        #TODO: Make tio.Queue and define the augmentation and preprocessing transformation for this dataset
        self.preprocess = self.getAugmentationTransform()
        self.augment = self.getAugmentationTransform()
        self.transform = tio.Compose([self.preprocess, self.augment])

        if stage == 'fit' or stage is None:
            trainImages, trainLabels = self._getImagesAndLabels('train')
            valImages, valLabels = self._getImagesAndLabels('val')

            trainSubjects = self._filesToSubject(trainImages, trainLabels)
            valSubjects = self._filesToSubject(valImages, valLabels)

            self.trainSet = tio.SubjectsDataset(trainSubjects, transform=self.transform)
            
            # TODO: Define hyperparameters as a config file
            self.sampler = tio.data.LabelSampler(
                patch_size=(128, 128, 64),
                label_name = 'label',
            )
            self.patchesTrainSet = tio.Queue(
                subjects_dataset=self.trainSet,
                max_length=100,
                samples_per_volume=75,
                sampler=self.sampler,
                num_workers=self.num_workers,
                shuffle_subjects=True,
                shuffle_patches=True,
            )

            if len(valSubjects) == 0:
                valSubjects = trainSubjects
                print("Warning: Validation set is empty, using training set for validation")
            self.valSet = tio.SubjectsDataset(valSubjects, transform=self.preprocess)
            self.patchesValSet = tio.Queue(
                subjects_dataset=self.valSet,
                max_length=100,
                samples_per_volume=75,
                sampler=self.sampler,
                num_workers=self.num_workers,
                shuffle_subjects=False,
                shuffle_patches=False,
            )

        if stage == 'test':
            testImages, testLabels = self._getImagesAndLabels('test')

            testSubjects = self._filesToSubject(testImages, testLabels)
            self.testSubjectGridSamplers = [tio.inference.GridSampler(subject=testSubject, patch_size=(128, 128, 64)) for testSubject in testSubjects]
            self.testAggregators = [tio.inference.GridAggregator(gridSampler) for gridSampler in self.testSubjectGridSamplers]

    @staticmethod
    def collate_fn(batch):
        batch = {
            'image': torch.stack([data['image'][tio.DATA] for data in batch], dim=0),
            'label': torch.stack([data['label'][tio.DATA] for data in batch], dim=0),
        }
        return batch

    def train_dataloader(self):
        return DataLoader(self.patchesTrainSet, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.patchesValSet, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn)

    def test_dataloader(self) -> Tuple[List[DataLoader], List[tio.GridSampler]]:
        return [DataLoader(testSubjectGridSampler, batch_size=self.batch_size, num_workers=0) for testSubjectGridSampler in self.testSubjectGridSamplers], self.testSubjectGridSamplers

    def getAugmentationTransform(self):
        preprocess =tio.Compose([])
        return preprocess

    def getAugmentationTransform(self):
        augment = tio.Compose([])
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

    def build_network(self):
        # prepare the 3D model
        # Initalize the model from nnUNet
        trainer = get_trainer_from_args(
            dataset_name_or_id=self.config['nnUNet']['dataset_name_or_id'],
            configuration=self.config['nnUNet']['configuration'],
            fold=self.config['nnUNet']['fold'],
            trainer_name=self.config['nnUNet']['trainer_name'],
            plans_identifier=self.config['nnUNet']['plans_identifier'])
        trainer.initialize()
        model = convert_nnUNet_to_Genesis(trainer.network)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total number of parameters in the model: ", pytorch_total_params)

        #Load pre-trained weights
        weight_dir = self.config['pre_trained_weight_path']
        checkpoint = torch.load(weight_dir, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        
        # Since torch.optimized is used we need to add "_orig_mod." from the keys
        unParalled_state_dict = {f"_orig_mod.{k}": v for k, v in unParalled_state_dict.items()}

        model.load_state_dict(unParalled_state_dict)
        return model

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        x = x.float()
        y = y.long()
        y_hat = self.model(x)
        y_hat = y_hat.sigmoid()

        loss = torch_dice_coef_loss(y_hat, y)
        self.train_metrics.update(y_hat.view(-1), y.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config['optimizer']['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config['optimizer']['scheduler_step_size'], gamma=self.config['optimizer']['scheduler_gamma'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        x = x.float()
        y = y.long()
        y_hat = self.model(x)
        y_hat = y_hat.sigmoid()

        loss = torch_dice_coef_loss(y_hat, y)
        self.val_metrics.update(y_hat.view(-1), y.view(-1))
        self.log('val_loss', loss)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch['image'][tio.DATA], batch['label'][tio.DATA]
        location = batch[tio.LOCATION]
        x = x.float()
        y = y.long()
        y_hat = self.model(x)
        y_hat = y_hat.sigmoid()
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0

        # Don't worry about this being in GPU, the aggregators are will put the data in the CPU
        self.prediction_aggregators[dataloader_idx].add_batch(y_hat, location)
        self.label_aggregators[dataloader_idx].add_batch(y, location)
        return None
    
    def on_test_epoch_end(self):
        for idx, (prediction_aggregator, label_aggregator) in enumerate(zip(self.prediction_aggregators, self.label_aggregators)):
            torch.save(prediction_aggregator.get_output_tensor(), Path(self.trainer.log_dir) / self.config['fold'] / f'prediction_{i}.pt')
            torch.save(label_aggregator.get_output_tensor(), Path(self.trainer.log_dir) / self.config['fold'] / f'label_{i}.pt')
            self.test_metrics.update(prediction_aggregator.get_output_tensor().view(-1), label_aggregator.get_output_tensor().view(-1))
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)