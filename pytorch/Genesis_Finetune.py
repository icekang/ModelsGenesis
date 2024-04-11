import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import KFoldNNUNetSegmentationDataModule
import torchio as tio
from nnunetv2.run.run_training import get_trainer_from_args
from utils import *
import wandb

class _config:
    seed = 42
    nb_epoch = 100
    batch_size = 4
    lr = 1e-3
    wandb_run_id = None
    weights = None
    wandb_resume = False
    wandb_project_name = 'GenesisFinetuneTest'
    wandb_run_name = None

    # logs
    model_path = "/storage_bizon/naravich/ModelGenesisNNUNetFinetuningV2_noNorm/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    def to_dict(self):
        return {a: getattr(self, a) for a in dir(self) if not a.startswith("__") and not callable(getattr(self, a))}
# for data in dm.train_dataloader():
#     break

config = _config()
set_seed(config.seed)

wandb_resume = config.wandb_run_id is not None
if config.wandb_run_id == None:
    config.wandb_run_id = wandb.util.generate_id()

#Declare the Dice Loss
def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

# prepare your own data
dm = KFoldNNUNetSegmentationDataModule(fold=0, dataDir='/storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset302_Calcium_OCTv2/')
dm.setup('fit')
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
# train_loader = DataLoader(Your Dataset, batch_size=config.batch_size, shuffle=True)

# prepare the 3D model
# Initalize the model from nnUNet
trainer = get_trainer_from_args(dataset_name_or_id="301", configuration="3d_fullres", fold=0, trainer_name="nnUNetTrainer", plans_identifier="nnUNetPlans")
trainer.initialize()
model = convert_nnUNet_to_Genesis(trainer.network)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters in the model: ", pytorch_total_params)

print("Total CUDA devices: ", torch.cuda.device_count())

scaler = torch.cuda.amp.GradScaler()

#Load pre-trained weights
# weight_dir = 'pretrained_weights/Genesis_Chest_Best.pt'
# checkpoint = torch.load(weight_dir, map_location=torch.device('cpu'))
# state_dict = checkpoint['state_dict']
# unParalled_state_dict = {}
# for key in state_dict.keys():
#     unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
# model.load_state_dict(unParalled_state_dict)

# print(model) # 19073665 # 30785994 


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
# model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])

criterion = torch_dice_coef_loss
optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, verbose=False)


# train the model
# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []
best_epoch_loss = 100000

intial_epoch = 0
num_epoch_no_improvement = 0
sys.stdout.flush()

# if config.weights != None:
	# checkpoint=torch.load(config.weights)
	# model.load_state_dict(checkpoint['state_dict'])
	# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	# scaler.load_state_dict(checkpoint['scaler_state_dict'])
	# intial_epoch=checkpoint['epoch']
	# best_epoch_loss = checkpoint['best_epoch_loss']
	# num_epoch_no_improvement = checkpoint['num_epoch_no_improvement']
	# print("Loading weights from ", config.weights)
sys.stdout.flush()

if not wandb_resume:
	wandb.init(
		id=config.wandb_run_id,
		project=config.wandb_project_name,
		name=config.wandb_run_name,
		config=config.to_dict(),
		dir=config.logs_path
	)
else:
	wandb.init(
		id=config.wandb_run_id,
		resume='must',
		project=config.wandb_project_name,
		name=config.wandb_run_name,
		config=config.to_dict(),
		dir=config.logs_path
	)

for batch_ndx, data in enumerate(train_loader):
    values = torch.unique(data['label'][tio.DATA]).tolist()
    if len(values) > 1:
        print('Found a batch with more than 1 unique value in the label', values)
        break
for epoch in range(intial_epoch, config.nb_epoch):
    model.train()
    sum_train_loss = 0.0
    count_train_loss = 0
    # for batch_ndx, (x,y) in enumerate(train_loader):
    #     break
    for i in range(10):

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            x, y = data['image'][tio.DATA], data['label'][tio.DATA]
            x, y = x.float().to(device), y.float().to(device)
            pred = model(x)
            pred = pred.sigmoid()
            loss = criterion(y, pred)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        sum_train_loss += loss.item()
        count_train_loss += 1

    if count_train_loss == 0:
        count_train_loss = 1
    avg_train_loss = sum_train_loss / count_train_loss
    wandb.log({'train/loss': avg_train_loss})
    print(f'Epoch [{epoch + 1}/{config.nb_epoch}], Loss: {avg_train_loss}')
    scheduler.step()
