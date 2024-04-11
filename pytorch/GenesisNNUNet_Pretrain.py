#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import torch
from torchsummary import summary
import sys
from utils import *
import unet3d
from config_genesis_nnunet import models_genesis_config
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from nnunetv2.run.run_training import get_trainer_from_args
import torch
from time import time


print("torch = {}".format(torch.__version__))


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
conf = models_genesis_config()

wandb_resume = conf.wandb_run_id is not None
if conf.wandb_run_id == None:
    conf.wandb_run_id = wandb.util.generate_id()

conf.display()
set_seed(conf.seed)

x_train = []
for i,fold in enumerate(tqdm(conf.train_fold)):
    file_name = "bat_"+str(conf.scale)+"_s_"+str(conf.input_rows)+"x"+str(conf.input_cols)+"x"+str(conf.input_deps)+"_"+str(fold)+".npy"
    # x_train.append(os.path.join(conf.data, file_name))
    s = np.load(os.path.join(conf.data, file_name))
    x_train.extend(s)
x_train = np.expand_dims(np.array(x_train), axis=1)

x_valid = []
for i,fold in enumerate(tqdm(conf.valid_fold)):
    file_name = "bat_"+str(conf.scale)+"_s_"+str(conf.input_rows)+"x"+str(conf.input_cols)+"x"+str(conf.input_deps)+"_"+str(fold)+".npy"
    # x_valid.append(os.path.join(conf.data, file_name))
    s = np.load(os.path.join(conf.data, file_name))
    x_valid.extend(s)
x_valid = np.expand_dims(np.array(x_valid), axis=1)

print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))

train_dataloader = PairDataGenerator(x_train, conf)
train_dataloader = DataLoader(train_dataloader, batch_size=conf.batch_size, shuffle=True, num_workers=conf.workers)

valid_dataloader = PairDataGenerator(x_valid, conf)
valid_dataloader = DataLoader(valid_dataloader, batch_size=conf.batch_size, shuffle=False, num_workers=conf.workers)


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initalize the model from nnUNet
trainer = get_trainer_from_args(dataset_name_or_id="301", configuration="3d_fullres", fold=0, trainer_name="nnUNetTrainer", plans_identifier="nnUNetPlans")
trainer.initialize()
model = convert_nnUNet_to_Genesis(trainer.network)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters in the model: ", pytorch_total_params)

model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
model.to(device)

print("Total CUDA devices: ", torch.cuda.device_count())

# summary(model, (1,conf.input_rows,conf.input_cols,conf.input_deps), batch_size=-1)
scaler = torch.cuda.amp.GradScaler()
criterion = nn.MSELoss()

if conf.optimizer == "sgd":
	optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
elif conf.optimizer == "adam":
	optimizer = torch.optim.Adam(model.parameters(), conf.lr)
else:
	raise

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(conf.patience * 0.8), gamma=0.5)

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

if conf.weights != None:
	checkpoint=torch.load(conf.weights)
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	scaler.load_state_dict(checkpoint['scaler_state_dict'])
	intial_epoch=checkpoint['epoch']
	best_epoch_loss = checkpoint['best_epoch_loss']
	num_epoch_no_improvement = checkpoint['num_epoch_no_improvement']
	print("Loading weights from ",conf.weights)
sys.stdout.flush()

if not wandb_resume:
	wandb.init(
		id=conf.wandb_run_id,
		project=conf.wandb_project_name,
		name=conf.wandb_run_name,
		config=conf.to_dict(),
		dir=conf.logs_path
	)
else:
	wandb.init(
		id=conf.wandb_run_id,
		resume='must',
		project=conf.wandb_project_name,
		name=conf.wandb_run_name,
		config=conf.to_dict(),
		dir=conf.logs_path
	)


for epoch in range(intial_epoch,conf.nb_epoch):
	scheduler.step(epoch)
	model.train()
	epoch_start = time()
	for iteration, data in enumerate(train_dataloader):
		with torch.autocast(device_type="cuda", dtype=torch.float16):
			image, gt = data
			gt = torch.repeat_interleave(gt,conf.nb_class,axis=1)
			image,gt = image.float().to(device), gt.float().to(device)
			pred=model(image)
			pred = pred.sigmoid()
			loss = criterion(pred,gt)
		
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
		optimizer.zero_grad(set_to_none=True)
		train_losses.append(loss.item())
		if (iteration + 1) % 5 ==0:
			iteration_train_loss = np.average(train_losses)
			print('Epoch [{}/{}], iteration {}, Loss: {:.6f}'
				.format(epoch + 1, conf.nb_epoch, iteration + 1, iteration_train_loss))
			sys.stdout.flush()
			wandb.log({
				"train/loss": iteration_train_loss,
			})
	epoch_end = time()
	print("Epoch {} took {:.2f} seconds".format(epoch, epoch_end - epoch_start))
	with torch.no_grad():
		model.eval()
		print("validating....")
		should_save_visual_logs = epoch % 10 == 0
		# Save the best/worst validation of the epoch
		# Initialize every validation
		best_loss = 100000
		worst_loss = -100000
		for x, y in valid_dataloader:
			y = torch.repeat_interleave(y,conf.nb_class,axis=1)
			image, gt = x.float(), y.float()
			gt=gt.to(device)
			pred=model(image)
			pred = pred.sigmoid()
			loss = criterion(pred,gt)
			valid_losses.append(loss.item())

			if should_save_visual_logs:
				# Save the best/worst validation 
				if loss.item() > worst_loss:
					worst_loss = loss.item()
					worst_pred = pred[0].cpu()
					worst_gt = y[0].cpu()
					worst_image = x[0].cpu()
				if loss.item() < best_loss:
					best_loss = loss.item()
					best_pred = pred[0].cpu()
					best_gt = y[0].cpu()
					best_image = x[0].cpu()
		if should_save_visual_logs:
			import matplotlib.pyplot as plt
			import torchio as tio
			worst_sample = tio.Subject(
				image=tio.ScalarImage(tensor=worst_image),
				prediction=tio.ScalarImage(tensor=worst_pred),
				target=tio.ScalarImage(tensor=worst_gt),
			)
			best_sample = tio.Subject(
				image=tio.ScalarImage(tensor=best_image),
				prediction=tio.ScalarImage(tensor=best_pred),
				target=tio.ScalarImage(tensor=best_gt),
			)
			try:
				fig = plt.figure(num=1, clear=True)
				ax = fig.add_subplot()
				worst_sample.plot()
				plt.title("Worst Sample Loss {:.4f}".format(worst_loss))
				plt.savefig("worst_sample.png")
				plt.close("all")
			except np.linalg.LinAlgError:
				print("Error plotting worst sample")
				np.save(f"worst_sample_error_{wandb.util.generate_id()}.npy", worst_sample)
			try:
				fig = plt.figure(num=1, clear=True)
				ax = fig.add_subplot()
				best_sample.plot()
				plt.title("Best Sample Loss {:.4f}".format(best_loss))
				plt.savefig("best_sample.png")
				plt.close("all")
			except np.linalg.LinAlgError:
				print("Error plotting best sample")
				np.save(f"best_sample_error_{wandb.util.generate_id()}.npy", best_sample)
			wandb.log({
				"valid/worst_loss": worst_loss,
				"valid/worst_sample": [wandb.Image(plt.imread("worst_sample.png"))],
				"valid/best_loss": best_loss,
				"valid/best_sample": [wandb.Image(plt.imread("best_sample.png"))],
			})

	#logging
	train_loss=np.average(train_losses)
	valid_loss=np.average(valid_losses)
	avg_train_losses.append(train_loss)
	avg_valid_losses.append(valid_loss)
	print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
	wandb.log({
		"train/epoch_loss": train_loss,
		"valid/epoch_loss": valid_loss,
	})


	train_losses=[]
	valid_losses=[]
	if valid_loss < best_epoch_loss:
		print("Validation loss decreases from {:.4f} to {:.4f}".format(best_epoch_loss, valid_loss))
		best_epoch_loss = valid_loss
		num_epoch_no_improvement = 0
		#save model
		torch.save({
			'epoch': epoch+1,
			'state_dict' : model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'best_epoch_loss': best_epoch_loss,
			'num_epoch_no_improvement': num_epoch_no_improvement,
			'scaler_state_dict': scaler.state_dict(),
		},os.path.join(conf.model_path, "Genesis_OCT_Best.pt"))
		print("Saving model ",os.path.join(conf.model_path,"Genesis_OCT_Best.pt"))
	else:
		print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_epoch_loss,num_epoch_no_improvement))
		num_epoch_no_improvement += 1
	# Save the latest checkpoint
	torch.save({
		'epoch': epoch+1,
		'state_dict' : model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'best_epoch_loss': best_epoch_loss,
		'num_epoch_no_improvement': num_epoch_no_improvement,
		'scaler_state_dict': scaler.state_dict(),
	},os.path.join(conf.model_path, "Genesis_OCT_Latest.pt"))
	print("Saving model ",os.path.join(conf.model_path,"Genesis_OCT_Latest.pt"))
	if num_epoch_no_improvement == conf.patience:
		print("Early Stopping")
		break
	sys.stdout.flush()
