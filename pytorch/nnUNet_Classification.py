import argparse
import torch
from utils import KFoldNNUNetTabularDataModule, nnUNetRegressionClassification
from utils import *
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
import wandb
import yaml


def main(config=None):
    assert config is not None, 'Please provide a config file'

    L.seed_everything(config['seed'], workers=True)



    # Data
    dm = KFoldNNUNetTabularDataModule(config=config)
    dm.setup('fit')
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # print(model) # 19073665 # 30785994 


    # Model
    model = nnUNetRegressionClassification(config=config)
    # model.requires_grad_(True)
    for name, param in model.named_parameters():
        # Check if the parameter has requires_grad
        if not param.requires_grad:
            print(f'Parameter {name} is not requires_grad')

    # Logger
    if config['wandb']['wandb_run_id'] == None:
        config['wandb']['wandb_run_id'] = wandb.util.generate_id()
    if config['wandb']['wandb_run_name'] == None:
        config['wandb']['wandb_run_name'] = f'fold_{config["data"]["fold"]}'

    Path(config['wandb']['logs_path']).mkdir(exist_ok=True)
    wandb_logger = WandbLogger(
        id=config['wandb']['wandb_run_id'],
        project=config['wandb']['wandb_project_name'],
        name=config['wandb']['wandb_run_name'],
        config=config,
        dir=config['wandb']['logs_path'],
        save_dir=config['wandb']['logs_path'],
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=config['train']['max_epochs'],
        deterministic=True,
        precision="16-mixed",
        logger=wandb_logger,
        default_root_dir=Path(config['wandb']['logs_path']) / f'fold_{config["data"]["fold"]}',
        callbacks=[
            ModelCheckpoint(dirpath=Path(config['wandb']['logs_path']) / f'fold_{config["data"]["fold"]}', monitor="val_loss", mode="min", save_top_k=1, save_last=True, verbose=True, filename='best_model'),
            ModelCheckpoint(dirpath=Path(config['wandb']['logs_path']) / f'fold_{config["data"]["fold"]}', filename="last_model"),
            EarlyStopping(monitor="val_loss", mode="min", patience=config['train']['patience'], verbose=True),
            LearningRateMonitor(logging_interval='step', log_momentum=True, log_weight_decay=True)
        ],
        )
    trainer.fit(model, train_loader, val_loader)

    dm.setup('test')
    test_loader = dm.test_dataloader()
    trainer.test(dataloaders=test_loader, ckpt_path="best")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--sweep', type=bool, default=False)
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if args.sweep:
            raise NotImplementedError('Sweep is not implemented yet')
        else:
            main(config)
    else:
        print('Please provide a config file')

