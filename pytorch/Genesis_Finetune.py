import argparse
import torch
from utils import KFoldNNUNetSegmentationDataModule, GenesisSegmentation
from utils import *
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import wandb
import yaml


def main(config=None):
    assert config is not None, 'Please provide a config file'

    L.seed_everything(config['seed'], workers=True)



    # Data
    dm = KFoldNNUNetSegmentationDataModule(fold=config['fold'], dataDir=config['data_directory'])
    dm.setup('fit')
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # print(model) # 19073665 # 30785994 


    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = GenesisSegmentation(config=config)

    # Logger
    if config['wandb']['wandb_run_id'] == None:
        config['wandb']['wandb_run_id'] = wandb.util.generate_id()
    if config['wandb']['wandb_run_name'] == None:
        config['wandb']['wandb_run_name'] = f'fold_{config["fold"]}'
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
        max_epochs=1,
        deterministic=True,
        precision="16-mixed",
        logger=wandb_logger,
        default_root_dir=Path(config['wandb']['logs_path']) / f'fold_{config["fold"]}',
        callbacks=[
            ModelCheckpoint(dirpath=Path(config['wandb']['logs_path']) / f'fold_{config["fold"]}', monitor="val_loss", mode="min", save_top_k=1, save_last=True, verbose=True, filename='best_model-{val_loss:.2f}'),
            ModelCheckpoint(dirpath=Path(config['wandb']['logs_path']) / f'fold_{config["fold"]}', filename="last_model"),
            EarlyStopping(monitor="val_loss", mode="min", patience=50, verbose=True)
        ],
        )
    trainer.fit(model, train_loader, val_loader)

    dm.setup('test')
    test_loader, test_grid_samplers = dm.test_dataloader()
    model.set_test_grid_samplers(test_grid_samplers)
    trainer.test(model, test_loader, ckpt_path="best")
    # for epoch in range(intial_epoch, config.nb_epoch):
    #     model.train()
    #     sum_train_loss = 0.0
    #     count_train_loss = 0
    #     for batch_ndx, data in enumerate(train_loader):
    #         with torch.autocast(device_type="cuda", dtype=torch.float16):
    #             x, y = data['image'][tio.DATA], data['label'][tio.DATA]
    #             x, y = x.float().to(device), y.float().to(device)
    #             pred = model(x)
    #             pred = pred.sigmoid()
    #             loss = criterion(y, pred)
            
    #         scaler.scale(loss).backward()
    #         scaler.step(optimizer)
    #         scaler.update()
    #         optimizer.zero_grad(set_to_none=True)
    #         sum_train_loss += loss.item()
    #         count_train_loss += 1

    #     if count_train_loss == 0:
    #         count_train_loss = 1
    #     avg_train_loss = sum_train_loss / count_train_loss
    #     wandb.log({'train/loss': avg_train_loss})
    #     print(f'Epoch [{epoch + 1}/{config.nb_epoch}], Loss: {avg_train_loss}')
    #     scheduler.step()

    #     model.eval()
    #     sum_valid_loss = 0.0
    #     count_valid_loss = 0
    #     with torch.no_grad():
    #         for batch_ndx, data in enumerate(val_loader):
    #             with torch.autocast(device_type="cuda", dtype=torch.float16):
    #                 x, y = data['image'][tio.DATA], data['label'][tio.DATA]
    #                 x, y = x.float().to(device), y.float().to(device)
    #                 pred = model(x)
    #                 pred = pred.sigmoid()
    #                 loss = criterion(y, pred)
    #             sum_valid_loss += loss.item()
    #             count_valid_loss += 1
            
    #         wandb.log({'val/loss': sum_valid_loss / count_valid_loss})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        main(config)
    else:
        print('Please provide a config file')