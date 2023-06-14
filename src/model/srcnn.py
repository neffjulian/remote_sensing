"""
Super-Resolution Convolutional Neural Network (2015)

Paper: https://arxiv.org/abs/1501.00092
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule

# torch.set_float32_matmul_precision("medium")

def psnr(y_hat, y):
    mse = torch.mean((y_hat - y) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

class SRCNN(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.batch_size = hparams["model"]["batch_size"]
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler = hparams["scheduler"]

        self.l1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.l2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.l3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(
                    optimizer=optimizer,
                    patience=self.scheduler["patience"],
                    min_lr=self.scheduler["min_lr"],
                    verbose=self.scheduler["verbose"]
                ),
                'monitor': 'val_l1_loss',
                'frequency': 5
            }
        }

    def shared_step(self, batch, stage):
        x, y = batch
        y_hat = self.forward(x)
        mse_loss = F.mse_loss(y_hat, y)
        l1_loss = F.l1_loss(y_hat, y)
        self.log(f"{stage}_mse_loss", mse_loss, sync_dist=True)        
        self.log(f"{stage}_l1_loss", l1_loss, sync_dist=True)        

        if stage == "val":
            rmse_loss = torch.sqrt(F.mse_loss(y_hat, y))
            self.log(f"{stage}_rmse_loss", rmse_loss, sync_dist=True)

            psnr_loss = psnr(y_hat, y)
            self.log(f"{stage}_psnr_loss", psnr_loss, sync_dist=True)

        return mse_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        x, y, names = batch
        y_hat = self.forward(x)
        return x, y_hat, y, names