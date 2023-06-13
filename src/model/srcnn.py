"""
Super-Resolution Convolutional Neural Network (2015)

Paper: https://arxiv.org/abs/1501.00092
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule

torch.set_float32_matmul_precision("medium")

def psnr(y_hat, y):
    mse = torch.mean((y_hat - y) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

class SRCNN(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.batch_size = hparams["model"]["batch_size"]
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler = hparams["scheduler"]

        self.l1 = nn.Conv2d(1, 256, kernel_size=9, padding=4)
        self.l2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.l3 = nn.Conv2d(128, 1, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.l1, self.l2, self.l3]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        y_l1 = F.relu(self.l1(x))
        y_l2 = F.relu(self.l2(y_l1))
        y_l3 = F.relu(self.l3(y_l2))
        return y_l3
    
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    scheduler = {
        'scheduler': ReduceLROnPlateau(
            optimizer=optimizer,
            patience=self.scheduler["patience"],
            min_lr=self.scheduler["min_lr"],
            verbose=self.scheduler["verbose"]
        ),
        'interval': 'epoch',
        'frequency': 1,
        'strict': True
    }
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_psnr_loss'
        }
    }

def shared_step(self, batch, stage):
    x, y = batch
    y_hat = self.forward(x)

    rmse_loss = torch.sqrt(F.mse_loss(y_hat, y))

    if stage == "val":
        l1_loss = F.l1_loss(y_hat, y)
        psnr_loss = psnr(y_hat, y)
        
        self.log(f"{stage}_l1_loss", l1_loss)
        self.log(f"{stage}_psnr_loss", psnr_loss)
        self.log(f"{stage}_rmse_loss", rmse_loss)

    return rmse_loss

def shared_epoch_end(self, outputs, stage):
    self.log_dict

def training_step(self, batch, batch_idx):
    return self.shared_step(batch, "train")

def validation_step(self, batch, batch_idx):
    return self.shared_step(batch, "val")

def predict_step(self, batch, batch_idx):
    x, y, names = batch
    y_hat = self.forward(x)
    return x, y_hat, y, names