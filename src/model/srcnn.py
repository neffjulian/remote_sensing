"""
Super-Resolution Convolutional Neural Network (2015) by Dong et al.

Paper: https://arxiv.org/abs/1501.00092
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from model.losses import mse, psnr

class SRCNN(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.batch_size = hparams["model"]["batch_size"]
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler = hparams["scheduler"]
        first_channel_size = hparams["model"]["channels"]
        second_channel_size = int(first_channel_size/2)

        self.l1 = nn.Conv2d(1, first_channel_size, kernel_size=9, padding=1, padding_mode="replicate")
        self.l2 = nn.Conv2d(first_channel_size, second_channel_size, kernel_size=3, padding=1, padding_mode="replicate")
        self.l3 = nn.Conv2d(second_channel_size, 1, kernel_size=5, padding=2, padding_mode="replicate")

        self.relu = nn.ReLU(inplace=True)

        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.l1, self.l2, self.l3]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            # nn.init.kaiming_normal_(layer.weight, a=.1)
            nn.init.constant_(layer.bias, 0)

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
                ),
                'monitor': 'val_mse_loss'
            }
        }

    def shared_step(self, batch, stage):
        x, y = batch
        y_hat = self.forward(x)
        mse_loss = mse(y_hat, y)
        self.log(f"{stage}_mse_loss", mse_loss, sync_dist=True)        
        if stage == "val":
            self.log(f"{stage}_psnr", psnr(mse_loss), sync_dist=True)
            self.log(f"{stage}_ssim", self.ssim(y_hat, y), sync_dist=True)
        return mse_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        x, y, names = batch
        y_hat = self.forward(x)
        return x, y_hat, y, names