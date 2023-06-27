"""
Super-Resolution Convolutional Neural Network (2015) by Dong et al.

Paper: https://arxiv.org/abs/1501.00092
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from model.losses import psnr

class SRCNN(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]
        self.scheduler = hparams["scheduler"]
        first_channel_size = hparams["model"]["channels"]
        second_channel_size = int(first_channel_size/2)

        self.l1 = nn.Conv2d(1, first_channel_size, kernel_size=9, padding=4, padding_mode="replicate")
        self.l2 = nn.Conv2d(first_channel_size, second_channel_size, kernel_size=3, padding=1, padding_mode="replicate")
        self.l3 = nn.Conv2d(second_channel_size, 1, kernel_size=5, padding=2, padding_mode="replicate")

        self.relu = nn.ReLU(inplace=True)
        self.mse = nn.MSELoss()

        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)

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
                'scheduler': MultiStepLR(
                    optimizer=optimizer,
                    milestones=[self.scheduler_step],
                    gamma=0.1
                )
            }
        }

    def shared_step(self, batch, stage):
        lr_image, hr_image = batch
        sr_image = self.forward(lr_image)
        mse_loss = self.mse(sr_image, hr_image)
        self.log(f"{stage}_mse_loss", mse_loss, sync_dist=True)        
        if stage == "val":
            self.log(f"{stage}_psnr", psnr(mse_loss), sync_dist=True)
            self.log(f"{stage}_ssim", self.ssim(sr_image, hr_image), sync_dist=True)
        return mse_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        lr_image, hr_image, names = batch
        sr_image = self.forward(lr_image)
        return lr_image, sr_image, hr_image, names