"""
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (2017) by Ledig et al.

Paper: https://arxiv.org/abs/1609.04802
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure

from losses import psnr

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")


class ResidualBlock(nn.Sequential):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )        

    def forward(self, x):
        return self.block(x) + x

class SRResNet(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.batch_size = hparams["model"]["batch_size"]
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]

        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)
        self.channels = hparams["model"]["channels"]
        self.nr_blocks = hparams["model"]["blocks"]
        self.mse = nn.MSELoss()
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, self.channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.PReLU()
        )
        
        blocks = [ResidualBlock(self.channels)] * self.nr_blocks
        self.body = nn.Sequential(*blocks)

        self.last_layer = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * 4, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.PReLU(),
            nn.Conv2d(self.channels * 4, self.channels, kernel_size=3, padding=1, padding_mode="replicate"),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(self.channels, 1, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x_hat = self.input_layer(x)
        x_body = self.body(x_hat)

        return self.output_layer(x_hat + self.last_layer(x_body))
    
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
        l1_loss = F.smooth_l1_loss(sr_image, hr_image)     

        self.log(f"{stage}_l1_loss", l1_loss, sync_dist=True)
        if stage == "val":
            self.log(f"{stage}_psnr", psnr(F.mse_loss(sr_image, hr_image)), sync_dist=True)
            self.log(f"{stage}_ssim", self.ssim(sr_image, hr_image), sync_dist=True)
        return l1_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        lr_image, hr_image, names = batch
        sr_image = self.forward(lr_image)
        return lr_image, sr_image, hr_image, names