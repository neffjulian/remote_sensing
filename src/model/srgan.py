"""
 SRGAN: a generative adversarial network (GAN) for image superresolution (SR).

Paper: https://arxiv.org/pdf/1609.04802.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

from model.losses import psnr

class Block(nn.Sequential):
    def __init__(self, channel_in: int, channel_out: int, stride: int):
        super(Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, padding_mode="replicate", stride=stride),
            nn.BatchNorm2d(channel_out),
            nn.LeakyReLU(negative_slope=0.2)
        )        

    def forward(self, x):
        return self.block(x)

class Discriminator(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.batch_size = hparams["model"]["batch_size"]
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler = hparams["scheduler"]
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)

        self.mse = nn.MSELoss()
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.2)
        )
        
        blocks = [
            Block(64, 128, 2),
            Block(128, 128, 1),
            Block(128, 256, 2),
            Block(256, 256, 1),
            Block(256, 512, 1),
            Block(512, 512, 2)
        ]
        self.body = nn.Sequential(*blocks)

        self.output_layer = nn.Sequential(
            nn.Linear(1025, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_hat = self.input_layer(x)
        x_hat = self.body(x_hat)
        return self.output_layer(x_hat)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(
                    optimizer=optimizer
                ),
                'monitor': 'val_l1_loss'
            }
        }

    def shared_step(self, batch, stage):
        x, y = batch
        y_hat = self.forward(x)
        l1_loss = F.smooth_l1_loss(y_hat, y)     

        mse_loss = self.mse(y_hat, y)
        self.log(f"{stage}_l1_loss", l1_loss, sync_dist=True)
        self.log(f"{stage}_mse_loss", mse_loss, sync_dist=True)        
        if stage == "val":
            self.log(f"{stage}_psnr", psnr(mse_loss), sync_dist=True)
            self.log(f"{stage}_ssim", self.ssim(y_hat, y), sync_dist=True)
        return l1_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        x, y, names = batch
        y_hat = self.forward(x)
        return x, y_hat, y, names