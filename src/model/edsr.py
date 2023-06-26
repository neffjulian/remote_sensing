"""
Enhanced Deep Residual Networks for Single Image Super-Resolution (2017) by Lim et al.

Paper: https://arxiv.org/abs/1707.02921
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure
from model.losses import psnr

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

class EDSR(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.batch_size = hparams["model"]["batch_size"]
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]
        self.scheduler = hparams["scheduler"]
        self.channels = hparams["model"]["channels"]
        self.nr_blocks = hparams["model"]["blocks"]

        self.input_layer = nn.Conv2d(1, self.channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.output_layer = nn.Conv2d(self.channels, 1, kernel_size=3, padding=1, padding_mode="replicate")
        
        residual_layers = [
            nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, padding_mode="replicate"),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, padding_mode="replicate")
            )
        ] * self.nr_blocks

        self.residual_layers = nn.Sequential(*residual_layers)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)

    def forward(self, x):
        x_hat = self.input_layer(x)
        return self.output_layer(x_hat + self.residual_layers(x_hat))
    
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
        x, y = batch
        y_hat = self.forward(x)
        l1_loss = F.l1_loss(y_hat, y)     
        mse_loss = nn.MSELoss(y_hat, y)

        self.log(f"{stage}_l1_loss", l1_loss, sync_dist=True)
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