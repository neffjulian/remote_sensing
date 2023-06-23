"""
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (2017) by Ledig et al.

Paper: https://arxiv.org/abs/1609.04802
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

def mse(y_hat, y):
    return torch.mean((y_hat - y) ** 2)

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

class ResidualBlock(nn.Sequential):
    def __init__(self, channels: int, last: bool):
        super(ResidualBlock, self).__init__()
        out_channels = channels if last is False else channels * 4
        self.last = last

        self.block = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2)
        )        

    def forward(self, x):
        if self.last:
            return self.block(x)
        else:
            return self.block(x) + x

class SRResNet(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.batch_size = hparams["model"]["batch_size"]
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler = hparams["scheduler"]
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)
        self.channels = hparams["model"]["channels"]
        self.nr_blocks = hparams["model"]["blocks"]
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, self.channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.2)
        )
        
        blocks = [ResidualBlock(self.channels, False)] * (self.nr_blocks - 1) + [ResidualBlock(self.channels, True)]
        self.body = nn.Sequential(*blocks)

        self.output_layer = nn.Sequential(
            nn.Conv2d(self.channels * 4, 1, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.2)
        )


    def forward(self, x):
        x_hat = self.input_layer(x)
        return self.output_layer(x_hat + self.body(x_hat))
    
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

        mse_loss = mse(y_hat, y)
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