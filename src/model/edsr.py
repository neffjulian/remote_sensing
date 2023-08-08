"""
Enhanced Deep Residual Networks for Single Image Super-Resolution (2017) by Lim et al.

Paper: https://arxiv.org/abs/1707.02921
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3)
        )

    def forward(self, x):
        return x + self.block(x) * 0.1

class EDSR(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]
        self.channels = hparams["model"]["channels"]
        self.nr_blocks = hparams["model"]["blocks"]
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)

        self.input_layer = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, self.channels, kernel_size=3)
        )
        
        residual_layers = [ResidualBlock(self.channels)] * self.nr_blocks
        self.residual_layers = nn.Sequential(*residual_layers)

        self.upscale = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(self.channels, self.channels * 6 * 6, kernel_size=3),
            nn.PixelShuffle(6),
            nn.ReplicationPad2d(1),
            nn.Conv2d(self.channels, self.channels, kernel_size=3)
        )

        self.output_layer = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(self.channels, 1, kernel_size=3)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.normal_(module.weight, mean=0, std=0.0001)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        x_hat = self.input_layer((x - mean) / std)
        x_hat = x_hat + self.residual_layers(x_hat) * 0.1
        x_hat = self.upscale(x_hat)
        return self.output_layer(x_hat) * std + mean
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': StepLR(
                    optimizer=optimizer,
                    step_size=self.scheduler_step,
                    gamma=0.5,
                ),
                "monitor": "val_ssim"
            }
        }

    def shared_step(self, batch, stage):
        lr_image, hr_image = batch
        sr_image = self.forward(lr_image)
        l1_loss = F.l1_loss(sr_image, hr_image)
        if stage == "val":
            mse_loss = F.mse_loss(sr_image, hr_image)
            self.log(f"{stage}_mse_loss", mse_loss, sync_dist=True)    
            self.log(f"{stage}_psnr", psnr(mse_loss), sync_dist=True)
            self.log(f"{stage}_ssim", self.ssim(sr_image, hr_image), sync_dist=True)
        return l1_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        lr_image, hr_image, names = batch
        sr_image = self.forward(lr_image)
        lr_image = F.interpolate(lr_image, size=(150, 150), mode='bicubic')
        return lr_image, sr_image, hr_image, names