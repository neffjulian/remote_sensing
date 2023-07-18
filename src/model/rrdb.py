"""
RRDB: Residual in Residual Dense Block

Paper: https://arxiv.org/pdf/1809.00219.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from torchmetrics import StructuralSimilarityIndexMeasure

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, residual_scaling: float = 0.2) -> None:
        super().__init__()
        self.residual_scaling = residual_scaling

        self.convs = nn.ModuleList()
        for i in range(5):
            self.convs.append(
                nn.Sequential(
                    nn.ReplicationPad2d(1),
                    nn.Conv2d(channels + i * channels, channels, kernel_size=3),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True) if i < 4 else nn.Identity()
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for conv in self.convs:
            out = conv(torch.cat(features, dim=1))
            features.append(out)
        return out * self.residual_scaling + x
    
class ResidualInResidual(nn.Module):
    def __init__(self, blocks: int, channels: int, residual_scaling: float = 0.2) -> None:
        self.blocks = [ResidualDenseBlock(channels)] * blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out += self.residual_scaling * block(out)
        return x + self.residual_scaling * out
    
class RRDB(pl.LightningModule):
    def __init__(self, hparams: dict) -> None:
        super().__init__()

        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]

        self.channels = hparams["model"]["channels"]
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)

        upscaling_factor = 6
        upscaling_channels = 32

        self.upsample = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, upscaling_factor * upscaling_factor * upscaling_channels, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.PixelShuffle(upscaling_factor),
            nn.ReplicationPad2d(1),
            nn.Conv2d(upscaling_channels, self.channels, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )


        self.blocks = ResidualInResidual(16, self.channels)
        
        self.out = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(self.channels, 1, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.upsample(x)))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': StepLR(
                    optimizer=optimizer,
                    step_size=self.scheduler_step,
                    gamma=0.5,
                    verbose=True
                ),
                'monitor': 'val_ssim'
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
        lr_image, hr_image = batch
        sr_image = self.forward(lr_image)
        return F.l1_loss(sr_image, hr_image)
    
    def validation_step(self, batch, batch_idx):
        lr_image, hr_image = batch
        sr_image = self.forward(lr_image)
        mse_loss = F.mse_loss(sr_image, hr_image)
        self.log(f"mse", mse_loss, sync_dist=True)     
        self.log(f"psnr", psnr(mse_loss), sync_dist=True)
        self.log(f"ssim", self.ssim(sr_image, hr_image), sync_dist=True)
        return F.l1_loss(sr_image, hr_image)

    def predict_step(self, batch, batch_idx):
        lr_image, hr_image, names = batch
        sr_image = self.forward(lr_image)
        lr_image = F.interpolate(lr_image, size=(150, 150), mode='bicubic')
        return lr_image, sr_image, hr_image, names