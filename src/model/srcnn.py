"""
Super-Resolution Convolutional Neural Network (2015) by Dong et al.

Paper: https://arxiv.org/abs/1501.00092

@date: 2023-08-30
@author: Julian Neff, ETH Zurich

Copyright (C) 2023 Julian Neff

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

# SRCNN (Super-Resolution Convolutional Neural Network) implemented as a PyTorch Lightning Module
class SRCNN(LightningModule):
    # Initialize the model
    def __init__(self, hparams: dict):
        super().__init__()

        # Get hyperparameters
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]
        first_channel_size = hparams["model"]["channels"]

        # Compute the size of the second convolutional layer channels
        second_channel_size = first_channel_size // 2
        output_size = (150, 150)  # Output image size

        # Define the neural network architecture
        self.model = nn.Sequential(
            nn.Upsample(size=output_size, mode="bicubic"),
            nn.ReplicationPad2d(4),
            nn.Conv2d(1, first_channel_size, kernel_size=9),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(first_channel_size, second_channel_size, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(2),
            nn.Conv2d(second_channel_size, 1, kernel_size=5)
        )

        # Define SSIM (Structural Similarity Index Measure) metric
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)

        # Initialize the weights and biases
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.normal_(module.weight, mean=0, std=0.001)
                if module.bias is not None:
                    module.bias.data.zero_()

    # Forward pass
    def forward(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        return self.model((x - mean) / std) * std + mean
    
    # Configure the optimizer and the learning rate scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
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

    # Shared logic for training and validation steps
    def shared_step(self, batch, stage):
        lr_image, hr_image = batch
        sr_image = self.forward(lr_image)
        mse_loss = F.mse_loss(sr_image, hr_image)
        self.log(f"{stage}_mse_loss", mse_loss, sync_dist=True)
        
        # Log validation-specific metrics
        if stage == "val":
            self.log(f"{stage}_psnr", psnr(mse_loss), sync_dist=True)
            self.log(f"{stage}_ssim", self.ssim(sr_image, hr_image), sync_dist=True)
        return mse_loss

    # Training step
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    # Validation step
    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    # Prediction step
    def predict_step(self, batch, batch_idx):
        lr_image, hr_image, names = batch
        sr_image = self.forward(lr_image)
        lr_image = F.interpolate(lr_image, size=(150, 150), mode='bicubic')
        return lr_image, sr_image, hr_image, names