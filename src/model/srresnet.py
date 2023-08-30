"""
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (2017) by Ledig et al.

Paper: https://arxiv.org/abs/1609.04802

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
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

class ResidualBlock(nn.Sequential):
    """
    Residual block module.

    Args:
        channels (int): Number of input and output channels.

    Returns:
        torch.Tensor: Output tensor with the same shape as the input tensor.
    """

    def __init__(self, channels: int):
        """
        Initializes the ResidualBlock module.

        Args:
            channels (int): Number of input and output channels.
        """
        super(ResidualBlock, self).__init__()

        # Create a block containing a convolution layer, batch normalization, and PReLU activation
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )

    def forward(self, x):
        """
        Forward pass of the ResidualBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input tensor.
        """
        # Add the output of the block to the input tensor
        return self.block(x) + x

class SRResNet(LightningModule):
    # Docstring:
    """
    Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (2017) by Ledig et al.

    Paper: https://arxiv.org/abs/1609.04802

    Attributes:
        batch_size (int): Batch size.
        lr (float): Learning rate.
        scheduler_step (int): Number of epochs before reducing the learning rate.
        ssim (torchmetrics.SSIM): Structural Similarity Index Measure.
        channels (int): Number of channels in the input image.
        nr_blocks (int): Number of residual blocks.
        mse (torch.nn.MSELoss): Mean Squared Error loss.
        input_layer (torch.nn.Sequential): Input layer.
        body (torch.nn.Sequential): Body of the network.
        last_layer (torch.nn.Sequential): Last layer of the network.
        output_layer (torch.nn.Sequential): Output layer.
    """

    def __init__(self, hparams: dict):
        super().__init__()

        # Save hyperparameters
        self.batch_size = hparams["model"]["batch_size"]
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]

        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)
        self.channels = hparams["model"]["channels"]
        self.nr_blocks = hparams["model"]["blocks"]
        self.mse = nn.MSELoss()
        
        # Create the network
        self.input_layer = nn.Sequential(
            nn.Conv2d(1, self.channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.PReLU()
        )
        
        # Create the body of the network
        blocks = [ResidualBlock(self.channels)] * self.nr_blocks
        self.body = nn.Sequential(*blocks)

        # Create the last layer of the network
        self.last_layer = nn.Sequential(
            nn.Conv2d(self.channels, self.channels * 4, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.PReLU(),
            nn.Conv2d(self.channels * 4, self.channels, kernel_size=3, padding=1, padding_mode="replicate"),
        )

        # Create the output layer of the network
        self.output_layer = nn.Sequential(
            nn.Conv2d(self.channels, 1, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.LeakyReLU(0.2)
        )

    # Forward pass of the network
    def forward(self, x):
        mean = torch.mean(x)
        std = torch.std(x)
        x_hat = self.input_layer((x - mean) / std)
        x_body = self.body(x_hat)

        return self.output_layer(x_hat + self.last_layer(x_body)) * std + mean
    
    # Configure the optimizer
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

    # Shared step between training and validation
    def shared_step(self, batch, stage):
        lr_image, hr_image = batch
        sr_image = self.forward(lr_image)
        l1_loss = F.smooth_l1_loss(sr_image, hr_image)     

        self.log(f"{stage}_l1_loss", l1_loss, sync_dist=True)
        if stage == "val":
            self.log(f"{stage}_psnr", psnr(F.mse_loss(sr_image, hr_image)), sync_dist=True)
            self.log(f"{stage}_ssim", self.ssim(sr_image, hr_image), sync_dist=True)
        return l1_loss

    # Training Step: Performed during each epoch of the training process.
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    # Validation Step: Performed during model validation.
    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    # Predict Step: Used for generating super-resolved images for a given batch.
    def predict_step(self, batch, batch_idx):
        lr_image, hr_image, names = batch
        sr_image = self.forward(lr_image)
        return lr_image, sr_image, hr_image, names