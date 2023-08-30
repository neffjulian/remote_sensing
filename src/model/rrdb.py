"""
RRDB: Residual in Residual Dense Block

Paper: https://arxiv.org/pdf/1809.00219.pdf

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
import pytorch_lightning as pl
from torchmetrics import StructuralSimilarityIndexMeasure

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block Class.

    This class is a custom layer that implements a Residual Dense Block (RDB).
    An RDB consists of several convolutional layers where each layer takes as input
    the concatenated outputs of all preceding layers and the original input.

    Attributes:
    ----------
    convs: nn.ModuleList
        A list of convolutional layers followed by activation functions.

    Parameters:
    ----------
    channels : int
        The number of input and output channels for the convolutional layers in this block.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass of the Residual Dense Block.
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize the ResidualDenseBlock with given channel size for convolution layers.
        """
        super().__init__()

        # Initialize an empty list to hold the convolutional layers.
        self.convs = nn.ModuleList()

        # Create 5 convolutional layers.
        for i in range(5):
            self.convs.append(
                nn.Sequential(
                    # Padding to maintain the spatial dimensions.
                    nn.ReplicationPad2d(1),

                    # Convolutional layer.
                    nn.Conv2d(channels + i * channels, channels, kernel_size=3),

                    # Activation function. LeakyReLU for the first 4 layers and Identity (no-op) for the last one.
                    nn.LeakyReLU(negative_slope=0.2) if i < 4 else nn.Identity()
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Residual Dense Block.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor.

        Returns:
        --------
        torch.Tensor
            The output tensor after passing through the Residual Dense Block.
        """
        # Initialize a list to store the output of each layer, starting with the input.
        features = [x]

        # Iterate through each convolutional layer.
        for conv in self.convs:
            # Concatenate the outputs of all preceding layers along with the original input.
            concatenated_features = torch.cat(features, dim=1)

            # Pass through the current convolutional layer.
            out = conv(concatenated_features)

            # Append the output to the list of outputs.
            features.append(out)

        # Apply a residual connection (with a scaling factor of 0.2) and return the output.
        return out * 0.2 + x
    
class ResidualInResidual(nn.Module):
    """
    Residual in Residual (RIR) Block Class.

    This class is a custom layer that encapsulates multiple Residual Dense Blocks
    (RDBs) within a larger residual structure. Each RDB modifies the input by a 
    small amount (scaled by 0.2), and these modifications are accumulated. The 
    final output is then formed by adding the accumulated modifications to the 
    original input (also scaled by 0.2).

    Attributes:
    ----------
    blocks: nn.ModuleList
        A list of Residual Dense Blocks (RDBs).

    Parameters:
    ----------
    blocks : int
        The number of Residual Dense Blocks (RDBs) to include in this block.
    channels : int
        The number of channels for each of the RDBs.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass of the Residual in Residual block.
    """

    def __init__(self, blocks: int, channels: int) -> None:
        """
        Initialize the ResidualInResidual block with a specified number of RDBs and channels.
        """
        super().__init__()

        # Initialize a list of `blocks` number of ResidualDenseBlocks.
        res_blocks = [ResidualDenseBlock(channels)] * blocks

        # Store the RDBs in a ModuleList so that PyTorch can recognize them as sub-modules.
        self.blocks = nn.ModuleList(res_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Residual in Residual Block.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor.

        Returns:
        --------
        torch.Tensor
            The output tensor after passing through the Residual in Residual block.
        """
        # Initialize `out` with the original input tensor.
        out = x

        # Iterate through each Residual Dense Block.
        for block in self.blocks:
            # Add the output of the RDB to `out`, scaled by 0.2.
            out += 0.2 * block(out)

        # Add the final `out` to the original input, scaled by 0.2, and return.
        return x + 0.2 * out
    
class RRDB(pl.LightningModule):
    """
    Residual in Residual Dense Block (RRDB) Network Class.

    This class extends the PyTorch LightningModule to define a custom neural network model 
    for image super-resolution. The model consists of a sequence of Conv2D, LeakyReLU, 
    PixelShuffle, and custom ResidualInResidual (RIR) blocks.

    Attributes:
    -----------
    lr : float
        Learning rate for the optimizer.
    scheduler_step : int
        Step for the learning rate scheduler.
    channels : int
        Number of channels in the ResidualInResidual blocks.
    ssim : StructuralSimilarityIndexMeasure
        Structural Similarity Index measure for evaluating the model.
    model : nn.Sequential
        The neural network model.
    """

    def __init__(self, hparams: dict) -> None:
        """
        Initialize the RRDB model with hyperparameters from a given dictionary.

        Parameters:
        -----------
        hparams : dict
            Dictionary containing all the necessary hyperparameters.
        """
        super().__init__()

        # Extract hyperparameters
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]
        self.channels = hparams["model"]["channels"]
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)

        # Define model parameters
        upscaling_factor = 6
        upscaling_channels = hparams["model"]["upscaling_channels"]
        blocks = hparams["model"]["blocks"]

        # Define the neural network model using nn.Sequential
        self.model = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, upscaling_factor * upscaling_factor * upscaling_channels, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.PixelShuffle(upscaling_factor),

            nn.ReplicationPad2d(1),
            nn.Conv2d(upscaling_channels, self.channels, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.2),

            ResidualInResidual(blocks, self.channels),

            nn.ReplicationPad2d(1),
            nn.Conv2d(self.channels, self.channels, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ReplicationPad2d(1),
            nn.Conv2d(self.channels, 1, kernel_size=3),
        )

    def forward(self, x):
        """
        Forward pass for the RRDB model.

        The input image tensor undergoes standardisation  before being fed into the model. 
        The output is then de-standardized before being returned.

        Parameters:
        -----------
        x : torch.Tensor
            The input image tensor.

        Returns:
        --------
        torch.Tensor
            The output image tensor after super-resolution.
        """
        mean = torch.mean(x)
        std = torch.std(x)
        return self.model((x - mean) / std) * std + mean
    
    # Configure Optimizers: Specifies the optimization strategy.
    def configure_optimizers(self):
        # Create an Adam optimizer for the model parameters with a specified learning rate.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        return {
            # Return the optimizer object.
            'optimizer': optimizer,
            
            # Configure learning rate scheduler.
            'lr_scheduler': {
                'scheduler': StepLR(
                    # Optimizer to update learning rate.
                    optimizer=optimizer,
                    
                    # Number of epochs after which to reduce the learning rate.
                    step_size=self.scheduler_step,
                    
                    # Factor to multiply the learning rate after 'step_size' epochs.
                    gamma=0.5,
                ),
                
                # Monitor validation SSIM for learning rate adjustment.
                'monitor': 'val_ssim'
            }
        }

    # Training Step: Performed during each epoch of the training process.
    def training_step(self, batch, batch_idx):
        # Unpack the batch to get low-resolution and high-resolution images.
        lr_image, hr_image = batch
        
        # Calculate L1 loss between the super-resolved image and high-resolution image.
        return F.l1_loss(self.forward(lr_image), hr_image)

    # Validation Step: Performed during model validation.
    def validation_step(self, batch, batch_idx):
        # Unpack the batch to get low-resolution and high-resolution images.
        lr_image, hr_image = batch
        
        # Generate the super-resolved image.
        sr_image = self.forward(lr_image)
        
        # Compute MSE loss between super-resolved and high-resolution images.
        mse_loss = F.mse_loss(sr_image, hr_image)
        
        # Log metrics for analysis.
        self.log(f"mse", mse_loss, sync_dist=True)
        self.log(f"psnr", psnr(mse_loss), sync_dist=True)
        self.log(f"ssim", self.ssim(sr_image, hr_image), sync_dist=True)
        
        # Return the L1 loss for further analysis or optimization.
        return F.l1_loss(sr_image, hr_image)

    # Predict Step: Used for generating super-resolved images for a given batch.
    def predict_step(self, batch, batch_idx):
        # Unpack the batch to get low-resolution and high-resolution images, and their names.
        lr_image, hr_image, names = batch
        
        # Upscale the low-resolution image to the size of (150, 150) for comparison.
        upscaled_lr_image = F.interpolate(lr_image, size=(150, 150), mode='bicubic')
        
        # Generate super-resolved image.
        sr_image = self.forward(lr_image)
        
        # Return the upscaled low-resolution, super-resolved, and high-resolution images and their names.
        return upscaled_lr_image, sr_image, hr_image, names