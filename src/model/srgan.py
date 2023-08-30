"""
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (2017) by Ledig et al.

Paper: https://arxiv.org/abs/1609.04802
Adpted from: https://github.com/https-deeplearning-ai/GANs-Public/blob/master/C3W2_SRGAN_(Optional).ipynb
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

from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from torchmetrics import StructuralSimilarityIndexMeasure

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))
class VGG19FeatureExtractor(nn.Module):
    """
    VGG19 Feature Extractor Class.

    This class extends the PyTorch nn.Module to define a custom feature extractor 
    based on the VGG19 architecture. It uses pretrained weights and omits the last layer 
    of the original VGG19's features sub-network.

    Attributes:
    -----------
    vgg : nn.Sequential
        The feature extraction model based on VGG19 architecture.
    """

    def __init__(self) -> None:
        """
        Initialize the VGG19FeatureExtractor model.
        
        The model is set to evaluation mode and gradients are turned off for all parameters.
        """
        super().__init__()
        
        # Load the pretrained VGG19 model
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        
        # Use only the feature extraction part of VGG19 and remove the last layer
        self.vgg = nn.Sequential(*list(vgg.features)[:-1]).eval()
        
        # Turn off gradients for VGG19 as we're only using it for feature extraction
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the VGG19 feature extraction.
        
        The input tensor is expected to have one channel. It is replicated to have 
        three channels before being fed into the VGG19 model.

        Parameters:
        -----------
        x : torch.Tensor
            The input image tensor with one channel.

        Returns:
        --------
        torch.Tensor
            The output feature map.
        """
        # Duplicate single-channel image to have three channels because input usually is RGB
        return self.vgg(x.repeat(1, 3, 1, 1))

class ResidualBlock(nn.Module):
    """
    Defines a residual block module for a neural network.
    
    Attributes:
    - block (nn.Sequential): A sequence of layers that form the residual block.
    """
    def __init__(self, feature_maps: int = 64) -> None:
        """
        Initialize the ResidualBlock.
        
        Parameters:
        - feature_maps (int, optional): The number of feature maps for the Conv2D layers.
                                        Defaults to 64.
        """
        super().__init__()
        
        # Define the residual block as a sequence of layers
        self.block = nn.Sequential(
            nn.ReplicationPad2d(1),  # Padding
            nn.Conv2d(feature_maps, feature_maps, kernel_size=3),  # Convolution
            nn.BatchNorm2d(feature_maps),  # Batch Normalization
            nn.PReLU(),  # Activation
            nn.ReplicationPad2d(1),  # Padding
            nn.Conv2d(feature_maps, feature_maps, kernel_size=3),  # Convolution
            nn.BatchNorm2d(feature_maps),  # Batch Normalization
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the residual block.
        
        Parameters:
        - x (torch.Tensor): The input tensor.
        
        Returns:
        - torch.Tensor: The output tensor, which is the sum of the input and the processed input.
        """
        # Element-wise sum of the input and the processed input
        return x + self.block(x)

class Generator(nn.Module):
    """
    Generator class for Super-Resolution using Convolutional Neural Networks (SRResNet).
    
    Parameters:
    - feature_maps (int): Number of feature maps. Default is 64.
    - num_res_blocks (int): Number of residual blocks. Default is 16.
    """
    
    def __init__(self, feature_maps: int = 64, num_res_blocks: int = 16) -> None:
        super().__init__()
        
        # Initial convolution block
        self.input_block = nn.Sequential(
            nn.ReplicationPad2d(4),
            nn.Conv2d(1, feature_maps, kernel_size=9),
            nn.PReLU()
        )
        
        # Create a list of residual blocks
        residual_blocks = [ResidualBlock(feature_maps)] * num_res_blocks
        
        # Additional layers for the residual blocks
        residual_blocks += [
            nn.ReplicationPad2d(1),
            nn.Conv2d(feature_maps, feature_maps, kernel_size=3),
            nn.BatchNorm2d(feature_maps)
        ]
        
        # Make it a sequential model
        self.residual_blocks = nn.Sequential(*residual_blocks)
        
        # Upscaling block
        self.upscale_block = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(feature_maps, feature_maps * 108, kernel_size=3),
            nn.PixelShuffle(6),
            nn.PReLU(),
        )
        
        # Output convolution block
        self.output_block = nn.Sequential(
            nn.ReplicationPad2d(4),
            nn.Conv2d(feature_maps * 3, 1, kernel_size=9),
        )
        
        # Initialize weights and biases
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.normal_(module.weight, mean=0, std=0.0001)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Parameters:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor.
        """
        
        x_input = self.input_block(x)
        
        # Add the residual blocks to the initial block
        x = x_input + self.residual_blocks(x_input)
        
        # Upscale
        x = self.upscale_block(x)
        
        # Final output block
        x = self.output_block(x)
        
        return x

class Discriminator(nn.Module):
    """
    Discriminator Class for a Generative Adversarial Network (GAN).

    This class is a PyTorch module that defines the architecture of the Discriminator. 
    It uses multiple convolutional layers followed by an MLP (Multilayer Perceptron) 
    to output the discriminator's prediction.

    Attributes:
    -----------
    conv_blocks : nn.Sequential
        Sequence of convolutional blocks that act as feature extractors.
    mlp : nn.Sequential
        Multilayer Perceptron to output the final discriminator prediction.
    """

    def __init__(self, feature_maps: int = 64) -> None:
        """
        Initialize the Discriminator model.

        Initializes the convolutional and MLP blocks of the Discriminator.
        """
        super().__init__()

        # Define the convolutional blocks
        self.conv_blocks = nn.Sequential(
            self._make_double_conv_block(1, feature_maps, first_batch_norm=False),
            self._make_double_conv_block(feature_maps, feature_maps * 2),
            self._make_double_conv_block(feature_maps * 2, feature_maps * 4),
            self._make_double_conv_block(feature_maps * 4, feature_maps * 8)
        )

        # Define the MLP that outputs the final prediction
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_maps * 8, feature_maps * 16, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feature_maps * 16, 1, kernel_size=1),
            nn.Flatten()
        )

        # Initialize weights and biases
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.normal_(module.weight, mean=0, std=0.001)
                if module.bias is not None:
                    module.bias.data.zero_()

    def _make_double_conv_block(self, in_channels: int, out_channels: int, first_batch_norm: bool = True) -> nn.Sequential:
        """
        Creates a block containing two convolutional layers.
        """
        return nn.Sequential(
            self._make_conv_block(in_channels, out_channels, batch_norm=first_batch_norm),
            self._make_conv_block(out_channels, out_channels, stride=2),
        )

    def _make_conv_block(self, in_channels: int, out_channels: int, stride: int = 1, batch_norm: bool = True) -> nn.Sequential:
        """
        Creates a single convolutional block.
        """
        return nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Discriminator model.

        Applies feature extraction with convolutional blocks followed by the MLP
        to output the final prediction.

        Parameters:
        -----------
        x : torch.Tensor
            The input image tensor.

        Returns:
        --------
        torch.Tensor
            The output prediction tensor.
        """
        # Normalize the input
        mean = torch.mean(x)
        std = torch.std(x)
        x = self.conv_blocks((x - mean) / std)
        
        # Apply MLP and return the output
        return self.mlp(x) * std + mean

class SRGAN(pl.LightningModule):
    """
    Super-resolution generative adversarial network (SRGAN) module.
    """

    def __init__(self, hparams: dict) -> None:
        """
        Initializes the SRGAN module.

        Args:
            hparams (dict): Hyperparameters for the SRGAN module.
        """
        super().__init__()

        # Set the learning rate and scheduler step from hparams
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]

        # Initialize the generator and discriminator
        self.generator = Generator(hparams["model"]["feature_maps_gen"], hparams["model"]["num_res_blocks"])
        self.discriminator = Discriminator(hparams["model"]["feature_maps_disc"])

        # Initialize the feature extractor and SSIM measure
        self.feature_extractor = VGG19FeatureExtractor()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)

    # Define optimizer and scheduler configurations
    def configure_optimizers(self) -> Tuple:
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        sched_gen = torch.optim.lr_scheduler.StepLR(opt_gen, step_size=self.scheduler_step, gamma=0.5)
        sched_disc = torch.optim.lr_scheduler.StepLR(opt_disc, step_size=self.scheduler_step, gamma=0.5)

        return [opt_gen, opt_disc], [sched_gen, sched_disc]

    # Forward pass through the generator
    def forward(self, lr_image: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(lr_image)
        std = torch.std(lr_image)
        return self.generator((lr_image - mean) / std) * std + mean

    # Training loop for both generator and discriminator
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, optimizer_idx: int) -> None:
        lr_image, hr_image = batch

        loss = None
        if optimizer_idx == 0:
            loss = self._generator_loss(lr_image, hr_image)
        if optimizer_idx == 1:
            loss = self._discriminator_loss(lr_image, hr_image)
        return loss
    
    # Validation Step: Performed during model validation.
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        lr_image, hr_image = batch
        sr_image = self(lr_image)

        loss_gen = self._generator_loss(lr_image, hr_image)
        loss_disc = self._discriminator_loss(lr_image, hr_image)

        self.log("val_loss_gen", loss_gen, on_epoch=True, sync_dist=True)
        self.log("val_loss_disc", loss_disc, on_epoch=True, sync_dist=True)

        mse_loss = F.mse_loss(sr_image, hr_image)
        self.log("val_psnr", psnr(mse_loss), on_epoch=True, sync_dist=True)
        self.log("val_ssim", self.ssim(sr_image, hr_image), on_epoch=True, sync_dist=True)

    # Prediction Step: Used for generating super-resolved images for a given batch.
    def predict_step(self, batch, batch_idx):
        lr_image, hr_image, names = batch
        sr_image = self(lr_image)
        lr_image = F.interpolate(lr_image, size=(150, 150), mode='bicubic')
        return lr_image, sr_image, hr_image, names
    
    # Generate fake prediction: Produces both the super-resolved image and its discriminator prediction.
    def _fake_pred(self, lr_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fake = self(lr_image)
        fake_pred = self.discriminator(fake)
        return fake, fake_pred

    # Discriminator Loss: Calculate the loss for training the discriminator.
    def _discriminator_loss(self, lr_image: torch.Tensor, hr_image: torch.Tensor) -> torch.Tensor:
        real_pred = self.discriminator(hr_image)
        real_loss = self._adv_loss(real_pred, ones=True)

        _, fake_pred = self._fake_pred(lr_image)
        fake_loss = self._adv_loss(fake_pred, ones=False)

        return 0.5 * (real_loss + fake_loss)

    # Generator Loss: Calculate the loss for training the generator.
    def _generator_loss(self, lr_image: torch.Tensor, hr_image: torch.Tensor) -> torch.Tensor:
        fake, fake_pred = self._fake_pred(lr_image)

        perceptual_loss = self._perceptual_loss(hr_image, fake)
        adv_loss = self._adv_loss(fake_pred, ones=True)
        content_loss = F.mse_loss(fake, hr_image)

        return 0.006 * perceptual_loss + 0.01 * adv_loss + content_loss

    # Adversarial Loss: Helper function to calculate adversarial loss.
    @staticmethod
    def _adv_loss(pred: torch.Tensor, ones: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if ones else torch.zeros_like(pred)
        return F.binary_cross_entropy_with_logits(pred, target)
    
    # Perceptual Loss: Calculate perceptual loss using features from the VGG network.
    def _perceptual_loss(self, hr_image: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        real_features = self.feature_extractor(hr_image)
        fake_features = self.feature_extractor(fake) #(16, 512 , 9, 9)
        return F.mse_loss(fake_features, real_features)