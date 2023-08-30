"""
ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (2018) by Wang et al.

Note: Ensure that the version of Pytorch Lightning is below < 2.0.0 as it does not allow to optimize multiple optimizers at once anymore.

Paper: https://arxiv.org/pdf/1809.00219.pdf
Adpted from: https://github.com/leverxgroup/esrgan

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

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import vgg19, VGG19_Weights
from torchmetrics import StructuralSimilarityIndexMeasure

# Directory where the model weights are stored for the generator (i.e. RRDB)
WEIGHT_DIR = Path(__file__).parent.parent.parent.joinpath("weights")

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
            nn.Conv2d(feature_maps * 16, feature_maps * 16, kernel_size=1),
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
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

class ESRGAN(pl.LightningModule):
    """
    ESRGAN Class (Enhanced Super-Resolution Generative Adversarial Networks).

    This class is a PyTorch Lightning module that defines the architecture and
    functionality of the ESRGAN model, including the Generator (RRDB),
    Discriminator, and the feature extractor (VGG19).

    Attributes:
    -----------
    generator : nn.Module
        The generator model (RRDB).
    discriminator : nn.Module
        The discriminator model.
    feature_extractor : nn.Module
        Feature extractor model based on VGG19.
    ssim : nn.Module
        Structural Similarity Index Measure for image quality assessment.
    lr : float
        Learning rate for optimizers.
    scheduler_step : int
        Step size for the learning rate scheduler.
    """

    def __init__(self, hparams: dict) -> None:
        super().__init__()
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]

        # Load pretrained generator
        self.generator = RRDB.load_from_checkpoint(
            checkpoint_path="path/to/checkpoint.ckpt",
            map_location=self.device,
            hparams=hparams
        )

        self.discriminator = Discriminator(hparams["model"]["feature_maps_disc"])
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
        return self.generator(lr_image)

    # Training loop for both generator and discriminator
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, optimizer_idx: int) -> None:
        lr_image, hr_image = batch
        loss = None
        if optimizer_idx == 0:  # Generator optimizer
            loss = self._generator_loss(lr_image, hr_image)
        elif optimizer_idx == 1:  # Discriminator optimizer
            loss = self._discriminator_loss(lr_image, hr_image)
        return loss

    # Validation Step: Performed during model validation.
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        lr_image, hr_image = batch
        sr_image = self(lr_image)

        loss_gen = self._generator_loss(lr_image, hr_image)
        loss_disc = self._discriminator_loss(lr_image, hr_image)

        self.log("loss_gen", loss_gen, on_epoch=True, sync_dist=True)
        self.log("loss_disc", loss_disc, on_epoch=True, sync_dist=True)

        mse_loss = F.mse_loss(sr_image, hr_image)
        self.log("psnr", psnr(mse_loss), on_epoch=True, sync_dist=True)
        self.log("ssim", self.ssim(sr_image, hr_image), on_epoch=True, sync_dist=True)

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
        _, fake_pred = self._fake_pred(lr_image)

        d1_loss = self._adv_loss(real_pred - fake_pred, ones=True)
        d2_loss = self._adv_loss(fake_pred - real_pred, ones=False)

        return (d1_loss + d2_loss) / 2.0

    # Generator Loss: Calculate the loss for training the generator.
    def _generator_loss(self, lr_image: torch.Tensor, hr_image: torch.Tensor) -> torch.Tensor:
        fake, fake_pred = self._fake_pred(lr_image)

        perceptual_loss = self._perceptual_loss(hr_image, fake)
        adv_loss = self._adv_loss(fake_pred, ones=True)
        # content_loss = F.mse_loss(fake, hr_image)
        content_loss = F.l1_loss(fake, hr_image)
        self.log("Perceptual Loss", perceptual_loss, on_epoch=True, sync_dist=True)
        self.log("Adv Loss", adv_loss, on_epoch=True, sync_dist=True)
        
        return 1 * perceptual_loss + 0.005 * adv_loss + content_loss * 0.01

    # Adversarial Loss: Helper function to calculate adversarial loss.
    @staticmethod
    def _adv_loss(pred: torch.Tensor, ones: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if ones else torch.zeros_like(pred)
        return F.binary_cross_entropy_with_logits(pred, target)
    
    # Perceptual Loss: Calculate perceptual loss using features from the VGG network.
    def _perceptual_loss(self, hr_image: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        real_features = self.feature_extractor(hr_image)
        fake_features = self.feature_extractor(fake)
        return F.l1_loss(fake_features, real_features)