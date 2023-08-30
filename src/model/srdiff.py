"""
SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models

This model was intended to not use positional encodings. However, 
no good results were achieved with it and it converged fast.

Paper: https://arxiv.org/abs/2104.14951

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

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure

WEIGHT_DIR = Path(__file__).parent.parent.parent.joinpath("weights", "rrdb.ckpt")

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

class PositionalEncoding(nn.Module):
    # Taken from: https://github.com/jbergq/simple-diffusion-model/blob/main/src/model/network/layers/positional_encoding.py

    def __init__(self, device: torch.device, max_time_steps: int, embedding_size: int, n: int = 10000) -> None:
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False, device=device)
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Returns embedding encoding time step `t`.

        Args:
            t (Tensor): Time step.

        Returns:
            Tensor: Returned position embedding.
        """
        return self.pos_embeddings[t, :]

class BetaScheduler:
    # Copied from: https://github.com/jbergq/simple-diffusion-model/blob/main/src/model/beta_scheduler.py

    # Initialize the BetaScheduler object
    def __init__(self, type="linear") -> None:
        # Dictionary to map string type to function handler for the schedule
        schedule_map = {
            "linear": self.linear_beta_schedule,
            "quadratic": self.quadratic_beta_schedule,
            "cosine": self.cosine_beta_schedule,
            "sigmoid": self.sigmoid_beta_schedule,
        }
        # Assign the function handler based on the input type
        self.schedule = schedule_map[type]

    # Callable interface for the BetaScheduler, invokes the selected scheduling function
    def __call__(self, timesteps):
        return self.schedule(timesteps)

    # Linear beta schedule from beta_start to beta_end over 'timesteps'
    @staticmethod
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    # Quadratic beta schedule from beta_start to beta_end over 'timesteps'
    @staticmethod
    def quadratic_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2

    # Sigmoid beta schedule from beta_start to beta_end over 'timesteps'
    @staticmethod
    def sigmoid_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    # Cosine beta schedule from 0 to 1 over 'timesteps', based on paper at https://arxiv.org/abs/2102.09672
    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
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
    
class RRDB(nn.Module):
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
    
class ConvBlock(nn.Module):
    """A Convolutional Block that performs a 2D convolution followed by a Mish activation.
    
    Args:
        channels_in (int): Number of input channels.
        channels_out (int): Number of output channels.
    """
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        # Define the convolutional block
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            nn.Mish()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ConvBlock.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)
    
class ResBlock(nn.Module):
    """A Residual Block that contains two ConvBlocks and has a residual connection.
    
    Args:
        channels_in (int): Number of input channels.
        channels_out (int): Number of output channels.
    """
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        # Define the first and second ConvBlocks
        self.conv1 = ConvBlock(channels_in, channels_in)
        self.conv2 = ConvBlock(channels_in, channels_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResBlock.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv2(self.conv1(x) + x)
    
class ContractingStep(nn.Module):
    """A Contracting step in U-Net consisting of ResBlocks and MaxPool.
    
    Args:
        channels_in (int): Number of input channels.
        channels_out (int): Number of output channels.
    """
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        # Define the contracting step
        self.contract = nn.Sequential(
            ResBlock(channels_in, channels_out),
            ResBlock(channels_out, channels_out),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ContractingStep.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.contract(x)
    
class ExpansiveStep(nn.Module):
    """An Expansive step in U-Net consisting of ResBlocks and Upsampling.
    
    Args:
        index (int): Index used to determine output size.
        channels_in (int): Number of input channels.
        channels_out (int): Number of output channels.
    """
    def __init__(self, index: int, channels_in: int, channels_out: int) -> None:
        super().__init__()
        # Define the output sizes based on index
        out_size = {1: (18, 18), 2: (37, 37), 3: (75, 75), 4: (150, 150)}

        # Define the expansive step
        self.res1 = ResBlock(channels_in * 2, channels_in)
        self.res2 = ResBlock(channels_in, channels_out)
        self.upsample = nn.UpsamplingBilinear2d(size=out_size[index])
        self.out_size = out_size[index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ExpansiveStep.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.res1(x)
        x = self.res2(x)
        x = self.upsample(x)
        return x
    
class MiddleStep(nn.Module):
    """The Middle step in U-Net that sits between the contracting and expansive paths.
    
    Args:
        channels_in (int): Number of input channels.
        channels_out (int): Number of output channels.
    """
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        # Define the middle step
        self.middle = nn.Sequential(
            ResBlock(channels_in, channels_in),
            ResBlock(channels_in, channels_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MiddleStep.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.middle(x)
    
class UNet(nn.Module):
    """Implementation of U-Net architecture for image segmentation.
    
    Args:
        channels (int, optional): Number of initial channels. Defaults to 64.
    """
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.channels = channels  # Number of initial channels

        # Contracting path: consists of four ContractingSteps
        self.contracting1 = ContractingStep(channels + 1, channels)
        self.contracting2 = ContractingStep(channels, channels * 2)
        self.contracting3 = ContractingStep(channels * 2, channels * 2)
        self.contracting4 = ContractingStep(channels * 2, channels * 4)
        
        # Middle part: sits between the contracting and expansive paths
        self.middle = MiddleStep(channels * 4, channels * 4)

        # Expansive path: consists of four ExpansiveSteps
        self.expansive1 = ExpansiveStep(1, channels * 4, channels * 2)
        self.expansive2 = ExpansiveStep(2, channels * 2, channels * 2)
        self.expansive3 = ExpansiveStep(3, channels * 2, channels)
        self.expansive4 = ExpansiveStep(4, channels, channels)

        # Output layer: single convolutional layer
        self.output = ConvBlock(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net architecture.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after forward pass.
        """
        # Contracting path
        c1 = self.contracting1(x)
        c2 = self.contracting2(c1)
        c3 = self.contracting3(c2)
        c4 = self.contracting4(c3)
        
        # Middle part
        m = self.middle(c4)
        
        # Expansive path with skip connections
        e1 = self.expansive1(torch.cat([m, c4], dim=1))
        e2 = self.expansive2(torch.cat([e1, c3], dim=1))
        e3 = self.expansive3(torch.cat([e2, c2], dim=1))
        e4 = self.expansive4(torch.cat([e3, c1], dim=1))
        
        # Output layer
        return self.output(e4)

class SRDIFF(LightningModule):
    """
    SRDIFF is a class for Super-Resolution using a Diffusion-based approach.
    """
    def __init__(self, hparams: dict) -> None:
        """
        Initialize the SRDIFF class.
        
        Parameters:
        - hparams (dict): Hyperparameters for the model.
        """
        super().__init__()
        # Basic model settings
        self.channels = hparams["model"]["channels"]
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]
        
        # Initialize the low-resolution encoder
        self.lr_encoder = self._get_lr_encoder()
        
        # Initialize Conv and UNet blocks
        self.start_block = ConvBlock(1, self.channels)
        self.unet = UNet(channels=self.channels)

        # Initialize alpha and beta for the diffusion process
        self.T = 100
        beta_scheduler = BetaScheduler()
        self.beta = beta_scheduler(self.T)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device)
        
        # Compute beta_hat
        self.beta_hat = torch.zeros(self.T, device=self.device)
        self.beta_hat[0] = self.beta[0]
        for t in range(1, self.T):
            self.beta_hat[t] = (1. - self.alpha_hat[t-1]) / (1. - self.alpha_hat[t]) * self.beta[t]

        # Initialize loss and upsampling
        self.loss = nn.L1Loss()
        self.upsample = nn.UpsamplingBilinear2d(size=(150, 150))

    def _get_lr_encoder(self) -> RRDB:
        """
        Load the RRDB model for low-resolution encoding.
        
        Returns:
        - RRDB model with loaded weights.
        """
        encoder = RRDB()
        checkpoint = torch.load(WEIGHT_DIR, map_location=self.device)
        encoder.load_state_dict(checkpoint["state_dict"])
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder

    def _conditional_noise_predictor(self, x_t: torch.Tensor, x_e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Conditional noise predictor using U-Net.
        
        Parameters:
        - x_t (torch.Tensor): Tensor of transformed images.
        - x_e (torch.Tensor): Tensor of encoded low-res images.
        - t (torch.Tensor): Time step tensor.
        
        Returns:
        - Predicted noise (torch.Tensor).
        """
        return self.unet(torch.cat([self.start_block(x_t), x_e], dim=1), t)
    
    def _is_last_batch(self, batch_idx: int):
        """
        Check if the current batch is the last batch.
        
        Parameters:
        - batch_idx (int): The current batch index.
        
        Returns:
        - bool: True if it's the last batch, False otherwise.
        """
        return batch_idx == self.trainer.num_training_batches - 1
        
    def _train(self, x_L: torch.Tensor, x_H: torch.Tensor) -> torch.Tensor:
        """
        Performs a training step.
        
        Parameters:
        - x_L (torch.Tensor): Low-resolution image tensor.
        - x_H (torch.Tensor): High-resolution image tensor.
        
        Returns:
        - torch.Tensor: Computed loss.
        """
        # Encode the low-res image
        x_e = self.lr_encoder(x_L)
        # Compute the residual high-res image
        x_r = x_H - self.upsample(x_L)
        
        num_imgs = x_L.shape[0]
        # Generate random time steps
        ts = torch.randint(0, self.T, size=(num_imgs,))
        alpha_hat_ts = self.alpha_hat[ts].to(self.device)
        alpha_hat_ts = alpha_hat_ts[:, None, None, None]
        
        # Generate noise
        noise = torch.normal(mean=0, std=1, size=x_H.shape, device=self.device)
        # Create the transformed image
        x_t = torch.sqrt(alpha_hat_ts) * x_r + torch.sqrt(1. - alpha_hat_ts) * noise
        
        # Predict noise
        noise_pred = self._conditional_noise_predictor(x_t, x_e, ts)
        
        # Calculate L1 Loss between predicted and actual noise
        loss = F.l1_loss(noise_pred, noise)
        return loss
    
    def _infere(self, x_L: torch.Tensor, show_steps: bool = False) -> torch.Tensor:
        """
        Performs an inference step.
        
        Parameters:
        - x_L (torch.Tensor): Low-resolution image tensor.
        - show_steps (bool, optional): Whether to show steps. Defaults to False.
        
        Returns:
        - torch.Tensor: The resulting tensor after the inference step.
        """
        # Upsample the low-res image
        up_x_L = self.upsample(x_L)
        # Encode the low-res image
        x_e = self.lr_encoder(x_L)        
        # Generate initial tensor with random noise
        x_T = torch.normal(mean=0, std=1, size=up_x_L.shape, device=self.device)
        
        steps = [x_T]
        # Iteratively refine the prediction
        for t in range(self.T-1, -1, -1):
            # Generate random noise
            z = torch.normal(mean=0, std=1, size=up_x_L.shape, device=self.device)
            # Update x_T
            x_T = (1. / torch.sqrt(self.alpha[t])) \
                * (x_T - (1. - self.alpha[t]) / torch.sqrt(1. - self.alpha_hat[t]) \
                * self._conditional_noise_predictor(x_T, x_e, t))
            if t > 0:
                x_T += torch.sqrt(self.beta_hat[t]) * z

            steps.append(x_T)

        return up_x_L + x_T, steps
    
    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': StepLR(
                    optimizer=optimizer,
                    step_size=self.scheduler_step,
                    gamma=0.5,
                ),
                'monitor': 'val_ssim'
            }
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference."""
        mean = torch.mean(x)
        std = torch.std(x)
        return self._infere((x - mean) / std) * std + mean

    def training_step(self, batch, batch_idx):
        """Training loop for one mini-batch.
        
        Args:
            batch: The mini-batch of data.
            batch_idx: Batch index.
        
        Returns:
            loss: The loss for this mini-batch.
        """
        lr_image, hr_image = batch
        loss = self._train(lr_image, hr_image)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation loop for one mini-batch.
        
        Args:
            batch: The mini-batch of data.
            batch_idx: Batch index.
        """
        lr_image, hr_image = batch
        sr_image = self._infere(lr_image)
        loss = self._train(lr_image, hr_image)
        ssim = self.ssim(sr_image, hr_image)
        
        self.log('val_loss', loss, sync_dist=True)
        
        # Compute the Peak Signal-to-Noise Ratio (PSNR)
        psnr_value = psnr(F.mse_loss(sr_image, hr_image))
        self.log('val_psnr', psnr_value, sync_dist=True)
        self.log('val_ssim', ssim, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        """Prediction loop for one mini-batch.
        
        Args:
            batch: The mini-batch of data.
            batch_idx: Batch index.
        
        Returns:
            lr_image: Low-res image after bicubic interpolation.
            sr_image: Super-resolved image.
            hr_image: High-res image.
            names: Image filenames.
        """
        lr_image, hr_image, names = batch
        sr_image = self._infere(lr_image)
        
        # Upsample using bicubic interpolation
        lr_image = F.interpolate(lr_image, size=(150, 150), mode='bicubic')
        return lr_image, sr_image, hr_image, names