"""
SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models (2021) by Li et al.

Paper: https://arxiv.org/pdf/2104.14951.pdf
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


class LREncoder(nn.Module):
    def __init__(self, channels: int, growth: int, residual_scaling: float = 0.2) -> None:
        super().__init__()
        self.residual_scaling = residual_scaling
        self.convs = nn.ModuleList()
        for i in range(4):
            self.convs.append(
                nn.Sequential(
                    nn.ReplicationPad2d(1),
                    nn.Conv2d(channels + i * growth, growth, kernel_size=3),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for conv in self.convs:
            out = conv(torch.cat(features, dim=1))
            features.append(out)
        return out * self.residual_scaling + x

class ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(channels_in, channels_out, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
class ResBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int,  te) -> None:
        self.conv1 = ConvBlock(channels_in, channels_out)
        self.conv2 = ConvBlock(channels_out, channels_out)
        self.te = te

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x) + self.te) + x

class ContractingStep(nn.Module):
    def __init__(self, channels: int, double_channels: bool, te) -> None:
        super().__init__()
        if double_channels:
            self.res1 = ResBlock(channels // 2, channels, te)
        else:
            self.res1 = ResBlock(channels, channels, te)

        self.res2 = ResBlock(channels, channels, te)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooling(self.res2(self.res1(x)))
    
class MiddleStep(nn.Module):
    def __init__(self, channels: int, te) -> None:
        super().__init__()
        self.res1 = ResBlock(channels, channels, te)
        self.res2 = ResBlock(channels, channels, te)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res2(self.res1(x))
    
class ExpansiveStep(nn.Module):
    def __init__(self, channels: int, double_channels: bool, te) -> None:
        super().__init__()
        if double_channels:
            self.res1 = ResBlock(channels // 2, channels, te)
        else:
            self.res1 = ResBlock(channels, channels, te)
        self.res2 = ResBlock(channels, channels * 4, te)
        self.upsample = nn.PixelShuffle(2)

class SRDiff(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.nr_steps = 30
        betas = torch.linspace(0.0001, 0.02, self.nr_steps)
        self.betas = torch.sigmoid(betas) * (3e-1 - 1e-5) + 1e-5
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, x):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': StepLR(
                    optimizer=optimizer,
                    step_size=self.scheduler_step,
                    gamma=0.1,
                    verbose=True
                ),
                'monitor': 'val_ssim'
            }
        }

    def shared_step(self, batch, stage):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def _forward_process(self, x: torch.Tensor, time_step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_step = time_step - 1
        beta_forward = self.betas[time_step]
        alpha_forward = self.alphas[time_step]
        alpha_cum_forward = self.alpha_bars[time_step]

        xt = x * torch.sqrt(alpha_cum_forward) + torch.randn_like(x) * torch.sqrt(1. - alpha_cum_forward)
        mu1_scl = torch.sqrt(alpha_cum_forward / alpha_forward)
        mu2_scl = 1. / torch.sqrt(alpha_forward)

        cov1 = 1. - alpha_cum_forward / alpha_forward
        cov2 = beta_forward / alpha_forward
        lam = 1. / cov1 + 1. / cov2
        mu = (x * mu1_scl / cov1 + xt * mu2_scl / cov2) / lam
        sigma = torch.sqrt(1. / lam)

        return mu, sigma, xt

    def _reverse_process(self, x: torch.Tensor, time_step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_step = time_step - 1
        if time_step == 0:
            return None, None, x
        mu, h = self.forward(x, time_step).chunk(2, dim=1)
        sigma = torch.sqrt(torch.exp(h))
        samples = mu + torch.randn_like(x) * sigma
        return mu, sigma, samples

    def _sample(self, size: torch.Tensor):
        noise = torch.randn((size, 2))
        samples = [noise]
        for t in range(self.nr_steps):
            _, _, noise = self._reverse_process(noise, t)
            samples.append(noise)
        return samples