"""
SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models (2021) by Li et al.

Paper: https://arxiv.org/pdf/2104.14951.pdf
Adapted from: https://github.com/jbergq/simple-diffusion-model/tree/main
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
    def __init__() -> None:
        super().__init__()


    
class ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
class ResBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int = 1) -> None:
        self.conv1 = ConvBlock(channels_in, channels_out)
        self.conv2 = ConvBlock(channels_out, channels_out, stride)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x) + te) + x

class ContractingStep(nn.Module):
    def __init__(self, channels: int, double_channels: bool) -> None:
        super().__init__()
        if double_channels:
            self.res1 = ResBlock(channels // 2, channels)
        else:
            self.res1 = ResBlock(channels, channels)

        self.res2 = ResBlock(channels, channels, stride=2)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        return self.pooling(self.res2(self.res1(x, te), te))
    
class MiddleStep(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.res1 = ResBlock(channels, channels)
        self.res2 = ResBlock(channels, channels)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        return self.res2(self.res1(x, te), te)
    
class ExpansiveStep(nn.Module):
    def __init__(self, channels: int, double_channels: bool) -> None:
        super().__init__()
        if double_channels:
            self.res1 = ResBlock(channels // 2, channels)
        else:
            self.res1 = ResBlock(channels, channels)
        self.res2 = ResBlock(channels, channels)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        return self.upsample(self.res2(self.res1(x, te), te))

class UNet(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.contract1 = ContractingStep(channels, double_channels=False)
        self.contract2 = ContractingStep(channels * 2, double_channels=True)
        self.contract3 = ContractingStep(channels * 2, double_channels=False)
        self.contract4 = ContractingStep(channels * 4, double_channels=True)

        self.middle = MiddleStep(channels * 4)

        self.expand1 = ExpansiveStep(channels * 4, double_channels=False)
        self.expand2 = ExpansiveStep(channels * 2, double_channels=True)
        self.expand3 = ExpansiveStep(channels * 2, double_channels=False)
        self.expand4 = ExpansiveStep(channels, double_channels=True)

        self.conv = ConvBlock(channels, channels)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        x1 = self.contract1(x, te)
        x2 = self.contract2(x1, te)
        x3 = self.contract3(x2, te)
        x4 = self.contract4(x3, te)

        x5 = self.middle(x4, te)

        x6 = self.expand1(x5, te)
        x7 = self.expand2(torch.cat([x6, x3], dim=1), te)
        x8 = self.expand3(torch.cat([x7, x2], dim=1), te)
        x9 = self.expand4(torch.cat([x8, x1], dim=1), te)

        return self.conv(x9)
        

class BetaScheduler:
    def __init__(self, type="linear") -> None:
        schedule_map = {
            "linear": self.linear_beta_schedule,
            "quadratic": self.quadratic_beta_schedule,
            "cosine": self.cosine_beta_schedule,
            "sigmoid": self.sigmoid_beta_schedule,
        }
        self.schedule = schedule_map[type]

    def __call__(self, timesteps):
        return self.schedule(timesteps)

    @staticmethod
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def quadratic_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2

    @staticmethod
    def sigmoid_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

class SRDiff(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.nr_steps = 100
        self.channels = 64
        self.input_dim = 1

        betascheduler = BetaScheduler(type="cosine")
        self.betas = betascheduler(self.nr_steps)
        self.alphas = 1 - self.betas
        self.alpha_cum = torch.cumprod(self.alphas, dim=0)

        self.preprocess_input = ConvBlock(self.input_dim, self.channels)
        self.lr_encoder = LREncoder(self.channels)
        self.model = UNet(self.channels)

    def forward(self, xt: torch.Tensor):
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