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

class SRDiff(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.nr_steps = 30
        betas = torch.linspace(-18, 10, self.nr_steps)
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

