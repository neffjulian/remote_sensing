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


