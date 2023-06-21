"""
 Enhanced Deep Super-Resolution Network (2017)

Paper: https://arxiv.org/pdf/1707.02921v1.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")

def mse(y_hat, y):
    return torch.mean((y_hat - y) ** 2)

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

class EDSR(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.batch_size = hparams["model"]["batch_size"]
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler = hparams["scheduler"]

        self.input_layer = nn.Conv2d(1, 256, kernel_size=3, padding=1)
        self.output_layer = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        
        residual_layers = [
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1)
            )
        ] * 8

        self.residual_layers = nn.Sequential(*residual_layers)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)

    def forward(self, x):
        x_hat = self.input_layer(x)
        return self.output_layer(x_hat + self.residual_layers(x_hat))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(
                    optimizer=optimizer,
                    patience=self.scheduler["patience"],
                    min_lr=self.scheduler["min_lr"],
                    verbose=self.scheduler["verbose"]
                ),
                'monitor': 'val_l1_loss'
            }
        }

    def shared_step(self, batch, stage):
        x, y = batch
        y_hat = self.forward(x)
        l1_loss = F.l1_loss(y_hat, y)     

        mse_loss = mse(y_hat, y)
        self.log(f"{stage}_l1_loss", l1_loss, sync_dist=True)
        self.log(f"{stage}_mse_loss", mse_loss, sync_dist=True)        
        if stage == "val":
            self.log(f"{stage}_psnr", psnr(mse_loss), sync_dist=True)
            self.log(f"{stage}_ssim", self.ssim(y_hat, y), sync_dist=True)
        return l1_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        x, y, names = batch
        y_hat = self.forward(x)
        return x, y_hat, y, names