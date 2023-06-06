"""
 Enhanced Deep Super-Resolution Network (2017)

Paper: https://arxiv.org/pdf/1707.02921v1.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from torchmetrics import PSNR

class EDSR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Conv2d(1, 256, kernel_size=3, padding=1)
        self.output_layer = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        
        residual_layers = [
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1)
            )
        ] * 32

        self.residual_layers = nn.Sequential(*residual_layers)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.input_layer(x) + self.residual_layers(x)
        x = self.output_layer(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-2)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return scheduler
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        psnr = PSNR(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_psnr', psnr, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        psnr = PSNR(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_psnr', psnr, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        psnr = PSNR(y_hat, y)
        self.log('prediction_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('prediction_psnr', psnr, on_step=True, on_epoch=True, prog_bar=True, logger=True)