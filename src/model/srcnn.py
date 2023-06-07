"""
Super-Resolution Convolutional Neural Network (2015)

Paper: https://arxiv.org/abs/1501.00092
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule

def psnr(y_hat, y):
    mse = F.mse_loss(y_hat, y)
    psnr_val = 20 * torch.log10(torch.max(y) / torch.sqrt(mse))
    return psnr_val

class SRCNN(LightningModule):
    def __init__(self, batch_size: int = 32, lr: float = 0.0003):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.super_resolution = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=9, padding=4),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=5, padding=2)
        )

        for module in self.super_resolution.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.super_resolution(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, 'min'),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
            'strict': True
        }
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        psnr_loss = psnr(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_psnr', psnr_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        psnr_loss = psnr(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_psnr', psnr_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        psnr_val = psnr(y_hat, y)
        self.logger.log_metrics({'psnr': psnr_val})
        return x, y_hat, y