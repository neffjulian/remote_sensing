"""
Bicubic interpolation

The input is already upsampled with bicubic interpolation. Thus we do not need to train as the input is always equal to the output.
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

class bicubic_interpolation(LightningModule):
    def __init__(self, batch_size: int = 32, lr: float = 0.0003):
        super().__init__()
        self.batch_size = batch_size

    def forward(self, x):
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, 'max'),
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

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        psnr_val = psnr(y_hat, y)
        self.logger.log_metrics({'psnr': psnr_val})
        return x, y_hat, y