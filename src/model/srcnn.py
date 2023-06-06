"""
Super-Resolution Convolutional Neural Network (2015)

Paper: https://arxiv.org/abs/1501.00092
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SRCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.super_resolution = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=5, padding=2)
        )

        for module in self.super_resolution.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.super_resolution(x.unsqueeze(1)).squeeze()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def prediction_step(self, pred_batch, batch_idx):
        x, y = pred_batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)