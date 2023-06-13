"""
 Enhanced Deep Super-Resolution Network (2017)

Paper: https://arxiv.org/pdf/1707.02921v1.pdf
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
        ] * 32

        self.residual_layers = nn.Sequential(*residual_layers)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.input_layer(x) + self.residual_layers(x)
        return self.output_layer(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer = optimizer, 
                patience = self.scheduler["patience"], 
                min_lr = self.scheduler["min_lr"], 
                verbose = self.scheduler["verbose"]),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
            'strict': True
        }
        return [optimizer], [scheduler]
    
    def log_loss(self,stage, loss, psnr_loss):
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_psnr", psnr_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
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
        self.log_loss("val", loss, psnr_loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        psnr_val = psnr(y_hat, y)
        self.logger.log_metrics({'psnr': psnr_val})
        return x, y_hat, y