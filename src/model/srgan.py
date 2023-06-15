"""
 SRGAN: a generative adversarial network (GAN) for image superresolution (SR).

Paper: https://arxiv.org/pdf/1609.04802.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule

class Generator(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()

        self.batch_size = hparams["model"]["batch_size"]
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler = hparams["scheduler"]

        self.input_layer = nn.Conv2d(1, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
