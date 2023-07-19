from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

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

WEIGHT_DIR = Path(__file__).parent.parent.parent.joinpath("weights", "rrdb.ckpt")
PIC_DIR = Path(__file__).parent.parent.parent.joinpath("data", "processed", "20m", "03_0000_00.npy")

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        for i in range(5):
            self.convs.append(
                nn.Sequential( # (64, 150, 150)
                    nn.ReplicationPad2d(1), # (64, 152, 152)
                    nn.Conv2d(channels + i * channels, channels, kernel_size=3), # (64, 150, 150)
                    nn.LeakyReLU(negative_slope=0.2) if i < 4 else nn.Identity()
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for conv in self.convs:
            out = conv(torch.cat(features, dim=0))
            features.append(out)
        return out * 0.2 + x
    
class ResidualInResidual(nn.Module):
    def __init__(self, blocks: int, channels: int) -> None:
        super().__init__()
        res_blocks = [ResidualDenseBlock(channels)] * blocks
        self.blocks = nn.ModuleList(res_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x # (64, 150, 150)
        for block in self.blocks:
            out += 0.2 * block(out) 
        return x + 0.2 * out

class RRDB(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.channels = 64
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)

        upscaling_factor = 6
        upscaling_channels = 3
        blocks = 16

        self.model = nn.Sequential(
            nn.ReplicationPad2d(1), # (1, 27, 27)
            nn.Conv2d(1, upscaling_factor * upscaling_factor * upscaling_channels, kernel_size=3), # (108, 25, 25)
            nn.LeakyReLU(negative_slope=0.2),

            nn.PixelShuffle(upscaling_factor), # (3, 150, 150)

            nn.ReplicationPad2d(1), # (3, 152, 152)
            nn.Conv2d(upscaling_channels, self.channels, kernel_size=3), # (64, 150, 150)
            nn.LeakyReLU(negative_slope=0.2),

            ResidualInResidual(blocks, self.channels),

            nn.ReplicationPad2d(1),
            nn.Conv2d(self.channels, self.channels, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.ReplicationPad2d(1),
            nn.Conv2d(self.channels, 1, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class SRDIFF_simple(LightningModule):
    def __init__(self, channels: int) -> None:
        super().__init__()

file = torch.Tensor(np.load(PIC_DIR))

rrdb = RRDB()
checkpoint = torch.load(WEIGHT_DIR, map_location=torch.device("cpu"))
rrdb.load_state_dict(checkpoint["state_dict"])


lr = file.numpy()
sr = rrdb(file.unsqueeze(0)).squeeze().detach().numpy()
import cv2
print(lr.shape, sr.shape)

plt.imshow(cv2.resize(lr, (150, 150), interpolation=cv2.INTER_CUBIC))
plt.show()

plt.imshow(sr)
plt.show()
