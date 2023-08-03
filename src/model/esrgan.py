"""
ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (2018) by Wang et al.

Paper: https://arxiv.org/pdf/1809.00219.pdf
Adpted from: https://github.com/leverxgroup/esrgan
"""

from typing import Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

from torchmetrics import StructuralSimilarityIndexMeasure

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from torchmetrics import StructuralSimilarityIndexMeasure

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        for i in range(5):
            self.convs.append(
                nn.Sequential(
                    nn.ReplicationPad2d(1),
                    nn.Conv2d(channels + i * channels, channels, kernel_size=3),
                    nn.LeakyReLU(negative_slope=0.2) if i < 4 else nn.Identity()
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for conv in self.convs:
            out = conv(torch.cat(features, dim=1))
            features.append(out)
        return out * 0.2 + x
    
class ResidualInResidual(nn.Module):
    def __init__(self, blocks: int, channels: int) -> None:
        super().__init__()
        res_blocks = [ResidualDenseBlock(channels)] * blocks
        self.blocks = nn.ModuleList(res_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out += 0.2 * block(out)
        return x + 0.2 * out
    
class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.channels = 64
        upscaling_factor = 6
        upscaling_channels = 3
        blocks = 16

        self.model = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, upscaling_factor * upscaling_factor * upscaling_channels, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.2),

            nn.PixelShuffle(upscaling_factor),

            nn.ReplicationPad2d(1),
            nn.Conv2d(upscaling_channels, self.channels, kernel_size=3),
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

class VGG19FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.vgg = nn.Sequential(*list(vgg.features)[:-1]).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg(x.repeat(1, 3, 1, 1))

class Discriminator(nn.Module):
    def __init__(self, feature_maps: int = 64) -> None:
        super().__init__()

        self.conv_blocks = nn.Sequential(
            self._make_double_conv_block(1, feature_maps, first_batch_norm=False),
            self._make_double_conv_block(feature_maps, feature_maps * 2),
            self._make_double_conv_block(feature_maps * 2, feature_maps * 4),
            self._make_double_conv_block(feature_maps * 4, feature_maps * 8)
        )

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_maps * 8, feature_maps * 16, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feature_maps * 16, feature_maps * 16, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feature_maps * 16, 1, kernel_size=1),
            nn.Flatten()
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.normal_(module.weight, mean=0, std=0.001)
                if module.bias is not None:
                    module.bias.data.zero_()

    def _make_double_conv_block(self, in_channels: int, out_channels: int, first_batch_norm: bool = True) -> nn.Sequential:
        return nn.Sequential(
            self._make_conv_block(in_channels, out_channels, batch_norm=first_batch_norm),
            self._make_conv_block(out_channels, out_channels, stride=2),
        )

    def _make_conv_block(self, in_channels: int, out_channels: int, stride: int = 1, batch_norm: bool = True) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.mlp(x)
        return x

class ESRGAN(pl.LightningModule):
    def __init__(self, hparams: dict) -> None:
        super().__init__()

        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]

        self.generator = Generator()
        self.discriminator = Discriminator(hparams["model"]["feature_maps_disc"])

        self.feature_extractor = VGG19FeatureExtractor()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)

    def configure_optimizers(self) -> Tuple:
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        sched_gen = torch.optim.lr_scheduler.StepLR(opt_gen, step_size=self.scheduler_step, gamma=0.5)
        sched_disc = torch.optim.lr_scheduler.StepLR(opt_disc, step_size=self.scheduler_step, gamma=0.5)

        return [opt_gen, opt_disc], [sched_gen, sched_disc]

    
    def forward(self, lr_image: torch.Tensor) -> torch.Tensor:
        return self.generator(lr_image)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, optimizer_idx: int) -> None:
        lr_image, hr_image = batch

        loss = None
        if optimizer_idx == 0:
            loss = self._generator_loss(lr_image, hr_image)
        if optimizer_idx == 1:
            loss = self._discriminator_loss(lr_image, hr_image)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        lr_image, hr_image = batch
        sr_image = self(lr_image)

        loss_gen = self._generator_loss(lr_image, hr_image)
        loss_disc = self._discriminator_loss(lr_image, hr_image)

        self.log("loss_gen", loss_gen, on_epoch=True, sync_dist=True)
        self.log("loss_disc", loss_disc, on_epoch=True, sync_dist=True)

        mse_loss = F.mse_loss(sr_image, hr_image)
        self.log("psnr", psnr(mse_loss), on_epoch=True, sync_dist=True)
        self.log("ssim", self.ssim(sr_image, hr_image), on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        lr_image, hr_image, names = batch
        sr_image = self(lr_image)
        lr_image = F.interpolate(lr_image, size=(150, 150), mode='bicubic')
        return lr_image, sr_image, hr_image, names
    
    def _fake_pred(self, lr_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fake = self(lr_image)
        fake_pred = self.discriminator(fake)
        return fake, fake_pred

    def _discriminator_loss(self, lr_image: torch.Tensor, hr_image: torch.Tensor) -> torch.Tensor:
        real_pred = self.discriminator(hr_image)
        _, fake_pred = self._fake_pred(lr_image)

        d1_loss = self._adv_loss(real_pred - fake_pred, ones=True)
        d2_loss = self._adv_loss(fake_pred - real_pred, ones=False)

        return (d1_loss + d2_loss) / 2.0

    def _generator_loss(self, lr_image: torch.Tensor, hr_image: torch.Tensor) -> torch.Tensor:
        fake, fake_pred = self._fake_pred(lr_image)

        perceptual_loss = self._perceptual_loss(hr_image, fake)
        adv_loss = self._adv_loss(fake_pred, ones=True)
        content_loss = F.mse_loss(fake, hr_image)
        self.log("Perceptual Loss", perceptual_loss, on_epoch=True, sync_dist=True)
        self.log("Adv Loss", adv_loss, on_epoch=True, sync_dist=True)
        
        return 0.01 * perceptual_loss + 0.01 * adv_loss + content_loss * 0.98

    @staticmethod
    def _adv_loss(pred: torch.Tensor, ones: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if ones else torch.zeros_like(pred)
        return F.binary_cross_entropy_with_logits(pred, target)
    
    def _perceptual_loss(self, hr_image: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        real_features = self.feature_extractor(hr_image)
        fake_features = self.feature_extractor(fake)
        return F.l1_loss(fake_features, real_features)