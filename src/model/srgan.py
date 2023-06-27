"""
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (2017) by Ledig et al.

Paper: https://arxiv.org/abs/1609.04802
Adpted from: https://github.com/https-deeplearning-ai/GANs-Public/blob/master/C3W2_SRGAN_(Optional).ipynb
"""

from typing import Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class VGG19FeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.vgg = nn.Sequential(*list(vgg.features)[:-1]).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg(x.repeat(1, 3, 1, 1))

class ResidualBlock(nn.Module):
    def __init__(self, feature_maps: int = 64) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(feature_maps, feature_maps, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(feature_maps),
            nn.PReLU(),
            nn.Conv2d(feature_maps, feature_maps, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(feature_maps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self, feature_maps: int = 64, num_res_blocks: int = 16) -> None:
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(1, feature_maps, kernel_size=9, padding=4, padding_mode="replicate"),
            nn.PReLU()
        )

        residual_blocks = [ResidualBlock(feature_maps)] * num_res_blocks
        residual_blocks += [
            nn.Conv2d(feature_maps, feature_maps, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(feature_maps)
        ]
        self.residual_blocks = nn.Sequential(*residual_blocks)

        self.output_block = nn.Sequential(
            nn.Conv2d(feature_maps, 1, kernel_size=9, padding=4, padding_mode="replicate"),
            # nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = self.input_block(x)
        x = x_input + self.residual_blocks(x_input)
        x = self.output_block(x)
        return x

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
            nn.Conv2d(feature_maps * 16, 1, kernel_size=1),
            nn.Flatten()
        )

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

class SRGAN(pl.LightningModule):
    def __init__(self, hparams: dict) -> None:
        super().__init__()

        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]

        self.generator = Generator(hparams["model"]["feature_maps_gen"], hparams["model"]["num_res_blocks"])
        self.discriminator = Discriminator(hparams["model"]["feature_maps_disc"])

        self.feature_extractor = VGG19FeatureExtractor()

    def configure_optimizers(self) -> Tuple:
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr = self.lr)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr = self.lr)

        sched_gen = torch.optim.lr_scheduler.MultiStepLR(opt_gen, milestones=[self.scheduler_step], gamma=0.1)
        sched_disc = torch.optim.lr_scheduler.MultiStepLR(opt_disc, milestones=[self.scheduler_step], gamma=0.1)

        return [opt_gen, opt_disc], [sched_gen, sched_disc]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, optimizer_idx: int) -> None:
        lr_image, hr_image = batch

        loss = None
        if optimizer_idx == 0:
            loss = self._generator_loss(lr_image, hr_image)
            self.log("train_loss_gen", loss, on_step=True, on_epoch=True)
        
        if optimizer_idx == 1:
            loss = self._discriminator_step(lr_image, hr_image)
            self.manual_backward(loss)
            self.log("train_loss_disc", loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, optimizer_idx: int) -> None:
        lr_image, hr_image = batch

        loss = self._generator_loss(lr_image, hr_image)
        self.log("val_loss_gen", loss, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, names = batch
        y_hat = self.forward(x)
        return x, y_hat, y, names
    
    def _fake_pred(self, lr_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        print("lr image", lr_image.requires_grad)
        fake = self(lr_image)
        fake_pred = self.discriminator(fake)
        return fake, fake_pred

    def _discriminator_loss(self, lr_image: torch.Tensor, hr_image: torch.Tensor) -> torch.Tensor:
        real_pred = self.discriminator(hr_image)
        real_loss = self._adv_loss(real_pred, ones=True)

        _, fake_pred = self._fake_pred(lr_image)
        fake_loss = self._adv_loss(fake_pred, ones=False)

        return 0.5 * (real_loss + fake_loss)

    def _generator_loss(self, lr_image: torch.Tensor, hr_image: torch.Tensor) -> torch.Tensor:
        fake, fake_pred = self._fake_pred(lr_image)
        print("fake", fake.requires_grad)

        perceptual_loss = self._perceptual_loss(fake, hr_image)
        adv_loss = self._adv_loss(fake_pred, ones=True)
        content_loss = F.mse_loss(fake, hr_image)

        return 0.006 * perceptual_loss + 0.001 * adv_loss + content_loss

    @staticmethod
    def _adv_loss(pred: torch.Tensor, ones: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if ones else torch.zeros_like(pred)
        return F.binary_cross_entropy_with_logits(pred, target)
    
    def _perceptual_loss(self, hr_image: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            real_features = self.feature_extractor(hr_image)
            fake_features = self.feature_extractor(fake)
            return F.mse_loss(fake_features, real_features)