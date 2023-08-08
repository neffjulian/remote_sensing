from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure

WEIGHT_DIR = Path(__file__).parent.parent.parent.joinpath("weights", "rrdb.ckpt")

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

class PositionalEncoding(nn.Module):
    # Taken from: https://github.com/jbergq/simple-diffusion-model/blob/main/src/model/network/layers/positional_encoding.py

    def __init__(self, device: torch.device, max_time_steps: int, embedding_size: int, n: int = 10000) -> None:
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False, device=device)
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Returns embedding encoding time step `t`.

        Args:
            t (Tensor): Time step.

        Returns:
            Tensor: Returned position embedding.
        """
        return self.pos_embeddings[t, :]

class BetaScheduler:
    # Copied from: https://github.com/jbergq/simple-diffusion-model/blob/main/src/model/beta_scheduler.py
    def __init__(self, type="linear") -> None:
        schedule_map = {
            "linear": self.linear_beta_schedule,
            "quadratic": self.quadratic_beta_schedule,
            "cosine": self.cosine_beta_schedule,
            "sigmoid": self.sigmoid_beta_schedule,
        }
        self.schedule = schedule_map[type]

    def __call__(self, timesteps):
        return self.schedule(timesteps)

    @staticmethod
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def quadratic_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2

    @staticmethod
    def sigmoid_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
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

class RRDB(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.channels = 64
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)

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
    
class ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            nn.Mish()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
class ResBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels_in, channels_in)
        self.conv2 = ConvBlock(channels_in, channels_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x) + x)
    
class ContractingStep(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, emb_size: int) -> None:
        super().__init__()
        self.r1 = ResBlock(channels_in, channels_out)
        self.r2 = ResBlock(channels_out, channels_out)
        self.mp = nn.MaxPool2d(2)
        self.proj = nn.Sequential(
            nn.Linear(emb_size, channels_out),
            nn.Mish()
        )

    def forward(self, x: torch.Tensor, t: torch.TensorType) -> torch.Tensor:
        t_emb = self.proj(t).unsqueeze(-1).unsqueeze(-1)
        return self.mp(self.rs(self.r1(x) + t_emb))
    
class ExpansiveStep(nn.Module):
    def __init__(self, index: int, channels_in: int, channels_out: int, emb_size: int) -> None:
        super().__init__()
        out_size = {1: (18, 18), 2: (37, 37), 3: (75, 75), 4: (150, 150)}

        self.res1 = ResBlock(channels_in * 2, channels_in)
        self.res2 = ResBlock(channels_in, channels_out)
        self.upsample = nn.UpsamplingBilinear2d(size=out_size[index])
        self.out_size = out_size[index]
        self.proj = nn.Sequential(
            nn.Linear(emb_size, channels_in),
            nn.Mish()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.proj(t).unsqueeze(-1).unsqueeze(-1)
        return self.upsample(self.res2(self.res1(x) + t_emb))
    
class MiddleStep(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, emb_size: int) -> None:
        super().__init__()
        self.r1 = ResBlock(channels_in, channels_in)
        self.r2 = ResBlock(channels_in, channels_out)

        self.proj = nn.Sequential(
            nn.Linear(emb_size, channels_in),
            nn.Mish()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.proj(t).unsqueeze(-1).unsqueeze(-1)
        return self.r1(self.r2(x) + t_emb)
    
class UNet(nn.Module):
    def __init__(self, channels: int = 64, time_steps: int = 100, embedding_size: int = 512) -> None:
        super().__init__()
        self.channels = channels

        self.embedding = nn.Sequential(
            PositionalEncoding(self.device, time_steps, embedding_size),
            nn.Linear(embedding_size, embedding_size)
        )

        self.contracting1 = ContractingStep(channels + 1, channels, embedding_size)
        self.contracting2 = ContractingStep(channels, channels * 2, embedding_size)
        self.contracting3 = ContractingStep(channels * 2, channels * 2, embedding_size)
        self.contracting4 = ContractingStep(channels * 2, channels * 4, embedding_size)
        self.middle = MiddleStep(channels * 4, channels * 4, embedding_size)
        self.expansive1 = ExpansiveStep(1, channels * 4, channels * 2, embedding_size)
        self.expansive2 = ExpansiveStep(2, channels * 2, channels * 2, embedding_size)
        self.expansive3 = ExpansiveStep(3, channels * 2, channels, embedding_size)
        self.expansive4 = ExpansiveStep(4, channels, channels, embedding_size)
        self.output = ConvBlock(channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.embedding(t)
        c1 = self.contracting1(x, t_emb)
        c2 = self.contracting2(c1, t)
        c3 = self.contracting3(c2, t)
        c4 = self.contracting4(c3, t)
        m = self.middle(c4, t)
        e1 = self.expansive1(torch.cat([m, c4], dim=1), t)
        e2 = self.expansive2(torch.cat([e1, c3], dim=1), t)
        e3 = self.expansive3(torch.cat([e2, c2], dim=1),t)
        e4 = self.expansive4(torch.cat([e3, c1], dim=1), t)
        return self.output(e4)

class SRDIFF(LightningModule):
    def __init__(self, hparams: dict) -> None:
        super().__init__()
        self.channels = hparams["model"]["channels"]
        self.ssim = StructuralSimilarityIndexMeasure(data_range=8.0)
        self.lr = hparams["optimizer"]["lr"]
        self.scheduler_step = hparams["optimizer"]["scheduler_step"]
        # Applies SR using the RRDB Model.
        self.lr_encoder = self._get_lr_encoder()
        
        # Get remaining blocks
        self.start_block = ConvBlock(1, self.channels)
        self.unet = UNet(channels=self.channels)

        # Get alphas and betas
        self.T = 100
        beta_scheduler = BetaScheduler()
        self.beta = beta_scheduler(self.T)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(self.device)
        
        self.beta_hat = torch.zeros(self.T, device=self.device)
        self.beta_hat[0] = self.beta[0]
        for t in range(1, self.T):
            self.beta_hat[t] = (1. - self.alpha_hat[t-1]) / (1. - self.alpha_hat[t]) * self.beta[t]

        self.loss = nn.L1Loss()
        self.upsample = nn.UpsamplingBilinear2d(size=(150, 150))

    def _get_lr_encoder(self) -> RRDB:
        encoder = RRDB()
        checkpoint = torch.load(WEIGHT_DIR, map_location=self.device)
        encoder.load_state_dict(checkpoint["state_dict"])

        for param in encoder.parameters():
            param.requires_grad = False

        return encoder
    
    def _conditional_noise_predictor(self, x_t: torch.Tensor, x_e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet(torch.cat([self.start_block(x_t), x_e], dim=1), t)
    
    def _is_last_batch(self, batch_idx: int):
        return batch_idx == self.trainer.num_training_batches - 1
    
    def _train(self, x_L: torch.Tensor, x_H: torch.Tensor) -> torch.Tensor:
        x_e = self.lr_encoder(x_L)
        x_r = x_H - self.upsample(x_L)

        num_imgs = x_L.shape[0]
        ts = torch.randint(0, self.T, size=(num_imgs,))
        alpha_hat_ts = self.alpha_hat[ts].to(self.device)
        alpha_hat_ts = alpha_hat_ts[:, None, None, None]
        noise = torch.normal(mean = 0, std = 1, size = x_H.shape, device=self.device)
        x_t = torch.sqrt(alpha_hat_ts) * x_r + torch.sqrt(1. - alpha_hat_ts) * noise
        noise_pred = self._conditional_noise_predictor(x_t, x_e, ts)
        loss = F.l1_loss(noise_pred, noise)
        return loss
    
    def _infere(self, x_L: torch.Tensor, show_steps: bool = False) -> torch.Tensor:
        up_x_L = self.upsample(x_L)
        x_e = self.lr_encoder(x_L)        
        x_T = torch.normal(mean = 0, std = 1, size = up_x_L.shape, device = self.device)

        steps = [x_T]
        for t in range(self.T-1, -1, -1):

            z = torch.normal(mean = 0, std = 1, size = up_x_L.shape, device=self.device)
            x_T = (1. / torch.sqrt(self.alpha[t])) \
                * (x_T - (1. - self.alpha[t]) / torch.sqrt(1. - self.alpha_hat[t]) \
                * self._conditional_noise_predictor(x_T, x_e, t))
            if t > 0:
                x_T += torch.sqrt(self.beta_hat[t]) * z

            steps.append(x_T)

        return up_x_L + x_T, steps
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': StepLR(
                    optimizer=optimizer,
                    step_size=self.scheduler_step,
                    gamma=0.5,
                ),
                'monitor': 'val_ssim'
            }
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x)
        std = torch.std(x)
        out, _ = self._infere((x - mean) / std)
        return out * std + mean
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self._infere(x)
        return out

    def training_step(self, batch, batch_idx):
        lr_image, hr_image = batch
        loss = self._train(lr_image, hr_image)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr_image, hr_image = batch
        sr_image, _ = self._infere(lr_image)
        loss = self._train(lr_image, hr_image)
        ssim = self.ssim(sr_image, hr_image)
        self.log('val_loss', loss, sync_dist=True)
        psnr_value = psnr(F.mse_loss(sr_image, hr_image))
        self.log('val_psnr', psnr_value, sync_dist=True)
        self.log('val_ssim', ssim, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        lr_image, hr_image, names = batch
        sr_image, steps = self._infere(lr_image)
        lr_image = F.interpolate(lr_image, size=(150, 150), mode='bicubic')
        
        for i, step in enumerate(steps):
            for j, data in enumerate(step):
                np.save(f"step_{i}_{j}.npy", data.numpy())
        return lr_image, sr_image, hr_image, names