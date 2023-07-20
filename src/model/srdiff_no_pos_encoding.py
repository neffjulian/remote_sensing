from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningModule
from torchmetrics import StructuralSimilarityIndexMeasure

WEIGHT_DIR = Path(__file__).parent.parent.parent.joinpath("weights", "rrdb.ckpt")

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

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
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.contract = nn.Sequential(
            ResBlock(channels_in, channels_out),
            ResBlock(channels_out, channels_out),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.contract(x)
    
class ExpansiveStep(nn.Module):
    def __init__(self, index: int, channels_in: int, channels_out: int) -> None:
        super().__init__()
        out_size = {1: (18, 18), 2: (37, 37), 3: (75, 75), 4: (150, 150)}

        self.res1 = ResBlock(channels_in * 2, channels_in)
        self.res2 = ResBlock(channels_in, channels_out)
        self.upsample = nn.UpsamplingBilinear2d(size=out_size[index])
        self.out_size = out_size[index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res1(x)
        x = self.res2(x)
        a = (len(x.shape) == 3)
        if a:
            x = x.unsqueeze(0)
        x = self.upsample(x)
        if a:
            x = x.squeeze(0)
        return x
    
class MiddleStep(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.middle = nn.Sequential(
            ResBlock(channels_in, channels_in),
            ResBlock(channels_in, channels_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.middle(x)
    
class UNet(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.channels = channels

        self.contracting1 = ContractingStep(channels + 1, channels)
        self.contracting2 = ContractingStep(channels, channels * 2)
        self.contracting3 = ContractingStep(channels * 2, channels * 2)
        self.contracting4 = ContractingStep(channels * 2, channels * 4)
        self.middle = MiddleStep(channels * 4, channels * 4)
        self.expansive1 = ExpansiveStep(1, channels * 4, channels * 2)
        self.expansive2 = ExpansiveStep(2, channels * 2, channels * 2)
        self.expansive3 = ExpansiveStep(3, channels * 2, channels)
        self.expansive4 = ExpansiveStep(4, channels, channels)
        self.output = ConvBlock(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.contracting1(x)
        c2 = self.contracting2(c1)
        c3 = self.contracting3(c2)
        c4 = self.contracting4(c3)
        m = self.middle(c4)
        e1 = self.expansive1(torch.cat([m, c4], dim=0))
        e2 = self.expansive2(torch.cat([e1, c3], dim=0))
        e3 = self.expansive3(torch.cat([e2, c2], dim=0))
        e4 = self.expansive4(torch.cat([e3, c1], dim=0))
        return self.output(e4)

class SRDIFF_simple(LightningModule):
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
        self.end_block = ConvBlock(64, 1)

        # Get alphas and betas
        self.T = 100
        beta_scheduler = BetaScheduler()
        self.beta = beta_scheduler(self.T)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.beta_hat = torch.zeros(self.T)
        self.beta_hat[0] = self.beta[0]
        for t in range(1, self.T):
            self.beta_hat[t] = (1. - self.alpha_hat[t-1]) / (1. - self.alpha_hat[t]) * self.beta[t]

        self.loss = nn.L1Loss()
        self.upsample = nn.UpsamplingBilinear2d(size=(150, 150))

    def _get_lr_encoder(self) -> RRDB:
        encoder = RRDB.load_state_dict(torch.load(WEIGHT_DIR)["state_dict"])
        # checkpoint = torch.load(WEIGHT_DIR, map_location=torch.device("cpu"))

        for param in encoder.parameters():
            param.requires_grad = False

        return encoder
    
    def _conditional_noise_predictor(self, x_t: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        return self.end_block(self.unet(torch.cat([self.start_block(x_t), x_e], dim=0)))
    
    def _train(self, x_L: torch.Tensor, x_H: torch.Tensor) -> torch.Tensor:
        up_x_L = self.upsample(x_L)
        x_e = self.lr_encoder(x_L)
        x_r = x_H - up_x_L

        num_imgs = x_L.shape[0]
        ts = torch.randint(0, self.T, size=(num_imgs,))
        alpha_hat_ts = self.alpha_hat[ts]
        noise = torch.normal(mean = 0, std = 1, size = x_H.shape)

        x_t = torch.sqrt(alpha_hat_ts) * x_r + torch.sqrt(1. - alpha_hat_ts) * noise
        noise_pred = self._conditional_noise_predictor(x_t, x_e)
        loss = F.l1_loss(noise_pred, noise)
        return loss
    
    def _infere(self, x_L: torch.Tensor) -> torch.Tensor:
        up_x_L = self.upsample(x_L)
        x_e = self.lr_encoder(x_L)
        x_T = torch.normal(mean = 0, std = 1, size = up_x_L.shape)
        for t in range(self.T-1, -1, -1):
            z = torch.normal(mean = 0, std = 1, size = up_x_L.shape)
            x_T = (1. / torch.sqrt(self.alpha[t])) \
                * (x_T - (1. - self.alpha[t]) / torch.sqrt(1. - self.alpha_hat[t]) \
                * self._conditional_noise_predictor(x_T, x_e))
            if t > 0:
                x_T += torch.sqrt(self.beta_hat[t]) * z

        return up_x_L + x_T
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': StepLR(
                    optimizer=optimizer,
                    step_size=self.scheduler_step,
                    gamma=0.5,
                    verbose=True
                ),
                'monitor': 'val_ssim'
            }
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._infere(x)

    def training_step(self, batch, batch_idx):
        lr_image, hr_image = batch
        loss = self._train(lr_image, hr_image)
        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx):
        lr_image, hr_image = batch
        sr_image = self._infere(lr_image)
        ssim = self.ssim(sr_image, hr_image)
        psnr = self.psnr(sr_image, hr_image)
        self.log('val_psnr', psnr)
        self.log('val_ssim', ssim)

    def predict_step(self, batch, batch_idx):
        lr_image, hr_image, names = batch
        sr_image = self._infere(lr_image)
        lr_image = F.interpolate(lr_image, size=(150, 150), mode='bicubic')
        return lr_image, sr_image, hr_image, names

# bs = BetaScheduler()

# betas = bs(100)
# alphas = 1. - betas
# alpha_hat = torch.cumprod(alphas, dim=0)

# beta_hats = torch.zeros(100)
# beta_hats[0] = betas[0]
# for t in range(1, 100):
#     beta_hats[t] = (1 - alpha_hat[t-1]) / (1 - alpha_hat[t]) * betas[t]

# print(alpha_hat)

# for t in range(99, -1, -1):
#     print(t)

# rrdb = RRDB()
# checkpoint = torch.load(WEIGHT_DIR, map_location=torch.device("cpu"))
# rrdb.load_state_dict(checkpoint["state_dict"])

# file = torch.Tensor(np.load(PIC_DIR)).unsqueeze(0)
# print(file.shape)
# upsample = nn.UpsamplingBilinear2d(size=(150, 150))

# print(file.shape, upsample(file).shape)

# lr = file.numpy()
# ts = torch.randint(0, 1, (16,))
# print(ts)
# print(alpha_hat[ts])
# sr = rrdb(file.unsqueeze(0)).squeeze().detach().numpy()
# import cv2
# print(lr.shape, sr.shape)

# plt.imshow(cv2.resize(lr, (150, 150), interpolation=cv2.INTER_CUBIC))
# plt.show()

# plt.imshow(sr)
# plt.show()


