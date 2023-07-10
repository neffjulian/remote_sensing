from math import log10, sqrt
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import pytorch_lightning as pl
from skimage.metrics import structural_similarity as ssim

DATA_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'processed')

def psnr(x, y):
    return 20 * log10(8. / sqrt(np.mean((x - y) ** 2)))


class SRPredictDataset(Dataset):
    def __init__(self, hparams: dict, files: list[str]) -> None:
        super().__init__()

        ps_4b_dir = DATA_DIR.joinpath("4b")
        ps_4b_in_situ_dir = DATA_DIR.joinpath("4b_in_situ")
        ps_4b_lr_dir = DATA_DIR.joinpath("4b_lr")
        s2_20m_dir = DATA_DIR.joinpath("20m")
        s2_20m_in_situ_dir = DATA_DIR.joinpath("20m_in_situ")

        self.files = []
        for file in s2_20m_in_situ_dir.iterdir():
            self.files.append((s2_20m_in_situ_dir.joinpath(file.name), ps_4b_in_situ_dir.joinpath(file.name), file.name))

        ps_4b_files = [file.name for file in ps_4b_dir.iterdir() if file.name.startswith("03_0000") or file.name.startswith("03_0001")]
        ps_4b_lr_files = [file.name for file in ps_4b_lr_dir.iterdir() if file.name.startswith("03_0000") or file.name.startswith("03_0001")]
        s2_20m_files = [file.name for file in s2_20m_dir.iterdir() if file.name.startswith("03_0000") or file.name.startswith("03_0001")]
        assert ps_4b_files == ps_4b_lr_files == s2_20m_files

        for file in ps_4b_files:
            self.files.append((ps_4b_lr_dir.joinpath(file), ps_4b_dir.joinpath(file), file))

        for file in s2_20m_files:
            self.files.append((s2_20m_dir.joinpath(file), ps_4b_dir.joinpath(file), file))


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        lr_filename, hr_filename, filename = self.files[idx]
        lr_file = torch.from_numpy(np.load(lr_filename))
        hr_file = torch.from_numpy(np.load(hr_filename))
        return lr_file.unsqueeze(0), hr_file.unsqueeze(0), filename

class SRDataset(Dataset):
    def __init__(self, hparams: dict, files: list[str]) -> None:
        super().__init__()
        
        planetscope_dir = DATA_DIR.joinpath(hparams["planetscope_bands"])
        planetscope_lr_dir = DATA_DIR.joinpath(f"{hparams['planetscope_bands']}_lr")
        self.planetscope_lr_files = [planetscope_lr_dir.joinpath(filename) for filename in files]
        self.planetscope_files = [planetscope_dir.joinpath(filename) for filename in files]
        self.augment = hparams["datamodule"]["augment"]

        print(f"Dataset size: {len(self.planetscope_files) * 8 if self.augment else len(self.planetscope_files)}")

    def __len__(self):
        if not self.augment:
            return len(self.planetscope_files)
        else:
            return len(self.planetscope_files) * 8
    
    def __getitem__(self, idx):
        if self.augment:
            index = idx // 8
            flip = idx % 8 >= 4
            rotate = idx % 4
            planetscope_lr_file = torch.rot90(torch.from_numpy(np.load(self.planetscope_lr_files[index])), rotate)
            planetscope_file = torch.rot90(torch.from_numpy(np.load(self.planetscope_files[index])), rotate)
            if flip:
                planetscope_lr_file = torch.flip(planetscope_lr_file, [1])
                planetscope_file = torch.flip(planetscope_file, [1])
        else:
            planetscope_lr_file = torch.from_numpy(np.load(self.planetscope_lr_files[idx]))
            planetscope_file = torch.from_numpy(np.load(self.planetscope_files[idx]))
        return planetscope_lr_file.unsqueeze(0), planetscope_file.unsqueeze(0)

class SRDataModule(pl.LightningDataModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.params = hparams
        self.batch_size = hparams["datamodule"]["batch_size"]
        self.sentinel_resolution = hparams["sentinel_resolution"]
        self.planetscope_bands = hparams["planetscope_bands"]

        self.files = [file.name for file in DATA_DIR.joinpath(self.planetscope_bands).iterdir()]

        # planetscope_folder = DATA_DIR.joinpath(self.planetscope_bands)
        # planetscope_lr_folder = DATA_DIR.joinpath(f"{self.planetscope_bands}_lr")

        # psnr_score = []
        # ssim_score = []

        # for file in self.files:
        #     ps_file = np.load(planetscope_folder.joinpath(file))
        #     ps_lr_file = np.load(planetscope_lr_folder.joinpath(file))

        #     upsampled_file = cv2.resize(ps_lr_file, (150, 150), interpolation=cv2.INTER_CUBIC)
        #     ps_psnr = psnr(upsampled_file, ps_file)
        #     ps_ssim, _ = ssim((upsampled_file * (255. / 8.)).astype(np.uint8), (ps_file * (255. / 8.)).astype(np.uint8), full=True)


        #     psnr_score.append(ps_psnr)
        #     ssim_score.append(ps_ssim)
        
        total_size = len(self.files)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        self.train_set, self.val_set = random_split(
            dataset = self.files, 
            lengths = [train_size, val_size], 
            generator =torch.Generator().manual_seed(hparams["random_seed"])
        )

        self.predict_files = DATA_DIR.joinpath(self.sentinel_resolution + "_in_situ")

    def setup(self, stage=None):
        # if stage == 'fit':
            self.train_dataset = SRDataset(self.params, self.files)
            self.val_dataset = SRDataset(self.params, self.val_set)
        # elif stage == 'predict':
            files_in_situ = [file.name for file in self.predict_files.iterdir()]
            self.predict_dataset = SRPredictDataset(self.params, files_in_situ)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)