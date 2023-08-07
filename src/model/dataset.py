from math import log10, sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

DATA_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'processed')

def psnr(x, y):
    return 20 * log10(8. / sqrt(np.mean((x - y) ** 2)))

class SRDataset(Dataset):
    def __init__(self, hparams: dict, files: list[str]) -> None:
        super().__init__()
        
        planetscope_dir = DATA_DIR.joinpath(hparams["planetscope_bands"])
        planetscope_lr_dir = DATA_DIR.joinpath(f"{hparams['planetscope_bands']}_lr")
        self.planetscope_lr_files = [planetscope_lr_dir.joinpath(filename) for filename in files]
        self.planetscope_files = [planetscope_dir.joinpath(filename) for filename in files]
        self.augment = hparams["datamodule"]["augment"]

        assert [file.name for file in self.planetscope_lr_files] == [file.name for file in self.planetscope_files]
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

        val_file = ['0000', '0001', '0002', '0003', '0004', '0006', '0008', '0011', '0012', '0023', '0025', '0026', '0028', '0029', '0030', '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0040', '0046']

        self.train_set = [file for file in self.files if file[3:7] not in val_file]
        self.val_set = [file for file in self.files if file[3:7] in val_file]

        print(f"Train set size: {len(self.train_set)}")
        print(f"Val set size: {len(self.val_set)}")

    def setup(self, stage=None):
            self.train_dataset = SRDataset(self.params, self.train_set)
            self.val_dataset = SRDataset(self.params, self.val_set)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)