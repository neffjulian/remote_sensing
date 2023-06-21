from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import pytorch_lightning as pl

DATA_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'processed')

class SRPredictDataset(Dataset):
    def __init__(self, sentinel_resolution: str, planetscope_band:str, files: list[str], predict: bool = False) -> None:
        super().__init__()

        if predict:
            self.sentinel_dir = DATA_DIR.joinpath("in_situ", sentinel_resolution)
            self.planetscope_dir = DATA_DIR.joinpath("in_situ", planetscope_band[-2:])
        else:
            self.sentinel_dir = DATA_DIR.joinpath(sentinel_resolution)
            self.planetscope_dir = DATA_DIR.joinpath(planetscope_band[-2:])

        self.sentinel_files = [self.sentinel_dir.joinpath(filename) for filename in files]
        self.planetscope_files = [self.planetscope_dir.joinpath(filename) for filename in files]

    def __len__(self):
        return len(self.sentinel_files)
    
    def __getitem__(self, idx):
        sentinel_file = torch.from_numpy(np.load(self.sentinel_files[idx]))
        planetscope_file = torch.from_numpy(np.load(self.planetscope_files[idx]))
        return sentinel_file.unsqueeze(0), planetscope_file.unsqueeze(0), self.sentinel_files[idx].name

class SRDataset(Dataset):
    def __init__(self, sentinel_resolution: str, planetscope_band:str, files: list[str], predict: bool = False) -> None:
        super().__init__()

        if predict:
            self.sentinel_dir = DATA_DIR.joinpath("in_situ", sentinel_resolution)
            self.planetscope_dir = DATA_DIR.joinpath("in_situ", planetscope_band)
        else:
            self.sentinel_dir = DATA_DIR.joinpath(sentinel_resolution)
            self.planetscope_dir = DATA_DIR.joinpath(planetscope_band)

        self.sentinel_files = [self.sentinel_dir.joinpath(filename) for filename in files]
        self.planetscope_files = [self.planetscope_dir.joinpath(filename) for filename in files]

    def __len__(self):
        return len(self.sentinel_files)
    
    def __getitem__(self, idx):
        sentinel_file = torch.from_numpy(np.load(self.sentinel_files[idx],  mmap_mode='r'))
        planetscope_file = torch.from_numpy(np.load(self.planetscope_files[idx],  mmap_mode='r'))
        return sentinel_file.unsqueeze(0), planetscope_file.unsqueeze(0)

class SRDataModule(pl.LightningDataModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.batch_size = hparams["datamodule"]["batch_size"]
        self.sentinel_resolution = hparams["sentinel_resolution"]
        self.planetscope_bands = hparams["planetscope_bands"]

        self.files = [file.name for file in DATA_DIR.joinpath(self.planetscope_bands).iterdir()]
        
        total_size = len(self.files)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        self.train_set, self.val_set = random_split(
            dataset = self.files, 
            lengths = [train_size, val_size], 
            generator =torch.Generator().manual_seed(hparams["random_seed"])
        )

        self.predict_files = DATA_DIR.joinpath("in_situ", self.planetscope_bands[-2:])

    def setup(self, stage=None):
        # if stage == 'fit':
            self.train_dataset = SRDataset(self.sentinel_resolution, self.planetscope_bands, self.train_set)
            self.val_dataset = SRDataset(self.sentinel_resolution, self.planetscope_bands, self.val_set)
        # elif stage == 'predict':
            files_in_situ = [file.name for file in self.predict_files.iterdir()]
            self.predict_dataset = SRPredictDataset(self.sentinel_resolution, self.planetscope_bands, files_in_situ, predict = True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, pin_memory=True)