from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import pytorch_lightning as pl

DATA_DIR = Path().absolute().parent.parent.joinpath('data', 'processed')

# class SRDataset(Dataset):
#     def __init__(self, sentinel_bands: str, planetscope_band:str, files: list[str]) -> None:
#         super().__init__()
#         self.sentinel_dir = DATA_DIR.joinpath(sentinel_bands)
#         self.planetscope_dir = DATA_DIR.joinpath(planetscope_band)

#         self.sentinel_files = [torch.from_numpy(np.load(self.sentinel_dir.joinpath(file))) for file in files]
#         self.planetscope_files = [torch.from_numpy(np.load(self.planetscope_dir.joinpath(file))) for file in files]

#     def __len__(self):
#         return len(self.sentinel_files)
    
#     def __getitem__(self, idx):
#         return self.sentinel_files[idx], self.planetscope_files[idx]

class SRDataset(Dataset):
    def __init__(self, sentinel_bands: str, planetscope_band:str, files: list[str], predict: bool = False) -> None:
        super().__init__()

        if predict:
            self.sentinel_dir = DATA_DIR.joinpath("in_situ", sentinel_bands)
            self.planetscope_dir = DATA_DIR.joinpath("in_situ", planetscope_band)
        else:
            self.sentinel_dir = DATA_DIR.joinpath(sentinel_bands)
            self.planetscope_dir = DATA_DIR.joinpath(planetscope_band)

        self.sentinel_files = [self.sentinel_dir.joinpath(filename) for filename in files]
        self.planetscope_files = [self.planetscope_dir.joinpath(filename) for filename in files]

    def __len__(self):
        return len(self.sentinel_files)
    
    def __getitem__(self, idx):
        sentinel_file = torch.from_numpy(np.load(self.sentinel_files[idx]))
        planetscope_file = torch.from_numpy(np.load(self.planetscope_files[idx]))
        return sentinel_file, planetscope_file

class SRDataModule(pl.LightningDataModule):
    def __init__(self, sentinel_bands: str, planetscope_bands: str, data_dir: Path = DATA_DIR, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sentinel_bands = sentinel_bands
        self.planetscope_bands = planetscope_bands

        self.files = [file.name for file in self.data_dir.joinpath("4b").iterdir()]
        
        total_size = len(self.files)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        self.train_set, self.val_set, self.test_set = random_split(self.files, [train_size, val_size, test_size])

        self.predict_files = DATA_DIR.joinpath("in_situ")

    def setup(self, stage=None):
        # if stage == 'fit':
            self.train_dataset = SRDataset(self.sentinel_bands, self.planetscope_bands, self.train_set)
            self.val_dataset = SRDataset(self.sentinel_bands, self.planetscope_bands, self.val_set)
        # elif stage == 'test':
            self.test_dataset = SRDataset(self.sentinel_bands, self.planetscope_bands, self.test_set)
        # elif stage == 'predict':
            files = [file.name for file in self.data_dir.joinpath("in_situ", "4b").iterdir()]
            self.predict_dataset = SRDataset(self.sentinel_bands, self.planetscope_bands, files, predict = True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)