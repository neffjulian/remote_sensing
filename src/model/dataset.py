"""
Dataset and DataModule classes for loading and processing the satellite images dataset.

@date: 2023-08-30
@author: Julian Neff, ETH Zurich

Copyright (C) 2023 Julian Neff

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

DATA_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'processed')

class SRDataset(Dataset):
    """
    A PyTorch Dataset for loading low and high resolution satellite images.
    """
    def __init__(self, hparams: dict, files: list[str]) -> None:
        """
        Args:
        hparams: dict
            A dictionary of hyperparameters.
        files: list of str
            A list of filenames to load.
        """
        super().__init__()
        
        planetscope_dir = DATA_DIR.joinpath(hparams["planetscope_bands"])
        planetscope_lr_dir = DATA_DIR.joinpath(f"{hparams['planetscope_bands']}_lr")
        self.planetscope_lr_files = [planetscope_lr_dir.joinpath(filename) for filename in files]
        self.planetscope_files = [planetscope_dir.joinpath(filename) for filename in files]
        self.augment = hparams["datamodule"]["augment"]

        assert [file.name for file in self.planetscope_lr_files] == [file.name for file in self.planetscope_files]
        print(f"Dataset size: {len(self.planetscope_files) * 8 if self.augment else len(self.planetscope_files)}")

    def __len__(self):
        """
        Returns the number of examples in the dataset.
        """
        if not self.augment:
            return len(self.planetscope_files)
        else:
            return len(self.planetscope_files) * 8
    
    def __getitem__(self, idx):
        """
        Loads and returns a single example from the dataset.
        
        Args:
        idx: int
            The index of the example to load.
            
        Returns:
        tuple of torch.Tensors
            The low and high resolution images.
        """
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
    """
    A PyTorch Lightning DataModule for loading and processing the satellite images dataset.
    """
    def __init__(self, hparams: dict):
        """
        Args:
        hparams: dict
            A dictionary of hyperparameters.
        """
        super().__init__()
        self.params = hparams
        self.batch_size = hparams["datamodule"]["batch_size"]
        self.sentinel_resolution = hparams["sentinel_resolution"]
        self.planetscope_bands = hparams["planetscope_bands"]

        self.files = [file.name for file in DATA_DIR.joinpath(self.planetscope_bands).iterdir()]

        test_files = ['0000', '0001', '0002', '0003', '0004', '0006', '0008', '0011', '0012', '0023', '0025', '0026', '0028', '0029', '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0040', '0046']

        files = [file for file in self.files if file[3:7] not in test_files]

        train_size = int(len(files) * 0.8)
        val_size = len(files) - train_size

        self.train_set, self.val_set = torch.utils.data.random_split(files, [train_size, val_size], generator=torch.Generator().manual_seed(hparams["random_seed"]))
        test_files = [file for file in self.files if file[3:7] in test_files]
        print(f"Train set size: {len(self.train_set)}, {len(self.train_set) / len(self.files) * 100}%")
        print(f"Val set size: {len(self.val_set)}, {len(self.val_set) / len(self.files) * 100}%")
        print(f"Test set size: {len(test_files)}, {len(test_files) / len(self.files) * 100}%")

    def setup(self, stage=None):
        """
        Sets up the dataset for training and validation.
        
        Args:
        stage: str or None
            The current stage of training ('train', 'val', 'test', or None). Not used here.
        """
        self.train_dataset = SRDataset(self.params, self.train_set)
        self.val_dataset = SRDataset(self.params, self.val_set)

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)