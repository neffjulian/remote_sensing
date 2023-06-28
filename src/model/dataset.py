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

        self.sentinel_dir = DATA_DIR.joinpath("in_situ", hparams["sentinel_resolution"])
        self.planetscope_dir = DATA_DIR.joinpath("in_situ", hparams["planetscope_bands"][-2:])
 
        self.sentinel_files = [self.sentinel_dir.joinpath(filename) for filename in files]
        self.planetscope_files = [self.planetscope_dir.joinpath(filename) for filename in files]

    def __len__(self):
        return len(self.sentinel_files)
    
    def __getitem__(self, idx):
        sentinel_file = torch.from_numpy(cv2.resize(np.load(self.sentinel_files[idx]), (160, 160), interpolation=cv2.INTER_CUBIC))
        planetscope_file = torch.from_numpy(np.load(self.planetscope_files[idx]))
        return sentinel_file.unsqueeze(0), planetscope_file.unsqueeze(0), self.sentinel_files[idx].name

class SRDataset(Dataset):
    def __init__(self, hparams: dict, files: list[str], train_dataset: bool) -> None:
        super().__init__()
        
        sentinel_dir = DATA_DIR.joinpath(hparams["sentinel_resolution"])
        planetscope_dir = DATA_DIR.joinpath(hparams["planetscope_bands"])

        sentinel_files = [sentinel_dir.joinpath(filename) for filename in files]
        print("Original number of files:", len(sentinel_files))
        planetscope_files = [planetscope_dir.joinpath(filename) for filename in files]

        if train_dataset:
            # With psnr_threshold = 20.0 and ssim_threshold = 0.5 we remove 46% of files
            psnr_threshold = 20.0
            ssim_threshold = 0.6

            to_drop = []

            for i, (s2_file, ps_file) in enumerate(zip(sentinel_files, planetscope_files)):
                s2_data = cv2.resize(np.load(s2_file), (160, 160), interpolation=cv2.INTER_CUBIC)
                ps_data = np.load(ps_file)

                if psnr(s2_data, ps_data) < psnr_threshold or ssim((s2_data * (255 / 8)).astype(np.uint8), (ps_data * (255 / 8)).astype(np.uint8), full=True)[0] < ssim_threshold:
                    to_drop.append(i)

            sentinel_files = [file for i, file in enumerate(sentinel_files) if i not in to_drop]
            planetscope_files = [file for i, file in enumerate(planetscope_files) if i not in to_drop]

        file_pairs = list(zip(sentinel_files, planetscope_files))
        print("Filtered number of files:", len(sentinel_files))

        self.files = [(x, y, i) for x, y in file_pairs for i in range(4)]
        print("Augmented number of files:", len(sentinel_files))


    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        sentinel_filename, planetscope_filename, rotate = self.files[idx]
        sentinel_file = torch.from_numpy(np.load(sentinel_filename)).rot90(rotate)
        planetscope_file = torch.from_numpy(np.load(planetscope_filename)).rot90(rotate)

        return sentinel_file.unsqueeze(0), planetscope_file.unsqueeze(0)

class SRDataModule(pl.LightningDataModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.params = hparams
        self.batch_size = hparams["datamodule"]["batch_size"]
        self.sentinel_resolution = hparams["sentinel_resolution"]
        self.planetscope_bands = hparams["planetscope_bands"]
        self.persis

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
            self.train_dataset = SRDataset(self.params, self.files, True)
            self.val_dataset = SRDataset(self.params, self.val_set, False)
        # elif stage == 'predict':
            files_in_situ = [file.name for file in self.predict_files.iterdir()]
            self.predict_dataset = SRPredictDataset(self.params, files_in_situ)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)