from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F

DATA_DIR = Path().absolute().parent.parent.joinpath('data', 'tiles')

class SRDataset(Dataset):
    def __init__(self, entries: list[str], sentinel_resolution: str, planetscope_bands: str):
        if sentinel_resolution != "10m" and sentinel_resolution != "20m":
            raise Exception(f"Invalid value for 'sentinel_resolution'. Choose one of: '10m' or '20m'")
        if planetscope_bands != "4b" and planetscope_bands != "8b":
            raise Exception(f"Invalid value for 'planetscope_bands'. Choose one of: '4b' or '8b'")
        
        self.s2_tiles = DATA_DIR.joinpath(f"sentinel_{sentinel_resolution}")
        self.ps_tiles = DATA_DIR.joinpath(f"planetscope_{planetscope_bands}")

        self.s2_entries = []
        self.ps_entries = []
        
        for entry in entries:
            s2_filename = self.s2_tiles.joinpath(entry)
            ps_filename = self.ps_tiles.joinpath(entry)

            np_s2 = (np.load(s2_filename))
            np_ps = (np.load(ps_filename))

            s2_file = torch.from_numpy(np_s2).unsqueeze(0).unsqueeze(0)
            ps_file = torch.from_numpy(np_ps).unsqueeze(0)

            s2_interpolated = F.interpolate(s2_file, size=(324, 324), mode='bicubic').squeeze(0)

            self.s2_entries.append(s2_interpolated)
            self.ps_entries.append(ps_file)

    def __len__(self):
        return len(self.s2_entries)
    
    def __getitem__(self, idx):
        return self.s2_entries[idx], self.ps_entries[idx]

def get_entries(sentinel_resolution: str, planetscope_bands: str):
    s2_tiles = DATA_DIR.joinpath(f"sentinel_{sentinel_resolution}")
    ps_tiles = DATA_DIR.joinpath(f"planetscope_{planetscope_bands}")

    s2_entries = [file.name for file in s2_tiles.iterdir()]
    ps_entries = [file.name for file in ps_tiles.iterdir()]

    if len([file for file in s2_entries if file not in ps_entries]) > 0:
        raise Exception("Invalid dirs.")
    
    return s2_entries

def get_datasets(sentinel_resolution: str, planetscope_bands: str):
    entries = get_entries(sentinel_resolution, planetscope_bands)
    len_train = int(len(entries) * 0.9)
    len_valid = len(entries) - len_train
    lengths = [len_train, len_valid]
    train, validation = random_split(entries, lengths)
    
    train_dataset = SRDataset(train, sentinel_resolution, planetscope_bands)
    val_dataset = SRDataset(validation, sentinel_resolution, planetscope_bands)
    return train_dataset, val_dataset

def show_random_result(train_model, sentinel_resolution, planetscope_bands):
    tiles = 2
    while(True):
        rand_int = random.randint(1, 296) 
        filename = f"{rand_int:04d}"

        s2 = [file for file in DATA_DIR.joinpath(f"sentinel_{sentinel_resolution}").iterdir() if filename in file.name]
        ps = [file for file in DATA_DIR.joinpath(f"planetscope_{planetscope_bands}").iterdir() if filename in file.name]
        if len(s2) == len(ps):
            break

    np_s2 = [np.load(filename) for filename in s2]
    np_ps = [np.load(filename) for filename in ps]

    torch_s2 = [torch.from_numpy(np).unsqueeze(0).unsqueeze(0) for np in np_s2]
    s2_interpolated = [F.interpolate(s2, size=(324, 324), mode='bicubic').squeeze(0) for s2 in torch_s2]

    model_in = [data.numpy() for data in s2_interpolated]

    model_out = [train_model(input_s2.unsqueeze(0)).detach().squeeze().numpy() for input_s2 in s2_interpolated]
    LR = np.empty((648, 648))
    HR = np.empty((648, 648))
    SR = np.empty((648, 648))

    size = int(648/tiles)
    for i in range(tiles):
        for j in range(tiles):
            LR[i*size:(i+1)*size,j*size:(j+1)*size] = model_in[i*2+j]
            HR[i*size:(i+1)*size,j*size:(j+1)*size] = np_ps[i*2+j]
            SR[i*size:(i+1)*size,j*size:(j+1)*size] = model_out[i*2+j]

    cv2.imshow("Original Input", LR)
    cv2.imshow("Model output", SR)
    cv2.imshow("Original Output", HR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()