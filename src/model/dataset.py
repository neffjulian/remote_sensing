from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F

DATA_DIR = Path().absolute().parent.parent.joinpath('data', 'tiles')

class SRDataset(Dataset):
    def __init__(self, entries):
        self.s2_tiles = DATA_DIR.joinpath("sentinel")
        self.ps_tiles = DATA_DIR.joinpath("planetscope")

        self.s2_entries = []
        self.ps_entries = []
        
        for entry in entries:
            s2_filename = self.s2_tiles.joinpath(entry)
            ps_filename = self.ps_tiles.joinpath(entry)

            np_s2 = np.nan_to_num(np.load(s2_filename))
            np_ps = np.nan_to_num(np.load(ps_filename))

            s2_file = torch.from_numpy(np_s2).unsqueeze(0).unsqueeze(0)
            ps_file = torch.from_numpy(np_ps).unsqueeze(0)

            s2_interpolated = F.interpolate(s2_file, size=(162, 162), mode='bicubic').squeeze(0)

            self.s2_entries.append(s2_interpolated)
            self.ps_entries.append(ps_file)

        self.get_mean()

    def __len__(self):
        return len(self.s2_entries)
    
    def __getitem__(self, idx):
        return self.s2_entries[idx], self.ps_entries[idx]
    
    def get_mean(self):
        sum = torch.zeros((1, 162, 162))
        for entry in self.s2_entries:
            sum += entry

        for entry in self.ps_entries:
            sum += entry

        mean = sum / (len(self.s2_entries) + len(self.ps_entries))

        sum_sq_diff = torch.zeros((1, 162, 162))
        for entry in self.s2_entries:
            sum_sq_diff += (entry - mean) ** 2

        for entry in self.ps_entries:
            sum_sq_diff += (entry - mean) ** 2

        std = sum_sq_diff / (len(self.s2_entries) + len(self.ps_entries))
        
        return mean, std

def get_entries():
    s2_tiles = DATA_DIR.joinpath("sentinel")
    ps_tiles = DATA_DIR.joinpath("planetscope")

    s2_entries = [file.name for file in s2_tiles.iterdir()]
    ps_entries = [file.name for file in ps_tiles.iterdir()]

    if len([file for file in s2_entries if file not in ps_entries]) > 0:
        raise Exception("Invalid dirs.")
    
    return s2_entries

def get_datasets():
    entries = get_entries()
    lengths = [int(len(entries)*0.9), int(len(entries)*0.1)]
    train, validation = random_split(entries, lengths)
    
    train_dataset = SRDataset(train)
    val_dataset = SRDataset(validation)
    return train_dataset, val_dataset

def show_random_result(train_model):
    while(True):
        rand_int = random.randint(1, 296) 
        filename = f"{rand_int:04d}"

        s2 = [file for file in DATA_DIR.joinpath("sentinel").iterdir() if filename in file.name]
        ps = [file for file in DATA_DIR.joinpath("planetscope").iterdir() if filename in file.name]
        if len(s2) == len(ps):
            break

    np_s2 = [np.nan_to_num(np.load(filename)) for filename in s2]
    np_ps = [np.nan_to_num(np.load(filename)) for filename in ps]

    torch_s2 = [torch.from_numpy(np).unsqueeze(0).unsqueeze(0) for np in np_s2]
    s2_interpolated = [F.interpolate(s2, size=(162, 162), mode='bicubic').squeeze(0) for s2 in torch_s2]

    model_in = [data.numpy() for data in s2_interpolated]

    model_out = [train_model(input_s2.unsqueeze(0)).detach().squeeze().numpy() for input_s2 in s2_interpolated]
    
    LR = np.empty((648, 648))
    HR = np.empty((648, 648))
    SR = np.empty((648, 648))

    for i in range(4):
        for j in range(4):
            LR[i*162:(i+1)*162,j*162:(j+1)*162] = model_in[i*4+j]
            HR[i*162:(i+1)*162,j*162:(j+1)*162] = np_ps[i*4+j]
            SR[i*162:(i+1)*162,j*162:(j+1)*162] = model_out[i*4+j]

    cv2.imshow("Original Input", LR)
    cv2.imshow("Model output", SR)
    cv2.imshow("Original Output", HR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()