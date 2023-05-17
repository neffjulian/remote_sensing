"""
 Enhanced Deep Super-Resolution Network (2017)

Paper: https://arxiv.org/pdf/1707.02921v1.pdf
"""

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
        
        self.entries = entries

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        s2_filename = self.s2_tiles.joinpath(self.entries[idx])
        ps_filename = self.ps_tiles.joinpath(self.entries[idx])

        np_s2 = np.nan_to_num(np.load(s2_filename))
        np_ps = np.nan_to_num(np.load(ps_filename))

        s2_file = torch.from_numpy(np_s2).unsqueeze(0).unsqueeze(0)
        ps_file = torch.from_numpy(np_ps).unsqueeze(0)

        output_tensor = F.interpolate(s2_file, size=(162, 162), mode='bicubic').squeeze(0)

        return output_tensor, ps_file
    
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

def show_random_result(train_model, train_dataset):
    rand_int = random.randint(1, 12528)
    test_input, test_output = train_dataset[rand_int]
    test_image = train_model(test_input.unsqueeze(0))
    cv2.imshow("Original Input", test_input.squeeze().numpy())
    cv2.imshow("Model output", test_image.detach().squeeze().numpy())
    cv2.imshow("Original Output", test_output.squeeze().numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()