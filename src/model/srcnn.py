from pathlib import Path$

import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

DATA_DIR = Path().absolute().parent.parent.joinpath('data', 'tiles')

# From https://arxiv.org/abs/1501.00092
# Based on https://github.com/Mirwaisse/SRCNN

class SuperResolutionDataset(Dataset):
    def __init__(self):
        self.s2_tiles = DATA_DIR.joinpath("sentinel")
        self.ps_tiles = DATA_DIR.joinpath("planetscope")

        s2_entries = [file.name for file in self.s2_tiles.iterdir()]
        ps_entries = [file.name for file in self.ps_tiles.iterdir()]

        if len([file for file in s2_entries if file not in ps_entries]) > 0:
            raise Exception("Invalid dirs.")
        
        self.entries = s2_entries
        # self.s2_data = []
        # self.ps_data = []
        # for entry in self.entries:
        #     s2_filename = self.s2_tiles.joinpath(entry)
        #     ps_filename = self.ps_tiles.joinpath(entry)

        #     np_s2 = np.nan_to_num(np.load(s2_filename))
        #     np_ps = np.nan_to_num(np.load(ps_filename))

        #     s2_file = torch.from_numpy(np_s2).unsqueeze(0).unsqueeze(0)
        #     ps_file = torch.from_numpy(np_ps).unsqueeze(0)

        #     output_tensor = F.interpolate(s2_file, size=(162, 162), mode='bicubic').squeeze(0)

        #     self.s2_data.append(output_tensor)
        #     self.ps_data.append(ps_file)

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

        # return self.s2_data[idx], self.ps_data[idx]

class SuperResNet(nn.Module):
    def __init__(self):
        super(SuperResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

        nn.init.normal_(self.conv1.weight, mean=0, std=0.001)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.001)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.001)

        nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.constant_(self.conv3.bias, 0)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x= nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def main():
    torch.manual_seed(42)
    dataset = SuperResolutionDataset()
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

    model = SuperResNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        [
            {"params": model.conv1.parameters(), "lr": 0.0001},  
            {"params": model.conv2.parameters(), "lr": 0.0001},
            {"params": model.conv3.parameters(), "lr": 0.00001},
        ], lr=0.00001,
    )

    total_loss = .0
    for j in range(5):
        i = 0
        total_loss = .0
        for input_data, output_data in dataloader:
            output = model(input_data)
            loss = criterion(output, output_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            i += 64
            print(i, loss.item())

        print("---------------")
        print("Total loss for epoch ", j, ": ", total_loss)
        print("---------------")

    for i in range(10):
        rand_int = random.randint(1, 13919)
        test_input, test_output = dataset[rand_int]
        test_image = model(test_input.unsqueeze(0))
        cv2.imshow("Original Input", test_input.squeeze().numpy())
        cv2.imshow("Model output", test_image.detach().squeeze().numpy())
        cv2.imshow("Original Output", test_output.squeeze().numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    # dataset = SuperResolutionDataset()
    # s2, ps = dataset[0]
    # print(s2.shape, ps.shape)