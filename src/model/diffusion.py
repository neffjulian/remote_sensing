import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch

DATA_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'processed', '4b', '03_0000_00.npy')

def visualize(data: np.ndarray):
    plt.imshow(data, cmap='viridis')
    plt.show()

def forward(data: torch.Tensor, timestep: int, alpha: torch.Tensor, beta: torch.Tensor, alpha_bar: torch.Tensor):
    beta_forward = beta[timestep]
    alpha_forward = alpha[timestep]
    alpha_bar_forward = alpha_bar[timestep]

    xt = data * torch.sqrt(alpha_bar_forward) + torch.randn_like(data) * torch.sqrt(1. - alpha_bar_forward)

    return xt

def main():
    data = torch.Tensor(np.load(DATA_DIR))
    # data = torch.Tensor((data - data.min()) / (data.max() - data.min()))

    betas = torch.linspace(0.0001, 0.02, 100)
    alphas = 1. - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    result = torch.zeros((600, 600))
    result[0:150, 0:150] = data
    print(alpha_bar)

    # for i in range(1, 16):
    #     row = i // 4
    #     col = i % 4

    #     result[row * 150:(row + 1) * 150, col * 150:(col + 1) * 150] = forward(data, i - 1, alphas, betas, alpha_bar)

    # visualize(result.numpy())




if __name__ == '__main__':
    main()