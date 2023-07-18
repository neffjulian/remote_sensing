import math
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

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

class PositionalEncoding(nn.Module):
    def __init__(self, max_time_steps: int = 1000, embedding_size: int = 512, n: int = 10000) -> None:
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False)
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.pos_embeddings[t, :]

def main():
    data = torch.Tensor(np.load(DATA_DIR))
    # data = torch.Tensor((data - data.min()) / (data.max() - data.min()))

    # betas = torch.linspace(0.0001, 0.02, 100)
    # alphas = 1. - betas
    # alpha_bar = torch.cumprod(alphas, dim=0)
    # result = torch.zeros((600, 600))
    # result[0:150, 0:150] = data
    # print(alpha_bar)

    pos_encoding = PositionalEncoding()
    print(pos_encoding(torch.Tensor([1])).shape)

    # for i in range(1, 16):
    #     row = i // 4
    #     col = i % 4

    #     result[row * 150:(row + 1) * 150, col * 150:(col + 1) * 150] = forward(data, i - 1, alphas, betas, alpha_bar)

    # visualize(result.numpy())




if __name__ == '__main__':
    main()