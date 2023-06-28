import torch

def psnr(mse):
    return 20 * torch.log10(1. / torch.sqrt(mse))