import torch

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))