import torch
import torch.nn as nn
import torchvision.models.vgg as vgg

def mse(y_hat, y):
    return torch.mean((y_hat - y) ** 2)

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

class VGG_loss(nn.Module):
     def __init__(self, name: str):
        super(VGG_loss, self).__init__()

        self.net = vgg