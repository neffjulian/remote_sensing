import torch
import torch.nn as nn
import torchvision.models.vgg as vgg

def psnr(mse):
    return 20 * torch.log10(8. / torch.sqrt(mse))

class VGG_loss(nn.Module):
    def __init__(self):
        super(VGG_loss, self).__init__()

        features = vgg.vgg16(pretrained=True).features
        modules = [m for m in features]
        mean = .485
        std = .229

        self.net = nn.Sequential(*modules[:35])
        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        feature_representation = self.net(prediction)
        with torch.no_grad():
            target_representation = self.net(target.detach())
            # TODO: Check if torch.sqrt() helps
        return self.mse(feature_representation, target_representation)