import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os


class SFTLayer(nn.Module):
    def __init__(self, s1, s2, k, p):
        super(SFTLayer, self).__init__()

        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64*s1, kernel_size=k, stride=s2,padding=p)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64*s1, kernel_size=k, stride=s2,padding=p)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
       # print(scale.size())
       # print(x[0].size())
        return x[0] * (scale + 1) + shift


class CondLayer(nn.Module):
    def __init__(self):
        super(CondLayer, self).__init__()
        self.CondNet = nn.Sequential(
            nn.Conv2d(2, 64, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
        )

    def forward(self, x):
        out = self.CondNet(x)
        return out
