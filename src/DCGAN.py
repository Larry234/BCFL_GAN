import torch
import torch.nn as nn


class ConvTransBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        pass
class Generator(nn.Module):
    
    def __init__(self, laten_dim, ngf=64):
        super(Generator, self).__init__()
        self.laten_dim = laten_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.laten_dim, ngf * 16, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(),
            # state size. (ngf*16) x 7 x 7
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            # state size. (ngf*8) x 14 x 14
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            # state size. (ngf*8) x 28 x 28
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            # state size. (ngf*8) x  x 28
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            # state size. (ngf*8) x 14 x 14

        )