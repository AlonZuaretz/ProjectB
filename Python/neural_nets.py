import torch.nn as nn
import torch

from NN_blocks import ConvBlock, ResidualBlock

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class Stage1Network(nn.Module):
    def __init__(self):
        super(Stage1Network, self).__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 4),
            ConvBlock(4, 16),
            ResidualBlock(16),
            ConvBlock(16, 32),
            ConvBlock(32, 32),
            ResidualBlock(32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            ResidualBlock(64),
            ResidualBlock(64),
            )
        self.output_conv = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16*4*4, 16)
        self.fc2 = nn.Linear(16*4*4, 16)
        self.paramaters_num = count_parameters(self)

    def forward(self, x):
        x = self.net(x)
        x = self.output_conv(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x1_mat = x1.view(x.size(0), 4, 4)
        x2_mat = x2.view(x.size(0), 4, 4)
        x = torch.cat((x1_mat.unsqueeze(1), x2_mat.unsqueeze(1)), dim=1)
        return x


class Stage2Network(nn.Module):
    def __init__(self):
        super(Stage2Network, self).__init__()
        self.pre_fc1 = nn.Linear(16,16)
        self.pre_fc2 = nn.Linear(16, 16)
        self.net = nn.Sequential(
            ConvBlock(2, 4),
            ConvBlock(4, 16),
            ResidualBlock(16),
        )
        self.output_conv = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        ch1 = x[:,0,:,:]
        ch2 = x[:,1,:,:]
        pre1 = ch1.view(ch1.size(0), 16)
        pre2 = ch2.view(ch2.size(0), 16)
        x1 = self.pre_fc1(pre1)
        x2 = self.pre_fc2(pre2)
        x1 = x1.view(x1.size(0),4,4)
        x2 = x2.view(x2.size(0),4,4)
        x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1)), dim=1)
        x = self.net(x)
        x = self.output_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
