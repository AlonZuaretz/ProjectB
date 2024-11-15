import torch.nn as nn

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
        self.output_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.paramaters_num = count_parameters(self)

    def forward(self, x):
        x = self.net(x)
        x = self.output_conv(x)
        return x


class Stage2Network(nn.Module):
    def __init__(self):
        super(Stage2Network, self).__init__()
        self.net = nn.Sequential(
            ConvBlock(2, 4),
            ConvBlock(4, 16),
            ResidualBlock(16),
            ConvBlock(16, 32),
            ConvBlock(32, 32),
            ResidualBlock(32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            ResidualBlock(64),
        )
        self.output_layer = nn.Linear(64*4*4, 16)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        return x
