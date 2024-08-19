import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.init_weights()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x + residual
        return x


    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)


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
        self.output_layer = nn.Linear(64*4*4, 8)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        return x
