import torch.nn as nn


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
        self.conv_block1 = ConvBlock(1, 4)  # Assuming input has 1 channel
        self.conv_block2 = ConvBlock(4, 16)
        self.res_block1 = ResidualBlock(16)
        self.res_block2 = ResidualBlock(16)
        self.res_block3 = ResidualBlock(16)
        self.output_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.output_conv(x)
        return x
