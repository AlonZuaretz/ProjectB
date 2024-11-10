import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='leaky_relu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = self._get_activation(activation)
        self.init_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def _get_activation(self, activation):
        if activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, activation='leaky_relu'):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(channels)
        self.activation = self._get_activation(activation)
        self.init_weights()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x + residual
        return x

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def _get_activation(self, activation):
        if activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")


class Stage1Network(nn.Module):
    def __init__(self, kernel_size=3, activation='leaky_relu'):
        super(Stage1Network, self).__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.net = nn.Sequential(
            ConvBlock(1, 4, kernel_size, activation),
            ConvBlock(4, 16, kernel_size, activation),
            ResidualBlock(16, kernel_size, activation),
            ConvBlock(16, 32, kernel_size, activation),
            ConvBlock(32, 32, kernel_size, activation),
            ResidualBlock(32, kernel_size, activation),
            ConvBlock(32, 64, kernel_size, activation),
            ConvBlock(64, 64, kernel_size, activation),
            ResidualBlock(64, kernel_size, activation),
            ResidualBlock(64, kernel_size, activation),
        )
        self.output_conv = nn.Conv2d(64, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.paramaters_num = count_parameters(self)

    def forward(self, x):
        x = self.net(x)
        x = self.output_conv(x)
        return x


class Stage2Network(nn.Module):
    def __init__(self, kernel_size=3, activation='leaky_relu'):
        super(Stage2Network, self).__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.net = nn.Sequential(
            ConvBlock(2, 4, kernel_size, activation),
            ConvBlock(4, 16, kernel_size, activation),
            ResidualBlock(16, kernel_size, activation),
            ConvBlock(16, 32, kernel_size, activation),
            ConvBlock(32, 32, kernel_size, activation),
            ResidualBlock(32, kernel_size, activation),
            ConvBlock(32, 64, kernel_size, activation),
            ConvBlock(64, 64, kernel_size, activation),
            ResidualBlock(64, kernel_size, activation),
        )
        self.output_layer = nn.Linear(64 * 4 * 4, 8)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        return x
