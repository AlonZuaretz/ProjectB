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
        self.parameters_num = count_parameters(self)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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


