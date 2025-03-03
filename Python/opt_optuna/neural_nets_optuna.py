import torch.nn as nn
import torch

from NNs.NN_blocks import ConvBlock, ResidualBlock


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Stage1NetworkOptuna(nn.Module):
    def __init__(self, architecture_choice):
        """
        Initialize the Stage1Network based on the chosen architecture.
        :param architecture_choice: An integer or string representing the architecture choice.
        """
        super(Stage1NetworkOptuna, self).__init__()

        architectures = {

            # Current architecture (kept unchanged)
            0: [("conv", 1, 4), ("conv", 4, 16), ("res", 16),
                ("conv", 16, 32), ("conv", 32, 32), ("res", 32),
                ("conv", 32, 64), ("conv", 64, 64), ("res", 64), ("res", 64)],

            # Proposal 2: Increased depth with gradual channel expansion
            1: [("conv", 1, 8), ("res", 8), ("conv", 8, 16), ("res", 16),
                ("conv", 16, 32), ("res", 32), ("conv", 32, 64), ("conv", 64, 128),
                ("res", 128)],

            # Proposal 3: Wider network with additional conv-res layers
            2: [("conv", 1, 8), ("res", 8), ("conv", 8, 16), ("res", 16),
                ("conv", 16, 32), ("res", 32), ("conv", 32, 64), ("conv", 64, 128),
                ("res", 128), ("conv", 128, 256), ("res", 256)],

            # Proposal 4: Balanced configuration with deeper and wider layers
            3: [("conv", 1, 16), ("res", 16), ("conv", 16, 32), ("res", 32),
                ("conv", 32, 64), ("res", 64), ("conv", 64, 128), ("res", 128),
                ("conv", 128, 256), ("res", 256)],

            # Proposal 5: Deeper network with additional residual blocks and expanded channels
            4: [("conv", 1, 8), ("res", 8), ("conv", 8, 16), ("res", 16),
                ("conv", 16, 32), ("res", 32), ("conv", 32, 64), ("conv", 64, 128),
                ("res", 128), ("conv", 128, 256), ("res", 256), ("conv", 256, 512),
                ("res", 512)],
        }

        chosen_arch = architectures[architecture_choice]

        layers = []
        for block in chosen_arch:
            layer_type = block[0]
            if layer_type == "conv":
                in_channels, out_channels = block[1], block[2]
                layers.append(ConvBlock(in_channels, out_channels))
            elif layer_type == "res":
                in_channels = block[1]
                layers.append(ResidualBlock(in_channels))

        last_block = chosen_arch[-1]

        if last_block[0] == "conv":
            out_channels = last_block[2]  # Extract the out_channels from conv block
        elif last_block[0] == "res":
            out_channels = last_block[1]  # Extract in_channels as out_channels in res block

        self.net = nn.Sequential(*layers)
        self.output_conv = nn.Conv2d(out_channels, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 4 * 4, 16)
        self.fc2 = nn.Linear(16 * 4 * 4, 16)
        self.parameters_num = count_parameters(self)
        self.initialize_weights()

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

