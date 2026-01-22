import torch.nn as nn

class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        padding = ((kernel_size - 1) // 2, 0)

        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=padding),
            nn.BatchNorm2d(channels),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x):
        return self.net(x)