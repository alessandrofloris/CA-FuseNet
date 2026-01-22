import torch
import torch.nn as nn
from ..layers.graph_conv import ConvTemporalGraphical
from ..layers.temporal_conv import TemporalConvBlock

class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_kernel_size: int,   # K
        temporal_kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be odd (for same padding).")
        padding = ((temporal_kernel_size - 1) // 2, 0)

        # GCN
        self.gcn = ConvTemporalGraphical(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=spatial_kernel_size,   # K
            t_kernel_size=1,
            t_stride=1,
            t_padding=0,
        )

        # TCN
        self.tcn = TemporalConvBlock(
            channels=out_channels,
            kernel_size=temporal_kernel_size,
            stride=stride,
            dropout=dropout,
        )
        
        # Residual
        if not residual:
            self.residual = None
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        # x: (B, C, T, V)
        res = 0
        if self.residual is not None:
            res = self.residual(x)

        x, A = self.gcn(x, A)          # (B, Cout, T, V)
        x = self.tcn(x) + res
        return self.relu(x), A