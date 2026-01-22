# src/ca_fusenet/models/pose/encoders/stgcn_encoder.py
from __future__ import annotations

import torch
import torch.nn as nn

from ...stgcn.backbone import STGCNBackbone


class PoseSTGCNEncoder(nn.Module):
    """
    Wrapper pose encoder.
    Input: (B, C, T, V)
    Output: (B, D)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_joints: int = 17,
        embed_dim: int = 256,
        strategy: str = "uniform",
        max_hop: int = 1,
        dilation: int = 1,
        temporal_kernel_size: int = 9,
        dropout: float = 0.5,
        edge_importance: bool = False,
        use_data_bn: bool = True,
        channels: tuple[int, ...] = (64, 64, 64, 64, 128, 128, 128, 256, 256, 256),
        strides: tuple[int, ...] = (1, 1, 1, 1, 2, 1, 1, 2, 1, 1),
    ):
        super().__init__()
        self.backbone = STGCNBackbone(
            in_channels=in_channels,
            num_joints=num_joints,
            embed_dim=embed_dim,
            strategy=strategy,
            max_hop=max_hop,
            dilation=dilation,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
            edge_importance=edge_importance,
            use_data_bn=use_data_bn,
            channels=channels,
            strides=strides,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B,C,T,V)
        return self.backbone(x)
