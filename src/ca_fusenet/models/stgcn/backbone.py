# src/ca_fusenet/models/stgcn/backbone.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.stgcn_block import STGCNBlock
from .graph.coco17 import build_adjacency


class STGCNBackbone(nn.Module):
    """
    Task-agnostic ST-GCN backbone.

    Input:  x (B, C, T, V)  [C=3, V=17]
    Output: emb (B, D)
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

        if num_joints != 17:
            raise ValueError(f"STGCNBackbone currently expects num_joints=17, got {num_joints}.")

        if len(channels) != len(strides):
            raise ValueError("channels and strides must have same length.")

        # Build adjacency once (numpy -> torch), register as buffer.
        A = build_adjacency(strategy=strategy, max_hop=max_hop, dilation=dilation)  # (K, V, V)
        A = torch.tensor(A, dtype=torch.float32)
        self.register_buffer("A", A)  # moves with .to(device)

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.edge_importance_enabled = edge_importance
        self.use_data_bn = use_data_bn

        K = self.A.size(0)

        # Optional "data BN" like original ST-GCN:
        # reshape (B, C, T, V) -> (B, V*C, T) and BN1d(V*C)
        if use_data_bn:
            self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        else:
            self.data_bn = None

        # Build ST-GCN blocks
        blocks: list[nn.Module] = []
        c_in = in_channels
        for i, (c_out, stride) in enumerate(zip(channels, strides)):
            blocks.append(
                STGCNBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    spatial_kernel_size=K,
                    temporal_kernel_size=temporal_kernel_size,
                    stride=stride,
                    dropout=dropout,
                    residual=(i != 0),  # first block residual=False like many baselines
                )
            )
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)

        # Edge-importance weighting (optional)
        if edge_importance:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones_like(self.A)) for _ in range(len(self.blocks))]
            )
        else:
            self.edge_importance = None

        # Projection head to embedding
        self.proj = nn.Sequential(
            nn.Linear(channels[-1], embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, V)  (float32)
        returns: (B, D)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape (B,C,T,V), got {tuple(x.shape)}")

        B, C, T, V = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={C}")
        if V != self.num_joints:
            raise ValueError(f"Expected V={self.num_joints}, got V={V}")

        # data normalization
        if self.data_bn is not None:
            # (B,C,T,V) -> (B,V,C,T) -> (B, V*C, T)
            x_bn = x.permute(0, 3, 1, 2).contiguous().view(B, V * C, T)
            x_bn = self.data_bn(x_bn)
            x = x_bn.view(B, V, C, T).permute(0, 2, 3, 1).contiguous()  # back to (B,C,T,V)

        # forward through blocks
        A = self.A
        for i, block in enumerate(self.blocks):
            if self.edge_importance is not None:
                x, _ = block(x, A * self.edge_importance[i])
            else:
                x, _ = block(x, A)

        # global average pool over (T, V)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)))  # (B, C_last, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, C_last)

        # project to embedding
        emb = self.proj(x)  # (B, D)
        return emb
