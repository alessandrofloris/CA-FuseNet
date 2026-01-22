# src/ca_fusenet/models/pose/baselines/stgcn_baseline.py
from __future__ import annotations

import torch
import torch.nn as nn

from ..encoders.stgcn_encoder import PoseSTGCNEncoder


class PoseSTGCNBaseline(nn.Module):
    """
    Pose-only baseline: ST-GCN encoder + linear head.
    Input: (B, C, T, V)
    Output: logits (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        num_joints: int = 17,
        embed_dim: int = 256,
        dropout_head: float = 0.0,
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = PoseSTGCNEncoder(
            in_channels=in_channels,
            num_joints=num_joints,
            embed_dim=embed_dim,
            **encoder_kwargs,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout_head) if dropout_head > 0 else nn.Identity(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)          # (B, D)
        logits = self.head(emb)        # (B, num_classes)
        return logits
