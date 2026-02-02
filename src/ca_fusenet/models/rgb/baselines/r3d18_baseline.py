from __future__ import annotations

import torch
from torch import nn

from ca_fusenet.models.rgb.encoders.r3d18_encoder import R3D18Encoder


class RGBOnlyBaseline(nn.Module):
    """
    RGB-only baseline:
    video -> encoder -> global feature -> FC -> logits (num_classes)
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        feature_dim: int | None = None,   # optional projection dim
        dropout_head: float = 0.2,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")

        self.encoder = R3D18Encoder(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            out_dim=feature_dim,
            proj_dropout=dropout_head if feature_dim is not None else 0.0,
        )

        dv = self.encoder.out_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout_head),
            nn.Linear(dv, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)   # [B,dv]
        logits = self.head(f) # [B,num_classes]
        return logits
