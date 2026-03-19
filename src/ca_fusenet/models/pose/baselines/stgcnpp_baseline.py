from __future__ import annotations

import torch
import torch.nn as nn

from ..encoders.stgcnpp_encoder import STGCNPPEncoder


class PoseSTGCNPPBaseline(nn.Module):
    """
    Pose-only baseline: STGCN++ encoder + linear head.
    Input: (B, C, T, V)
    Output: logits (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 37,
        d_output: int = 256,
        dropout_head: float = 0.0,
        pretrained: str | None = None,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = STGCNPPEncoder(
            d_output=d_output,
            pretrained=pretrained,
            **encoder_kwargs,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout_head) if dropout_head > 0 else nn.Identity(),
            nn.Linear(d_output, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)          # (B, d_output)
        logits = self.head(emb)        # (B, num_classes)
        return logits
