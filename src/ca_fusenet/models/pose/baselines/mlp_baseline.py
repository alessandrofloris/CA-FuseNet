from __future__ import annotations

from typing import Literal
import logging

from ca_fusenet.models import pose
import torch
from torch import nn

log = logging.getLogger(__name__)

class PoseMLPBaseline(nn.Module):
    """Minimal pose-only baseline with global pooling and a small MLP."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pooling: Literal["mean", "max"] = "mean",
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0; got {num_classes}")
        if pooling not in ("mean", "max"):
            raise ValueError(f"pooling must be 'mean' or 'max'; got {pooling}")
        self.in_channels = in_channels
        self.pooling = pooling

        # Neural architecture definition
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if not isinstance(pose, torch.Tensor):
            raise TypeError(f"pose must be torch.Tensor; got {type(pose)}")
        if pose.ndim != 4:
            raise ValueError(
                f"pose expected shape (B, C, T, V); got shape={tuple(pose.shape)}"
            )
        if pose.shape[1] != self.in_channels:
            raise ValueError(
                f"pose second dim (C) expected {self.in_channels}; got {pose.shape[1]}"
            )
        
        if self.pooling == "mean":
            x = pose.mean(dim=(2, 3)) 
        else:
            x = pose.flatten(2).max(dim=2).values 
        
        return self.mlp(x)
        