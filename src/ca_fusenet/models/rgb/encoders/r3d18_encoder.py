# src/ca_fusenet/models/rgb/encoders/r3d18_encoder.py
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights

log = logging.getLogger(__name__)


class R3D18Encoder(nn.Module):
    """
    3D ResNet-18 encoder.
    Returns a global feature vector per clip: [B, d_v].
    """

    def __init__(
        self,
        *,
        pretrained: bool = True,
        weights: Optional[str] = "KINETICS400_V1",
        freeze_depth: int = 0,
        freeze_proj: bool = False,
        d_output: Optional[int] = None,  # if you want a projection head
        dropout_proj: float = 0.0,
    ) -> None:
        super().__init__()

        if pretrained:
            if weights == "KINETICS400_V1":
                w = R3D_18_Weights.KINETICS400_V1
            else:
                raise ValueError(f"Unknown weights='{weights}'")
        else:
            w = None

        if not 0 <= freeze_depth <= 5:
            raise ValueError(f"freeze_depth must be in [0, 5], got {freeze_depth}")

        backbone = r3d_18(weights=w)

        # Remove the final classification head
        self.backbone = nn.Sequential(OrderedDict(list(backbone.named_children())[:-1]))
        self.feature_dim = 512
        self._freeze_stages = ["stem", "layer1", "layer2", "layer3", "layer4"]
        self._frozen_stage_names: list[str] = []

        if freeze_depth > 0:
            for stage_name in self._freeze_stages[:freeze_depth]:
                stage = self.backbone.get_submodule(stage_name)
                for p in stage.parameters():
                    p.requires_grad = False
                self._frozen_stage_names.append(stage_name)
            log.info("Froze video encoder stages: %s", ", ".join(self._frozen_stage_names))
            for stage_name in self._frozen_stage_names:
                self.backbone.get_submodule(stage_name).eval()

        self.proj = None
        if d_output is not None and d_output != self.feature_dim:
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_proj) if dropout_proj > 0 else nn.Identity(),
                nn.Linear(self.feature_dim, d_output),
            )
            self.d_output = d_output
        else:
            self.d_output = self.feature_dim

        if freeze_proj and self.proj is not None:
            for p in self.proj.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> R3D18Encoder:
        super().train(mode)
        # Keep frozen stages in eval mode so batch norm stats don't update.
        for stage_name in self._frozen_stage_names:
            self.backbone.get_submodule(stage_name).eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,T,H,W] float
        returns: [B, d_v]
        """
        feat = self.backbone(x)          # [B,512,1,1,1]
        feat = feat.flatten(1)           # [B,512]
        if self.proj is not None:
            feat = self.proj(feat)       # [B,d_output]
        return feat
