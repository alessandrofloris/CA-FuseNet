from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ca_fusenet.models.stgcnpp.stgcn import STGCN

log = logging.getLogger(__name__)


class STGCNPPEncoder(nn.Module):
    """
    STGCN++ pose encoder wrapper.
    Input: (B, C, T, V)
    Output: (B, d_output)
    """

    def __init__(
        self,
        in_channels: int = 3,
        d_output: int = 256,
        graph_cfg: dict[str, Any] = dict(layout="coco", mode="spatial"),
        base_channels: int = 64,
        num_stages: int = 10,
        inflate_stages: list[int] = [5, 8],
        down_stages: list[int] = [5, 8],
        ch_ratio: int = 2,
        num_person: int = 1,
        pretrained: str | None = None,
        gcn_adaptive: str = "init",
        gcn_with_res: bool = True,
        tcn_type: str = "mstcn",
    ) -> None:
        super().__init__()
        graph_cfg = dict(graph_cfg)
        inflate_stages = list(inflate_stages)
        down_stages = list(down_stages)

        self.backbone = STGCN(
            graph_cfg=graph_cfg,
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            inflate_stages=inflate_stages,
            down_stages=down_stages,
            ch_ratio=ch_ratio,
            num_person=num_person,
            gcn_adaptive=gcn_adaptive,
            gcn_with_res=gcn_with_res,
            tcn_type=tcn_type,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_person = num_person
        self.d_output = d_output

        backbone_out_channels = self._infer_backbone_output_channels()
        if backbone_out_channels != d_output:
            raise ValueError(
                f"d_output ({d_output}) must match STGCN output channels ({backbone_out_channels})."
            )

        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _infer_backbone_output_channels(self) -> int:
        if len(self.backbone.gcn) == 0:
            return self.backbone.in_channels
        return int(self.backbone.gcn[-1].gcn.out_channels)

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint must be a dict or contain a dict state_dict.")

        for key in ("state_dict", "model_state_dict", "model", "net", "model_state"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict
        return checkpoint

    def _select_prefix_to_strip(self, state_dict: dict[str, torch.Tensor]) -> str:
        model_keys = set(self.backbone.state_dict().keys())    
        candidates = ["", "backbone.", "encoder.", "encoder.backbone."]
        best_prefix = ""
        best_matches = 0
        for prefix in candidates:
            matches = sum(1 for k in state_dict if k.startswith(prefix) 
                        and k[len(prefix):] in model_keys)
            if matches > best_matches:
                best_matches = matches
                best_prefix = prefix
        
        return best_prefix

    @staticmethod
    def _strip_prefix(
        state_dict: dict[str, torch.Tensor], prefix: str
    ) -> dict[str, torch.Tensor]:
        if not prefix:
            return dict(state_dict)
        return {
            (key[len(prefix):] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }

    def _load_pretrained(self, pretrained_path: str) -> None:
        ckpt_path = Path(pretrained_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        
        raw_state_dict = self._extract_state_dict(checkpoint)
        prefix = self._select_prefix_to_strip(raw_state_dict)
        state_dict = self._strip_prefix(raw_state_dict, prefix)

        model_state = self.backbone.state_dict()
        compatible_state: dict[str, torch.Tensor] = {}
        unexpected_count = 0
        shape_mismatch_count = 0

        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                unexpected_count += 1
                continue
            if key not in model_state:
                unexpected_count += 1
                continue
            if model_state[key].shape != value.shape:
                shape_mismatch_count += 1
                continue
            compatible_state[key] = value

        incompatible = self.backbone.load_state_dict(compatible_state, strict=False)
        loaded_count = len(compatible_state)
        missing_count = len(incompatible.missing_keys)
        unexpected_count += len(incompatible.unexpected_keys)

        stripped_prefix = prefix if prefix else "<none>"
        log.info(
            "Loaded STGCN++ checkpoint from %s (stripped prefix: %s): "
            "loaded=%d, missing=%d, unexpected=%d, shape_mismatch=%d",
            pretrained_path,
            stripped_prefix,
            loaded_count,
            missing_count,
            unexpected_count,
            shape_mismatch_count,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, T, V], got shape {tuple(x.shape)}.")

        x = x.permute(0, 2, 3, 1).contiguous()  # [B, T, V, C]
        x = x.unsqueeze(1)  # [B, 1, T, V, C]

        x = self.backbone(x)  # [N, M, C_out, T_out, V_out]

        n, m, c_out, t_out, v_out = x.shape
        x = x.reshape(n * m, c_out, t_out, v_out)
        x = self.pool(x)  # [N*M, C_out, 1, 1]
        x = x.reshape(n, m, c_out)
        x = x.mean(dim=1)  # [N, C_out]

        if x.shape[1] != self.d_output:
            raise RuntimeError(
                f"Encoder output dim mismatch: got {x.shape[1]}, expected {self.d_output}."
            )
        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    model = STGCNPPEncoder(pretrained=None)
    inputs = torch.randn(4, 3, 300, 17)

    outputs = model(inputs)
    assert outputs.shape == (4, 256), f"Unexpected output shape: {tuple(outputs.shape)}"

    loss = outputs.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients found after backward pass."

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print("All tests passed")
