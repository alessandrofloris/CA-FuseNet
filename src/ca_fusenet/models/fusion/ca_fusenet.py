from typing import Any

import torch
from torch import Tensor, nn

try:
    from .gating import CrowdAwareGating
except ImportError:
    from gating import CrowdAwareGating


class CaFuseNet(nn.Module):
    def __init__(
        self,
        video_encoder: nn.Module,
        pose_encoder: nn.Module,
        d_video: int = 512,
        d_pose: int = 256,
        d_common: int = 128,
        gate_hidden: int = 64,
        d_fused: int = 256,
        n_indicators: int = 3,
        dropout: float = 0.3,
        num_classes: int = 37,
    ) -> None:
        super().__init__()
        self.video_encoder = video_encoder
        self.pose_encoder = pose_encoder

        self.gating = CrowdAwareGating(
            d_video=d_video,
            d_pose=d_pose,
            d_common=d_common,
            gate_hidden=gate_hidden,
            d_fused=d_fused,
            n_indicators=n_indicators,
            dropout=dropout,
        )
        self.classification_head = nn.Linear(d_fused, num_classes)

    def forward(
        self,
        video_input: Any,
        pose_input: Any,
        occlusion_indicators: Tensor,
    ) -> dict[str, Tensor]:
        f_video = self.video_encoder(video_input)
        f_pose = self.pose_encoder(pose_input)
        f_fused, alpha = self.gating(f_video, f_pose, occlusion_indicators)
        logits = self.classification_head(f_fused)
        return {"logits": logits, "alpha": alpha, "f_fused": f_fused}

    def get_param_groups(self, lr_encoder: float, lr_fusion: float) -> list[dict]:
        encoder_params = []
        encoder_param_ids = set()
        for param in list(self.video_encoder.parameters()) + list(self.pose_encoder.parameters()):
            param_id = id(param)
            if param_id not in encoder_param_ids:
                encoder_params.append(param)
                encoder_param_ids.add(param_id)

        fusion_params = [param for param in self.parameters() if id(param) not in encoder_param_ids]
        return [
            {"params": encoder_params, "lr": lr_encoder},
            {"params": fusion_params, "lr": lr_fusion},
        ]


if __name__ == "__main__":
    B = 8
    video_encoder = nn.Linear(1024, 512)
    pose_encoder = nn.Linear(300, 256)

    model = CaFuseNet(video_encoder=video_encoder, pose_encoder=pose_encoder)
    video_input = torch.randn(B, 1024)
    pose_input = torch.randn(B, 300)
    occlusion_indicators = torch.randn(B, 3)

    outputs = model(video_input, pose_input, occlusion_indicators)
    logits = outputs["logits"]
    alpha = outputs["alpha"]
    f_fused = outputs["f_fused"]

    assert logits.shape == (B, 37)
    assert alpha.shape == (B, 1)
    assert f_fused.shape == (B, 256)

    logits.sum().backward()
    for param in model.parameters():
        assert param.grad is not None

    lr_encoder = 1e-4
    lr_fusion = 1e-3
    param_groups = model.get_param_groups(lr_encoder=lr_encoder, lr_fusion=lr_fusion)
    assert len(param_groups) == 2
    assert param_groups[0]["lr"] == lr_encoder
    assert param_groups[1]["lr"] == lr_fusion

    encoder_param_ids = {
        id(p) for p in list(model.video_encoder.parameters()) + list(model.pose_encoder.parameters())
    }
    fusion_param_ids = {id(p) for p in model.parameters() if id(p) not in encoder_param_ids}

    group0_ids = {id(p) for p in param_groups[0]["params"]}
    group1_ids = {id(p) for p in param_groups[1]["params"]}

    assert group0_ids == encoder_param_ids
    assert group1_ids == fusion_param_ids

    print("All tests passed")
