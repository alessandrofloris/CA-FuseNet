import torch
from torch import Tensor, nn


class CrowdAwareGating(nn.Module):
    def __init__(
        self,
        d_video: int = 512,
        d_pose: int = 256,
        d_common: int = 128,
        gate_hidden: int = 64,
        d_fused: int = 256,
        n_indicators: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.video_proj = nn.Linear(d_video, d_common)
        self.pose_proj = nn.Linear(d_pose, d_common)

        gate_in_dim = 2 * d_common + n_indicators
        self.gate_norm = nn.LayerNorm(gate_in_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_mlp[2].bias, 0.0)

        self.fusion_proj = nn.Sequential(
            nn.Linear(d_pose + d_video, d_fused),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, f_video: Tensor, f_pose: Tensor, o: Tensor) -> tuple[Tensor, Tensor]:
        v_gate = self.video_proj(f_video)
        p_gate = self.pose_proj(f_pose)

        z = torch.cat((v_gate, p_gate, o), dim=1)
        z = self.gate_norm(z)
        alpha = self.gate_mlp(z)

        fused_weighted = torch.cat((alpha * f_pose, (1.0 - alpha) * f_video), dim=1)
        f_fused = self.fusion_proj(fused_weighted)
        return f_fused, alpha


if __name__ == "__main__":
    B = 8
    f_video = torch.randn(B, 512)
    f_pose = torch.randn(B, 256)
    o = torch.randn(B, 3)

    model = CrowdAwareGating()
    f_fused, alpha = model(f_video, f_pose, o)

    assert f_fused.shape == (B, 256)
    assert alpha.shape == (B, 1)
    assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0)

    f_fused.sum().backward()
    for p in model.parameters():
        assert p.grad is not None

    print("All tests passed")
