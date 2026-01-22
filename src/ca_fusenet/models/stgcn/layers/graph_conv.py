import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        t_kernel_size: int = 1,
        t_stride: int = 1,
        t_padding: int = 0,
        t_dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        # x: (B, Cin, T, V), A: (K, V, V)
        K = A.size(0)
        if K != self.kernel_size:
            raise ValueError(f"A has K={K}, expected K={self.kernel_size}")

        x = self.conv(x)  # (B, Cout*K, T', V)
        B, KC, T, V = x.shape
        x = x.view(B, self.kernel_size, KC // self.kernel_size, T, V)  # (B,K,Cout,T,V)
        x = torch.einsum("bkctv,kvw->bctw", x, A)  # (B,Cout,T,W=V)

        return x.contiguous(), A