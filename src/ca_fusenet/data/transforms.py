"""Video tubelet transforms for CA-FuseNet.

This module applies sample-level augmentations to tubelets shaped ``(C, T, H, W)``.
All spatial transforms share one set of random parameters across all frames to
preserve temporal coherence.

Integration note:
The dataset should instantiate ``VideoTransform(train=True, ...)`` for training
and ``VideoTransform(train=False, ...)`` for validation/test, then call it in
``__getitem__`` after loading the tubelet.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Subset
from torchvision.transforms import ColorJitter


class VideoTransform:
    def __init__(
        self,
        train: bool = True,
        crop_size: int = 112,
        crop_padding: int = 8,
        flip_p: float = 0.5,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.2,
        hue: float = 0.1,
        erase_p: float = 0.3,
        erase_scale: tuple = (0.02, 0.15),
        erase_ratio: tuple = (0.3, 3.3),
        temporal_subsample_rate: float = 1.0,
        max_temporal_shift: int = 2,
        input_scale_255: bool = False,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        self.train = bool(train)
        self.crop_size = int(crop_size)
        self.crop_padding = int(crop_padding)
        self.flip_p = float(flip_p)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.erase_p = float(erase_p)
        self.erase_scale = tuple(erase_scale)
        self.erase_ratio = tuple(erase_ratio)
        self.temporal_subsample_rate = float(temporal_subsample_rate)
        self.max_temporal_shift = int(max_temporal_shift)
        self.input_scale_255 = bool(input_scale_255)

        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1, 1)

        self._color_jitter = ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )

    def __call__(self, tubelet: torch.Tensor) -> torch.Tensor:
        if not isinstance(tubelet, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tubelet)}")
        if tubelet.ndim != 4:
            raise ValueError(f"Expected shape (C, T, H, W), got {tuple(tubelet.shape)}")
        if tubelet.shape[0] != 3:
            raise ValueError(f"Expected C=3, got C={tubelet.shape[0]}")

        x = tubelet.to(dtype=torch.float32)

        if self.input_scale_255:
            x = x / 255.0

        if self.train:
            x = self._random_horizontal_flip(x)
            x = self._random_spatial_crop(x)
            x = self._color_jitter_same_params(x)
            x = self._random_erasing_same_rect(x)
            x = self._random_temporal_subsample(x)
            x = self._random_temporal_shift(x)
        else:
            x = self._center_crop_if_needed(x)

        x = self._normalize(x)
        return x

    def _random_horizontal_flip(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.flip_p:
            return torch.flip(x, dims=[3])
        return x

    def _random_spatial_crop(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if self.crop_size >= h and self.crop_size >= w:
            return x

        padded = F.pad(x, padding=[self.crop_padding] * 4, padding_mode="reflect")
        _, _, hp, wp = padded.shape
        if self.crop_size > hp or self.crop_size > wp:
            raise ValueError(
                f"crop_size={self.crop_size} does not fit padded size ({hp}, {wp})"
            )
        max_top = hp - self.crop_size
        max_left = wp - self.crop_size
        top = int(torch.randint(0, max_top + 1, (1,)).item())
        left = int(torch.randint(0, max_left + 1, (1,)).item())
        return padded[:, :, top : top + self.crop_size, left : left + self.crop_size]

    def _center_crop_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if self.crop_size >= h and self.crop_size >= w:
            return x
        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2
        return x[:, :, top : top + self.crop_size, left : left + self.crop_size]

    def _color_jitter_same_params(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.brightness <= 0.0
            and self.contrast <= 0.0
            and self.saturation <= 0.0
            and self.hue <= 0.0
        ):
            return x

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = (
            self._color_jitter.get_params(
                self._color_jitter.brightness,
                self._color_jitter.contrast,
                self._color_jitter.saturation,
                self._color_jitter.hue,
            )
        )

        frames = x.permute(1, 0, 2, 3).contiguous()
        out_frames: list[torch.Tensor] = []
        for frame in frames:
            out = frame
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    out = F.adjust_brightness(out, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    out = F.adjust_contrast(out, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    out = F.adjust_saturation(out, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    out = F.adjust_hue(out, hue_factor)
            out_frames.append(out)
        return torch.stack(out_frames, dim=0).permute(1, 0, 2, 3).contiguous()

    def _random_erasing_same_rect(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.erase_p:
            return x

        c, _, h, w = x.shape
        area = h * w
        log_ratio = torch.log(torch.tensor(self.erase_ratio, dtype=torch.float32))
        min_log_ratio = float(log_ratio[0].item())
        max_log_ratio = float(log_ratio[1].item())

        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(*self.erase_scale).item()
            aspect = torch.exp(torch.empty(1).uniform_(min_log_ratio, max_log_ratio)).item()
            erase_h = int(round((erase_area * aspect) ** 0.5))
            erase_w = int(round((erase_area / aspect) ** 0.5))
            if 0 < erase_h < h and 0 < erase_w < w:
                top = int(torch.randint(0, h - erase_h + 1, (1,)).item())
                left = int(torch.randint(0, w - erase_w + 1, (1,)).item())
                x[:, :, top : top + erase_h, left : left + erase_w] = torch.zeros(
                    (c, 1, erase_h, erase_w), dtype=x.dtype, device=x.device
                )
                return x
        return x

    def _random_temporal_subsample(self, x: torch.Tensor) -> torch.Tensor:
        rate = self.temporal_subsample_rate
        if rate >= 1.0:
            return x
        if rate <= 0.0:
            raise ValueError(f"temporal_subsample_rate must be > 0, got {rate}")

        _, t, _, _ = x.shape
        keep_t = max(1, int(round(t * rate)))
        if keep_t >= t:
            return x

        keep_idx = torch.randperm(t, device=x.device)[:keep_t]
        keep_idx = torch.sort(keep_idx).values
        reduced = x[:, keep_idx, :, :]

        restore_idx = torch.linspace(
            0,
            keep_t - 1,
            steps=t,
            device=x.device,
            dtype=torch.float32,
        ).round().to(dtype=torch.long)
        return reduced[:, restore_idx, :, :]

    def _random_temporal_shift(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_temporal_shift <= 0:
            return x
        shift = int(
            torch.randint(
                -self.max_temporal_shift,
                self.max_temporal_shift + 1,
                (1,),
            ).item()
        )
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std


class TransformSubset(Subset):
    """Subset wrapper that applies a transform to one sample-dict key."""

    def __init__(self, dataset, indices, transform=None, transform_key: str = "tublet"):
        super().__init__(dataset, indices)
        self.transform = transform
        self.transform_key = transform_key

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if self.transform is None or not isinstance(sample, dict):
            return sample
        if self.transform_key not in sample:
            return sample

        value = sample[self.transform_key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32)
        else:
            value = value.float()

        out = dict(sample)
        out[self.transform_key] = self.transform(value)
        return out


def _assert_all_frames_identical(tubelet: torch.Tensor, atol: float = 1e-5) -> None:
    first = tubelet[:, 0, :, :]
    diffs = (tubelet - first.unsqueeze(1)).abs().max().item()
    if diffs > atol:
        raise AssertionError(f"Frames are not identical across time; max diff={diffs}")


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.rand(3, 16, 112, 112) * 255.0

    train_tf = VideoTransform(train=True)
    train_out = train_tf(x.clone())
    assert train_out.shape == (3, 16, 112, 112), f"Unexpected shape: {train_out.shape}"
    print(f"Min original: {x.min()}, Max: {x.max()}, Mean: {x.mean():.3f}")
    print(f"Min transformed: {train_out.min()}, Max: {train_out.max()}, Mean: {train_out.mean():.3f}")


    eval_tf = VideoTransform(train=False)
    eval_out = eval_tf(x.clone())
    assert eval_out.shape == (3, 16, 112, 112), f"Unexpected shape: {eval_out.shape}"

    no_crop_tf = VideoTransform(train=True, crop_size=112)
    no_crop_out = no_crop_tf(x.clone())
    assert no_crop_out.shape == (3, 16, 112, 112), f"Unexpected shape: {no_crop_out.shape}"

    identical_frame = torch.rand(3, 1, 112, 112) * 255.0
    identical_tubelet = identical_frame.repeat(1, 16, 1, 1)
    coherent_out = VideoTransform(train=True)(identical_tubelet)
    _assert_all_frames_identical(coherent_out)

    print("All tests passed")
