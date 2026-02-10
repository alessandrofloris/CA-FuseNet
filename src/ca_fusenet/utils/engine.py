from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any
from pathlib import Path
from contextlib import nullcontext
from sklearn.metrics import f1_score

from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader

from ca_fusenet.data.collate import cafusenet_collate_fn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    acc: float

    def prefixed(self, prefix: str) -> str:
        return f"{prefix}loss={self.loss:.4f} {prefix}acc={self.acc:.4f}"


def ensure_output_dirs(cfg) -> dict[str, Path]:
    out = Path(cfg.paths.output_dir)

    ckpt_dir = Path(cfg.paths.ckpt_dir)
    eval_dir = Path(cfg.paths.eval_dir)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    return {
        "out": out,
        "ckpt": ckpt_dir,
        "eval": eval_dir,
    }

def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def select_cfg(cfg: DictConfig, key: str, default: Any) -> Any:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return default
    return value

def get_training_cfg(cfg: DictConfig) -> DictConfig:
    if "training" in cfg.training:
        return cfg.training.training
    return cfg.training

def get_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg.model:
        return cfg.model.model
    return cfg.model


def get_input_key(cfg: DictConfig) -> str | None:
    return select_cfg(cfg, "training.input_key", None)


def get_data_cfg(cfg: DictConfig) -> DictConfig:
    if "dataset" in cfg.data:
        return cfg.data.dataset
    return cfg.data


def get_split_data_cfg(data_cfg: DictConfig | None, split: str) -> DictConfig | None:
    if data_cfg is None:
        return None
    if split in data_cfg:
        return data_cfg[split]
    return None


def build_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool | None = None,
) -> DataLoader:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=cafusenet_collate_fn,
    )


def validate_batch(batch: dict[str, Any], input_key: str) -> None:
    required = {input_key, "label"}
    missing = required - set(batch.keys())
    if missing:
        raise KeyError(f"Batch missing keys: {sorted(missing)}")
    x = batch[input_key]
    labels = batch["label"]
    if x.numel() == 0:
        raise ValueError("Batch input tensor is empty")
    if labels.numel() == 0:
        raise ValueError("Batch label tensor is empty")
    if labels.ndim != 1:
        raise ValueError(f"Batch label must be 1D; got shape={tuple(labels.shape)}")


def _sanity_check_fusion(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> None:
    try:
        batch = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("Training dataloader is empty; cannot run sanity check.") from exc

    required_keys = {"tublet", "pose", "indicators", "label"}
    missing = required_keys - set(batch.keys())
    if missing:
        raise KeyError(f"Sanity check batch missing keys: {sorted(missing)}")

    tublet = batch["tublet"].to(device, non_blocking=True)
    pose = batch["pose"].to(device, non_blocking=True)
    indicators = batch["indicators"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    if labels.dtype != torch.long:
        labels = labels.long()

    model.eval()
    with torch.no_grad():
        output = model(
            video_input=tublet,
            pose_input=pose,
            occlusion_indicators=indicators,
        )

    if "logits" not in output:
        raise KeyError("Sanity check failed: model output missing key: logits")
    if "alpha" not in output:
        raise KeyError("Sanity check failed: model output missing key: alpha")

    logits = output["logits"]
    alpha = output["alpha"]
    batch_size = labels.shape[0]

    if logits.shape != (batch_size, num_classes):
        raise ValueError(
            "Sanity check failed: logits shape mismatch. "
            f"Expected {(batch_size, num_classes)}, got {tuple(logits.shape)}."
        )
    if alpha.shape != (batch_size, 1):
        raise ValueError(
            "Sanity check failed: alpha shape mismatch. "
            f"Expected {(batch_size, 1)}, got {tuple(alpha.shape)}."
        )

    logger.info("Sanity check passed")


def run_epoch(
    model: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
    input_key: str,
) -> EpochMetrics:
    if loader is None:
        logger.debug("run_epoch skipped because loader is None")
        return EpochMetrics(float("nan"), float("nan"))

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = nullcontext() if train else torch.no_grad()

    with context:
        for batch in loader:
            validate_batch(batch, input_key)
            x = batch[input_key].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if labels.dtype != torch.long:
                labels = labels.long()
            if x.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: x={x.shape[0]} labels={labels.shape[0]}"
                )

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, labels)
            if train and optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples else float("nan")
    avg_acc = total_correct / total_samples if total_samples else float("nan")
    return EpochMetrics(avg_loss, avg_acc)

def run_epoch_fusion(
    model: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    train: bool = True,
) -> dict[str, float]:
    if loader is None:
        logger.debug("run_epoch_fusion skipped because loader is None")
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "mean_alpha": float("nan"),
            "std_alpha": float("nan"),
        }

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_alphas: list[float] = []

    context = nullcontext() if train else torch.no_grad()

    with context:
        for batch in loader:
            required_keys = {"tublet", "pose", "indicators", "label"}
            missing = required_keys - set(batch.keys())
            if missing:
                raise KeyError(f"Batch missing keys: {sorted(missing)}")

            tublet = batch["tublet"].to(device, non_blocking=True)
            pose = batch["pose"].to(device, non_blocking=True)
            indicators = batch["indicators"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if labels.dtype != torch.long:
                labels = labels.long()

            if train:
                if optimizer is None:
                    raise ValueError("optimizer is required when train=True")
                optimizer.zero_grad(set_to_none=True)

            output = model(
                video_input=tublet,
                pose_input=pose,
                occlusion_indicators=indicators,
            )
            if "logits" not in output:
                raise KeyError("Model output missing key: logits")
            if "alpha" not in output:
                raise KeyError("Model output missing key: alpha")

            logits = output["logits"]
            alpha = output["alpha"]
            if logits.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: logits={logits.shape[0]} labels={labels.shape[0]}"
                )

            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_alphas.extend(alpha.detach().reshape(-1).cpu().tolist())

    if total_samples == 0:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "mean_alpha": float("nan"),
            "std_alpha": float("nan"),
        }

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    alpha_tensor = torch.tensor(all_alphas, dtype=torch.float32)
    mean_alpha = float(alpha_tensor.mean().item())
    std_alpha = float(alpha_tensor.std(unbiased=False).item())

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "macro_f1": macro_f1,
        "mean_alpha": mean_alpha,
        "std_alpha": std_alpha,
    }


def _resolve_encoder_cfgs(cfg: DictConfig, training_cfg: DictConfig) -> tuple[DictConfig, DictConfig]:
    video_encoder_cfg = select_cfg(cfg, "video_encoder", None)
    pose_encoder_cfg = select_cfg(cfg, "pose_encoder", None)

    if video_encoder_cfg is None:
        raise ValueError("Missing video encoder config. Expected training.video_encoder or encoder.video.")
    if pose_encoder_cfg is None:
        raise ValueError("Missing pose encoder config. Expected training.pose_encoder or encoder.pose.")

    return video_encoder_cfg, pose_encoder_cfg


def _read_d_output(encoder_cfg: DictConfig, encoder_name: str) -> int:
    d_output = select_cfg(encoder_cfg, "d_output", None)
    if d_output is None:
        raise ValueError(
            f"Encoder config for {encoder_name} must define d_output to specify embedding dimensionality."
        )
    return int(d_output)