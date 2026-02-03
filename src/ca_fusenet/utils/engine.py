from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any
from pathlib import Path

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

    from contextlib import nullcontext

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
