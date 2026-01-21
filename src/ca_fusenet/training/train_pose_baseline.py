from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ca_fusenet.data.collate import cafusenet_collate_fn
from ca_fusenet.models.pose_baseline import PoseBaseline


def _select_cfg(cfg: DictConfig, key: str, default: Any) -> Any:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return default
    return value


def _get_data_cfg(cfg: DictConfig) -> DictConfig:
    if "dataset" in cfg.data:
        return cfg.data.dataset
    return cfg.data


def _validate_batch(batch: dict[str, Any]) -> None:
    required = {"pose", "label"}
    missing = required - set(batch.keys())
    if missing:
        raise KeyError(f"Batch missing keys: {sorted(missing)}")
    pose = batch["pose"]
    labels = batch["label"]
    if pose.numel() == 0:
        raise ValueError("Batch pose tensor is empty")
    if labels.numel() == 0:
        raise ValueError("Batch label tensor is empty")
    if labels.ndim != 1:
        raise ValueError(f"Batch label must be 1D; got shape={tuple(labels.shape)}")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
) -> tuple[float, float]:
    if loader is None:
        return float("nan"), float("nan")

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Use torch.no_grad() context only during evaluation
    from contextlib import nullcontext
    context = nullcontext() if train else torch.no_grad()

    with context:
        for batch in loader:
            _validate_batch(batch)
            pose = batch["pose"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if labels.dtype != torch.long:
                labels = labels.long()
            if pose.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: pose={pose.shape[0]} labels={labels.shape[0]}"
                )

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(pose)
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
    return avg_loss, avg_acc


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed = int(_select_cfg(cfg, "seed", 42))
    torch.manual_seed(seed)

    data_cfg = _get_data_cfg(cfg)
    dataset = hydra.utils.instantiate(data_cfg)
    if len(dataset) <= 0:
        raise ValueError("Dataset is empty; check data configuration and artifacts.")

    batch_size = int(_select_cfg(cfg, "training.batch_size", 32))
    num_workers = int(_select_cfg(cfg, "training.num_workers", 0))
    val_split = float(_select_cfg(cfg, "training.val_split", 0.1))

    if val_split < 0.0 or val_split >= 1.0:
        raise ValueError(f"training.val_split must be in [0, 1); got {val_split}")

    n_samples = len(dataset)
    if val_split > 0.0 and n_samples < 2:
        raise ValueError("Dataset too small for train/val split (need N>=2).")

    generator = torch.Generator().manual_seed(seed)
    if val_split == 0.0:
        train_ds = dataset
        val_ds = None
    else:
        val_size = max(1, int(n_samples * val_split))
        if val_size >= n_samples:
            val_size = n_samples - 1
        train_size = n_samples - val_size
        indices = torch.randperm(n_samples, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=cafusenet_collate_fn,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=cafusenet_collate_fn,
        )

    num_classes = _select_cfg(cfg, "data.num_classes", None)
    if num_classes is None:
        num_classes = _select_cfg(cfg, "data.dataset.num_classes", None)
    if num_classes is None or int(num_classes) <= 0:
        raise ValueError(
            "num_classes must be set in cfg.data.num_classes (and > 0) for training."
        )

    hidden_dim = int(_select_cfg(cfg, "model.hidden_dim", 256))
    dropout = float(_select_cfg(cfg, "model.dropout", 0.2))
    pooling = _select_cfg(cfg, "model.pooling", "mean")
    in_channels = int(_select_cfg(cfg, "model.in_channels", 3))

    model = PoseBaseline(
        num_classes=int(num_classes),
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pooling=pooling,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    lr = float(_select_cfg(cfg, "training.lr", 1e-3))
    weight_decay = float(_select_cfg(cfg, "training.weight_decay", 1e-2))
    epochs = int(_select_cfg(cfg, "training.epochs", 5))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = float("-inf")
    best_path = Path(_select_cfg(cfg, "training.artifacts.best_pose_path", "artifacts/best_pose_baseline.pt"))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, device, criterion, optimizer, train=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, device, criterion, None, train=False
        )

        print(
            f"epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loader is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                best_path,
            )


if __name__ == "__main__":
    main()
