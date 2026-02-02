from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ca_fusenet.data.collate import cafusenet_collate_fn


def _select_cfg(cfg: DictConfig, key: str, default: Any) -> Any:
    value = OmegaConf.select(cfg, key, default=None)
    if value is None:
        return default
    return value

def _get_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg.model:
        return cfg.model.model
    return cfg.model

def _get_input_key(cfg: DictConfig) -> str:
    return _select_cfg(cfg, "training.input_key", None)

def _get_data_cfg(cfg: DictConfig) -> DictConfig:
    if "dataset" in cfg.data:
        return cfg.data.dataset
    return cfg.data

def _get_train_data_cfg(data_cfg: DictConfig) -> DictConfig:
    if "train" in data_cfg:
        return data_cfg.train
    return None

def _get_val_data_cfg(data_cfg: DictConfig) -> DictConfig:
    if "val" in data_cfg:
        return data_cfg.val
    return None


def _validate_batch(batch: dict[str, Any], input_key: str) -> None:
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


def _run_epoch(
    model: nn.Module,
    loader: DataLoader | None,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
    input_key: str,
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
            _validate_batch(batch, input_key)
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
    return avg_loss, avg_acc


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed = int(_select_cfg(cfg, "seed", 42))
    torch.manual_seed(seed)

    input_key = _get_input_key(cfg)

    batch_size = int(_select_cfg(cfg, "training.batch_size", 32))
    num_workers = int(_select_cfg(cfg, "training.num_workers", 0))
    val_split = float(_select_cfg(cfg, "training.val_split", 0.1))

    data_cfg = _get_data_cfg(cfg)
    train_data_cfg = _get_train_data_cfg(data_cfg)
    val_data_cfg = _get_val_data_cfg(data_cfg)

    if train_data_cfg is not None:
        train_dataset = hydra.utils.instantiate(train_data_cfg)
    else:
        # fallback
        data_cfg = _get_data_cfg(cfg)
        train_dataset = hydra.utils.instantiate(data_cfg)

    if len(train_dataset) <= 0:
        raise ValueError("Training dataset is empty; check data configuration and artifacts.")

    val_dataset = None
    if val_data_cfg is not None:
        val_dataset = hydra.utils.instantiate(val_data_cfg)
        if len(val_dataset) <= 0:
            raise ValueError("Val dataset is empty; check data configuration and artifacts.")


    if val_dataset is None:

        if val_split < 0.0 or val_split >= 1.0:
            raise ValueError(f"training.val_split must be in [0, 1); got {val_split}")

        n_samples = len(train_dataset)
        if n_samples < 2:
            raise ValueError("Dataset too small for train/val split (need N>=2).")
    
        if val_split == 0.0:
            train_ds = train_dataset
            val_ds = None
        else:
            generator = torch.Generator().manual_seed(seed)
            val_size = max(1, int(n_samples * val_split))
            if val_size >= n_samples:
                val_size = n_samples - 1
            train_size = n_samples - val_size
            indices = torch.randperm(n_samples, generator=generator).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            train_ds = Subset(train_dataset, train_indices)
            val_ds = Subset(train_dataset, val_indices)
    else:
        train_ds = train_dataset
        val_ds = val_dataset

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

    model_cfg = _get_model_cfg(cfg)
    model = hydra.utils.instantiate(model_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    lr = float(_select_cfg(cfg, "training.lr", 1e-3))
    weight_decay = float(_select_cfg(cfg, "training.weight_decay", 1e-2))
    epochs = int(_select_cfg(cfg, "training.epochs", 5))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = float("-inf")
    best_path = Path(_select_cfg(cfg, "training.artifacts.best_path", "artifacts/best_baseline.pt"))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, device, criterion, optimizer, train=True, input_key=input_key
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, device, criterion, None, train=False, input_key=input_key    
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
