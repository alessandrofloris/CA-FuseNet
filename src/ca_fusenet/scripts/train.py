from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import Subset

from ca_fusenet.utils.engine import (
    build_dataloader,
    get_data_cfg,
    get_input_key,
    get_model_cfg,
    get_split_data_cfg,
    run_epoch,
    select_cfg,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed = int(select_cfg(cfg, "seed", 42))
    torch.manual_seed(seed)

    input_key = get_input_key(cfg)

    batch_size = int(select_cfg(cfg, "training.batch_size", 32))
    num_workers = int(select_cfg(cfg, "training.num_workers", 0))
    val_split = float(select_cfg(cfg, "training.val_split", 0.1))

    data_cfg = get_data_cfg(cfg)
    train_data_cfg = get_split_data_cfg(data_cfg, "train")
    val_data_cfg = get_split_data_cfg(data_cfg, "val")

    if train_data_cfg is not None:
        train_dataset = hydra.utils.instantiate(train_data_cfg)
    else:
        # fallback
        data_cfg = get_data_cfg(cfg)
        train_dataset = hydra.utils.instantiate(data_cfg)

    if len(train_dataset) <= 0:
        raise ValueError("Training dataset is empty; check data configuration and artifacts.")

    # Instantiate validation dataset if specified
    val_dataset = None
    if val_data_cfg is not None:
        val_dataset = hydra.utils.instantiate(val_data_cfg)
        if len(val_dataset) <= 0:
            raise ValueError("Val dataset is empty; check data configuration and artifacts.")

    # If no validation dataset provided, split train dataset
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
    train_loader = build_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = build_dataloader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    model_cfg = get_model_cfg(cfg)
    model = hydra.utils.instantiate(model_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    lr = float(select_cfg(cfg, "training.lr", 1e-3))
    weight_decay = float(select_cfg(cfg, "training.weight_decay", 1e-2))
    epochs = int(select_cfg(cfg, "training.epochs", 5))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = float("-inf")
    best_path = Path(select_cfg(cfg, "training.artifacts.best_path", "artifacts/best_baseline.pt"))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, device, criterion, optimizer, train=True, input_key=input_key
        )
        val_loss, val_acc = run_epoch(
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
