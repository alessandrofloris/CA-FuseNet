from __future__ import annotations

from pathlib import Path

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import Subset

from ca_fusenet.utils.engine import (
    build_dataloader,
    get_training_cfg,
    get_data_cfg,
    get_model_cfg,
    get_split_data_cfg,
    run_epoch,
    select_cfg,
    ensure_output_dirs,   
)

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:

    # Set random seed for reproducibility
    seed = int(select_cfg(cfg, "seed", 42))
    torch.manual_seed(seed)

    # Training configuration
    training_cfg = get_training_cfg(cfg)

    input_key = str(select_cfg(training_cfg, "input_key", "unknown"))
    batch_size = int(select_cfg(training_cfg, "batch_size", 32))
    num_workers = int(select_cfg(training_cfg, "num_workers", 0))
    val_split = float(select_cfg(training_cfg, "val_split", 0.1))
    lr = float(select_cfg(training_cfg, "lr", 1e-3))
    weight_decay = float(select_cfg(training_cfg, "weight_decay", 1e-2))
    epochs = int(select_cfg(training_cfg, "epochs", 5))

    # Output directories (checkpoints, metrics, eval)
    out_dirs = ensure_output_dirs(cfg)
    best_path = out_dirs["ckpt"] / "best.pt"

    # Data configuration
    data_cfg = get_data_cfg(cfg)
    train_data_cfg = get_split_data_cfg(data_cfg, "train")
    val_data_cfg = get_split_data_cfg(data_cfg, "val")

    # Model configuration
    model_cfg = get_model_cfg(cfg)

    # Logging configurations
    logger.info("config.training=%s", OmegaConf.to_container(training_cfg, resolve=True))
    logger.info("config.model=%s", OmegaConf.to_container(model_cfg, resolve=True))
    logger.info("config.data=%s", OmegaConf.to_container(data_cfg, resolve=True))

    # Dataset instantiation
    if train_data_cfg is not None:
        train_dataset = hydra.utils.instantiate(train_data_cfg)
    else:
        train_dataset = hydra.utils.instantiate(data_cfg)

    if len(train_dataset) <= 0:
        raise ValueError("Training dataset is empty; check data configuration and artifacts.")

    # Split the train dataset
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
    
    # Train data loaders
    pin_memory = torch.cuda.is_available()
    train_loader = build_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Validation data loader
    val_loader = None
    if val_ds is not None:
        val_loader = build_dataloader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # Model instantiation and setup
    model = hydra.utils.instantiate(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("device=%s", device)

    # Optimizer and loss function setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = float("-inf")
    for epoch in range(1, epochs + 1):
        
        train_metrics = run_epoch(
            model, train_loader, device, criterion, optimizer, train=True, input_key=input_key
        )
        
        val_metrics = run_epoch(
            model, val_loader, device, criterion, None, train=False, input_key=input_key
        )

        logger.info(
            "epoch=%d/%d %s %s",
            epoch,
            epochs,
            train_metrics.prefixed("train_"),
            val_metrics.prefixed("val_"),
        )

        # Checkpointing best model
        if val_loader is not None and val_metrics.acc > best_val_acc:
            best_val_acc = val_metrics.acc
            
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch,
                },
                best_path,
            )
            
            logger.info(
                "checkpoint=best path=%s val_acc=%.4f",
                best_path,
                best_val_acc,
            )


if __name__ == "__main__":
    main()
