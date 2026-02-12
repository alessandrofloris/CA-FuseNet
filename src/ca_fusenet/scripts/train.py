from __future__ import annotations

import json
import logging
import math

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
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
    validate_batch,
)

logger = logging.getLogger(__name__)


def _to_json_float(value: float) -> float | None:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _collect_val_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
    input_key: str,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for batch in loader:
            validate_batch(batch, input_key)
            x = batch[input_key].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if labels.dtype != torch.long:
                labels = labels.long()
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

    return all_labels, all_preds


def _save_confusion_matrix(
    labels: list[int],
    preds: list[int],
    *,
    num_classes: int,
    out_npy_path,
    out_plot_path,
) -> None:
    cm = confusion_matrix(
        labels,
        preds,
        labels=list(range(num_classes)),
        normalize="true",
    )
    np.save(out_npy_path, cm)

    cm = (cm *100).round(0).astype(int)

    fig, ax = plt.subplots(figsize=(15, 13))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(range(num_classes)),
    )
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True,
        xticks_rotation="vertical",
    )
    ax.tick_params(axis='both', which='major', labelsize=8)
    for text in display.text_.ravel():
        text.set_fontsize(7)
    ax.set_title("Validation Confusion Matrix (Best Model, Row-Normalized)")
    fig.tight_layout()
    fig.savefig(out_plot_path, dpi=300)
    plt.close(fig)


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
    patience = int(select_cfg(training_cfg, "patience", 10))

    # Output directories (checkpoints, metrics, eval)
    out_dirs = ensure_output_dirs(cfg)
    best_path = out_dirs["ckpt"] / "best.pt"
    metrics_path = out_dirs["out"] / "metrics_train_val.json"
    cm_npy_path = out_dirs["out"] / "val_confusion_matrix_best.npy"
    cm_plot_path = out_dirs["out"] / "val_confusion_matrix_best.png"

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
    num_classes = int(select_cfg(model_cfg, "num_classes", 37))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("device=%s", device)

    # Optimizer and loss function setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    metrics_history: list[dict] = []
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

        metrics_history.append(
            {
                "epoch": epoch,
                "train": {
                    "loss": _to_json_float(train_metrics.loss),
                    "acc": _to_json_float(train_metrics.acc),
                    "macro_f1": _to_json_float(train_metrics.macro_f1),
                },
                "val": {
                    "loss": _to_json_float(val_metrics.loss),
                    "acc": _to_json_float(val_metrics.acc),
                    "macro_f1": _to_json_float(val_metrics.macro_f1),
                },
            }
        )

        # Checkpointing best model
        if val_loader is not None and val_metrics.acc > best_val_acc:
            best_val_acc = val_metrics.acc
            best_epoch = epoch
            epochs_without_improvement = 0
            
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
        elif val_loader is not None:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d, best val_acc=%.4f",
                    epoch,
                    best_val_acc,
                )
                break

    metrics_payload = {
        "best_epoch": best_epoch if best_epoch >= 1 else None,
        "best_val_acc": _to_json_float(best_val_acc),
        "history": metrics_history,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("saved_metrics path=%s", metrics_path)

    if val_loader is not None and best_epoch >= 1 and best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        val_labels, val_preds = _collect_val_predictions(
            model=model,
            loader=val_loader,
            device=device,
            input_key=input_key,
        )
        _save_confusion_matrix(
            labels=val_labels,
            preds=val_preds,
            num_classes=num_classes,
            out_npy_path=cm_npy_path,
            out_plot_path=cm_plot_path,
        )
        logger.info("saved_confusion_matrix npy=%s plot=%s", cm_npy_path, cm_plot_path)


if __name__ == "__main__":
    main()
