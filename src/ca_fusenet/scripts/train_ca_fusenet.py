from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ca_fusenet.models.fusion.ca_fusenet import CaFuseNet
from ca_fusenet.utils.engine import (
    build_dataloader,
    get_training_cfg,
    get_data_cfg,
    get_model_cfg,
    get_split_data_cfg,
    select_cfg,
    ensure_output_dirs,
    _read_d_output,
    _resolve_encoder_cfgs,
    run_epoch_fusion,
    _sanity_check_fusion
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    training_cfg = get_training_cfg(cfg)
    seed = int(select_cfg(training_cfg, "seed", select_cfg(cfg, "seed", 42)))
    torch.manual_seed(seed)

    # Training configuration
    batch_size = int(select_cfg(training_cfg, "batch_size", 32))
    num_workers = int(select_cfg(training_cfg, "num_workers", 0))
    val_split = float(select_cfg(training_cfg, "val_split", 0.1))
    lr_fusion = float(select_cfg(training_cfg, "lr_fusion", 1e-3))
    lr_encoder = float(select_cfg(training_cfg, "lr_encoder", 1e-4))
    weight_decay = float(select_cfg(training_cfg, "weight_decay", 1e-2))
    epochs = int(select_cfg(training_cfg, "epochs", 5))
    freeze_encoder_epochs = int(select_cfg(training_cfg, "freeze_encoder_epochs", 0))
    patience = int(select_cfg(training_cfg, "patience", 10))
    eta_min = float(select_cfg(training_cfg, "eta_min", 1e-6))

    # Output directories (checkpoints, metrics, eval)
    out_dirs = ensure_output_dirs(cfg)
    best_path = out_dirs["ckpt"] / "best.pt"

    # Data configuration
    data_cfg = get_data_cfg(cfg)
    train_data_cfg = get_split_data_cfg(data_cfg, "train")
    val_data_cfg = get_split_data_cfg(data_cfg, "val")

    # Model & Encoders configuration
    model_cfg = get_model_cfg(cfg)
    video_encoder_cfg, pose_encoder_cfg = _resolve_encoder_cfgs(cfg, training_cfg)

    # Logging configurations
    logger.info("config.training=%s", OmegaConf.to_container(training_cfg, resolve=True))
    logger.info("config.model=%s", OmegaConf.to_container(model_cfg, resolve=True))
    logger.info("config.data=%s", OmegaConf.to_container(data_cfg, resolve=True))
    logger.info("config.encoder.video=%s", OmegaConf.to_container(video_encoder_cfg, resolve=True))
    logger.info("config.encoder.pose=%s", OmegaConf.to_container(pose_encoder_cfg, resolve=True))

    # Dataset instantiation
    if train_data_cfg is not None:
        train_dataset = hydra.utils.instantiate(train_data_cfg)
    else:
        train_dataset = hydra.utils.instantiate(data_cfg)

    # Split the train dataset 
    if val_data_cfg is not None:
        train_ds = train_dataset
        val_ds = hydra.utils.instantiate(val_data_cfg)
    else:
        if val_split < 0.0 or val_split >= 1.0:
            raise ValueError(f"training.val_split must be in [0, 1); got {val_split}")

        n_samples = len(train_dataset)
        if n_samples < 2:
            raise ValueError("Dataset too small for train/val split (need N>=2).")

        if val_split == 0.0:
            raise ValueError(
                "Validation split is required for macro-F1 checkpointing and early stopping; "
                "set training.val_split > 0 or provide data.val."
            )

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

    # Train Dataloader
    pin_memory = torch.cuda.is_available()
    train_loader = build_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Validation Dataloader
    val_loader = build_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Encoders instantiation
    video_encoder = hydra.utils.instantiate(video_encoder_cfg)
    pose_encoder = hydra.utils.instantiate(pose_encoder_cfg)

    # Model instantiation and setup
    d_video = _read_d_output(video_encoder_cfg, "video")
    d_pose = _read_d_output(pose_encoder_cfg, "pose")
    d_common = int(select_cfg(model_cfg, "d_common", 128))
    gate_hidden = int(select_cfg(model_cfg, "gate_hidden", 64))
    d_fused = int(select_cfg(model_cfg, "d_fused", 256))
    n_indicators = int(select_cfg(model_cfg, "n_indicators", 3))
    dropout = float(select_cfg(model_cfg, "dropout", select_cfg(model_cfg, "dropout_gate", 0.3)))
    num_classes = int(select_cfg(model_cfg, "num_classes", 37))

    model = CaFuseNet(
        video_encoder=video_encoder,
        pose_encoder=pose_encoder,
        d_video=d_video,
        d_pose=d_pose,
        d_common=d_common,
        gate_hidden=gate_hidden,
        d_fused=d_fused,
        n_indicators=n_indicators,
        dropout=dropout,
        num_classes=num_classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("device=%s", device)

    # Optimizer, loss, and scheduler setup
    param_groups = model.get_param_groups(lr_encoder=0.0, lr_fusion=lr_fusion)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=eta_min,
    )

    _sanity_check_fusion(model, train_loader, device, num_classes)

    # Training loop
    best_val_f1 = float("-inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        current_encoder_lr = 0.0 if epoch <= freeze_encoder_epochs else lr_encoder
        if epoch == freeze_encoder_epochs + 1:
            logger.info(
                "Unfreezing encoders at epoch %d with lr_encoder=%s",
                epoch,
                lr_encoder,
            )
        optimizer.param_groups[0]["lr"] = current_encoder_lr

        train_metrics = run_epoch_fusion(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            train=True,
        )
        val_metrics = run_epoch_fusion(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=None,
            train=False,
        )

        scheduler.step()
        optimizer.param_groups[0]["lr"] = current_encoder_lr

        logger.info(
            "epoch=%d/%d train_loss=%.4f train_acc=%.4f train_f1=%.4f "
            "val_loss=%.4f val_acc=%.4f val_f1=%.4f alpha_mean=%.4f alpha_std=%.4f "
            "lr_fusion=%.6g lr_encoder=%.6g",
            epoch,
            epochs,
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["macro_f1"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
            val_metrics["mean_alpha"],
            val_metrics["std_alpha"],
            optimizer.param_groups[1]["lr"],
            optimizer.param_groups[0]["lr"],
        )

        # Checkpointing best model based on validation macro-F1 score
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                best_path,
            )
            logger.info("checkpoint=best path=%s val_f1=%.4f", best_path, best_val_f1)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d, best val_f1=%.4f",
                    epoch,
                    best_val_f1,
                )
                break


if __name__ == "__main__":
    main()
