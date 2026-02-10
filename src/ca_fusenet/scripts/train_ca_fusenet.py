from __future__ import annotations

from contextlib import nullcontext
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
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
)

logger = logging.getLogger(__name__)


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


def _sanity_check(
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    training_cfg = get_training_cfg(cfg)
    seed = int(select_cfg(training_cfg, "seed", select_cfg(cfg, "seed", 42)))
    torch.manual_seed(seed)

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

    if epochs <= 0:
        raise ValueError(f"training.epochs must be > 0; got {epochs}")
    if freeze_encoder_epochs < 0:
        raise ValueError(
            f"training.freeze_encoder_epochs must be >= 0; got {freeze_encoder_epochs}"
        )
    if patience < 0:
        raise ValueError(f"training.patience must be >= 0; got {patience}")

    out_dirs = ensure_output_dirs(cfg)
    best_path = out_dirs["ckpt"] / "best.pt"

    data_cfg = get_data_cfg(cfg)
    train_data_cfg = get_split_data_cfg(data_cfg, "train")
    val_data_cfg = get_split_data_cfg(data_cfg, "val")
    model_cfg = get_model_cfg(cfg)
    video_encoder_cfg, pose_encoder_cfg = _resolve_encoder_cfgs(cfg, training_cfg)

    logger.info("config.training=%s", OmegaConf.to_container(training_cfg, resolve=True))
    logger.info("config.model=%s", OmegaConf.to_container(model_cfg, resolve=True))
    logger.info("config.data=%s", OmegaConf.to_container(data_cfg, resolve=True))
    logger.info("config.encoder.video=%s", OmegaConf.to_container(video_encoder_cfg, resolve=True))
    logger.info("config.encoder.pose=%s", OmegaConf.to_container(pose_encoder_cfg, resolve=True))

    if train_data_cfg is not None:
        train_dataset = hydra.utils.instantiate(train_data_cfg)
    else:
        train_dataset = hydra.utils.instantiate(data_cfg)

    if len(train_dataset) <= 0:
        raise ValueError("Training dataset is empty; check data configuration and artifacts.")

    if val_data_cfg is not None:
        train_ds = train_dataset
        val_ds = hydra.utils.instantiate(val_data_cfg)
        if len(val_ds) <= 0:
            raise ValueError("Validation dataset is empty; check data configuration and artifacts.")
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

    pin_memory = torch.cuda.is_available()
    train_loader = build_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    video_encoder = hydra.utils.instantiate(video_encoder_cfg)
    pose_encoder = hydra.utils.instantiate(pose_encoder_cfg)

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

    param_groups = model.get_param_groups(lr_encoder=0.0, lr_fusion=lr_fusion)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=eta_min,
    )

    _sanity_check(model, train_loader, device, num_classes)

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

        if val_metrics["accuracy"] > best_val_f1:
            best_val_f1 = val_metrics["accuracy"]
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
