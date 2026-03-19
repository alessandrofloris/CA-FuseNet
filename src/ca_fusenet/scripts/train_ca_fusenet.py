from __future__ import annotations

import json
from json import encoder
import logging
import math

from ca_fusenet.training import trainer
from ca_fusenet.training.data import buildDataloaders
from ca_fusenet.training.optimizer import buildMultiModalOptimizer, buildOptimizer
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
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


def _to_json_float(value: float) -> float | None:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _collect_val_predictions_fusion(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
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

            output = model(
                video_input=tublet,
                pose_input=pose,
                occlusion_indicators=indicators,
            )
            logits = output["logits"]
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
    fig.savefig(out_plot_path, dpi=200)
    plt.close(fig)

def buildModel(video_encoder, pose_encoder, model_cfg, video_encoder_cfg, pose_encoder_cfg):
    d_video = video_encoder_cfg.get("d_output", None)
    d_pose = pose_encoder_cfg.get("d_output", None)
    d_common = model_cfg.get("d_common", 128)
    gate_hidden = model_cfg.get("gate_hidden", 64)
    d_fused = model_cfg.get("d_fused", 256)
    n_indicators = model_cfg.get("n_indicators", 3)
    dropout = model_cfg.get("dropout", 0.3)
    num_classes = model_cfg.get("num_classes", 37)

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

    return model

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    
    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # Loading configurations
    training_cfg = cfg.get("training", None)
    data_cfg = cfg.get("data", None)
    model_cfg = cfg.get("model", None)
    video_encoder_cfg = cfg.get("video_encoder", None)
    pose_encoder_cfg = cfg.get("pose_encoder", None)

    # Output directories (checkpoints, metrics, eval)
    out_dirs = ensure_output_dirs(cfg)
    best_path = out_dirs["ckpt"] / "best.pt"
    metrics_path = out_dirs["out"] / "metrics_train_val.json"
    cm_npy_path = out_dirs["out"] / "val_confusion_matrix_best.npy"
    cm_plot_path = out_dirs["out"] / "val_confusion_matrix_best.png"

    # Logging configurations
    logger.info("config.training=%s", OmegaConf.to_container(training_cfg, resolve=True))
    logger.info("config.model=%s", OmegaConf.to_container(model_cfg, resolve=True))
    logger.info("config.data=%s", OmegaConf.to_container(data_cfg, resolve=True))
    logger.info("config.encoder.video=%s", OmegaConf.to_container(video_encoder_cfg, resolve=True))
    logger.info("config.encoder.pose=%s", OmegaConf.to_container(pose_encoder_cfg, resolve=True))

    # Data loaders
    train_loader, val_loader, all_labels = buildDataloaders(seed, data_cfg, training_cfg)

    # Encoders instantiation
    video_encoder = hydra.utils.instantiate(video_encoder_cfg)
    ckpt_video = torch.load(training_cfg.get("ckpt_video_path", None))
    state_dict = {k.replace("encoder.", "", 1): v 
              for k, v in ckpt_video["model_state"].items() 
              if k.startswith("encoder.")}
    missing, unexpected = video_encoder.load_state_dict(state_dict, strict=False)
    logger.info("Loaded %d keys, missing %d, unexpected %d", 
        len(state_dict) - len(unexpected), len(missing), len(unexpected))
    
    pose_encoder = hydra.utils.instantiate(pose_encoder_cfg)
    ckpt_pose = torch.load(training_cfg.get("ckpt_pose_path", None))
    state_dict = {k.replace("encoder.", "", 1): v 
              for k, v in ckpt_pose["model_state"].items() 
              if k.startswith("encoder.")}
    missing, unexpected = pose_encoder.load_state_dict(state_dict, strict=False)
    logger.info("Loaded %d keys, missing %d, unexpected %d", 
        len(state_dict) - len(unexpected), len(missing), len(unexpected))
    
    # Model instantiation and setup
    model = buildModel(video_encoder, pose_encoder, model_cfg, video_encoder_cfg, pose_encoder_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("device=%s", device)

    # Optimizer
    optimizer, has_encoder_group = buildMultiModalOptimizer(model, training_cfg)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler setup
    epochs = int(select_cfg(training_cfg, "epochs", 5))
    eta_min = float(select_cfg(training_cfg, "eta_min", 1e-6))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    # Training loop
    best_epoch, best_val_acc, metrics_history = trainer.trainerMultiModal(
        cfg=cfg,
        scheduler=scheduler,
        best_path=best_path,
        training_confg=training_cfg,
        criterion=criterion,
        has_encoder_group=has_encoder_group,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    metrics_payload = {
        "best_epoch": best_epoch if best_epoch >= 1 else None,
        "best_val_acc": _to_json_float(best_val_acc),
        "history": metrics_history,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("saved_metrics path=%s", metrics_path)

    if best_epoch >= 1 and best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        val_labels, val_preds = _collect_val_predictions_fusion(
            model=model,
            loader=val_loader,
            device=device,
        )
        _save_confusion_matrix(
            labels=val_labels,
            preds=val_preds,
            num_classes=37,
            out_npy_path=cm_npy_path,
            out_plot_path=cm_plot_path,
        )
        logger.info("saved_confusion_matrix npy=%s plot=%s", cm_npy_path, cm_plot_path)


if __name__ == "__main__":
    main()
