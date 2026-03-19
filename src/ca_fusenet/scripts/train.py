from __future__ import annotations

import hydra
import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf
from ca_fusenet.training.optimizer import buildOptimizer
from ca_fusenet.training.data import buildDataloaders
from ca_fusenet.training.criterion import buildCriterion
from ca_fusenet.training.trainer import trainer
from ca_fusenet.utils.metrics import finalize_training
from ca_fusenet.utils.engine import ensure_output_dirs
import logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:

    # Set random seed for reproducibility
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # Loading configurations
    training_cfg = cfg.get("training", None)
    data_cfg = cfg.get("data", None)
    model_cfg = cfg.get("model", None)

    # Output directories (checkpoints, metrics, eval)
    out_dirs = ensure_output_dirs(cfg)
    best_path = out_dirs["ckpt"] / "best.pt"
    metrics_path = out_dirs["reports"] / "metrics_train_val.json"
    cm_npy_path = out_dirs["reports"] / "val_confusion_matrix_best.npy"
    cm_plot_path = out_dirs["reports"] / "val_confusion_matrix_best.png"

    # Logging configurations
    logger.info("config.training=%s", OmegaConf.to_container(training_cfg, resolve=True))
    logger.info("config.model=%s", OmegaConf.to_container(model_cfg, resolve=True))
    logger.info("config.data=%s", OmegaConf.to_container(data_cfg, resolve=True))

    # Data loaders
    train_loader, val_loader, all_labels = buildDataloaders(seed, data_cfg, training_cfg)
    
    # Model
    model = hydra.utils.instantiate(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    logger.info("device=%s", device)

    # Optimizer
    optimizer, has_encoder_group = buildOptimizer(model, training_cfg)
    
    # Loss function
    criterion = buildCriterion(all_labels, device)

    # Training loop
    best_epoch, best_val_acc, metrics_history = trainer(
        cfg=cfg,
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

    # Saving training metrics
    finalize_training(
        metrics_history=metrics_history,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        metrics_path=metrics_path,
        model=model,
        val_loader=val_loader,
        device=device,
        best_path=best_path,
        evaluation_params={
            "input_key": training_cfg.get("input_key", "unknown"),
            "num_classes": model_cfg.get("num_classes", 37),
            "cm_npy_path": cm_npy_path,
            "cm_plot_path": cm_plot_path,
        }
    )

if __name__ == "__main__":
    main()
