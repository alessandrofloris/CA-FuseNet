import math
import torch
from omegaconf import OmegaConf
from ca_fusenet.utils.engine import run_epoch, run_epoch_fusion
import logging

logger = logging.getLogger(__name__)


def _to_json_float(value: float) -> float | None:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)

def _store_metrics_history(metrics_history, epoch, train_metrics, val_metrics):
    metrics_history.append(
            {
                "epoch": epoch,
                "train": {
                    "loss": _to_json_float(train_metrics["loss"]),
                    "acc": _to_json_float(train_metrics["accuracy"]),
                    "macro_f1": _to_json_float(train_metrics["macro_f1"]),
                },
                "val": {
                    "loss": _to_json_float(val_metrics["loss"]),
                    "acc": _to_json_float(val_metrics["accuracy"]),
                    "macro_f1": _to_json_float(val_metrics["macro_f1"]),
                },
            }
        )

def _save_checkpoint(model, cfg, epoch, path, val_acc):
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch,
        },
        path,
    )
    logger.info("Best checkpoint saved to %s with val_acc=%.4f", path, val_acc)

def _maybe_unfreeze_encoder(epoch, has_encoder_group, freeze_encoder_epochs, optimizer, lr_encoder):
    if not has_encoder_group or epoch != freeze_encoder_epochs + 1:
        return False
    
    optimizer.param_groups[0]["lr"] = lr_encoder

    logger.info("Unfreezing encoder trainable params at epoch %d with lr_encoder=%s", epoch, lr_encoder)

    return True

def trainer(cfg, best_path, training_confg, criterion, has_encoder_group, model, optimizer, train_loader, val_loader, device):

    input_key = training_confg.get("input_key", None)
    epochs = training_confg.get("epochs", 10)
    freeze_encoder_epochs = training_confg.get("freeze_encoder_epochs", 0)
    lr_encoder = training_confg.get("lr_encoder", training_confg.get("lr", 1e-4))
    patience = training_confg.get("patience", 5)

    # Training loop
    best_val_acc = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    metrics_history: list[dict] = []
    
    for epoch in range(1, epochs + 1):
        _maybe_unfreeze_encoder(epoch, has_encoder_group, freeze_encoder_epochs, optimizer, lr_encoder)

        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, train=True, input_key=input_key)
        val_metrics = run_epoch(model, val_loader, device, criterion, None, train=False, input_key=input_key)
        logger.info("epoch=%d/%d %s %s", epoch, epochs, train_metrics.prefixed("train_"), val_metrics.prefixed("val_"))

        _store_metrics_history(metrics_history, epoch, train_metrics, val_metrics)

        # Patience mechanism and checkpointing best model
        if val_loader is None:
            continue

        if val_metrics.acc > best_val_acc:
            best_val_acc = val_metrics.acc
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(model, cfg, best_epoch, best_path, best_val_acc)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d, best val_acc=%.4f", epoch, best_val_acc)
                break    
        
    return best_epoch, best_val_acc, metrics_history

def trainerMultiModal(cfg, scheduler, best_path, training_confg, criterion, has_encoder_group, model, optimizer, train_loader, val_loader, device):

    input_key = training_confg.get("input_key", None)
    epochs = training_confg.get("epochs", 10)
    freeze_encoder_epochs = training_confg.get("freeze_encoder_epochs", 0)
    lr_encoder = training_confg.get("lr_encoder", training_confg.get("lr", 1e-4))
    patience = training_confg.get("patience", 5)

    # Training loop
    best_val_acc = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    metrics_history: list[dict] = []
    
    for epoch in range(1, epochs + 1):
        unfreeze = _maybe_unfreeze_encoder(epoch, has_encoder_group, freeze_encoder_epochs, optimizer, lr_encoder)

        train_metrics = run_epoch_fusion(model, train_loader, device, criterion, optimizer, train=True)
        val_metrics = run_epoch_fusion(model, val_loader, device, criterion, None, train=False)

        #if unfreeze:        
        #    scheduler.step()

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

        _store_metrics_history(metrics_history, epoch, train_metrics, val_metrics)

        # Patience mechanism and checkpointing best model
        if val_loader is None:
            continue

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(model, cfg, best_epoch, best_path, best_val_acc)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d, best val_acc=%.4f", epoch, best_val_acc)
                break    
        
    return best_epoch, best_val_acc, metrics_history