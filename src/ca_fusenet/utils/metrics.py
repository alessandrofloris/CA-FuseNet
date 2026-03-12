import math
import torch
from torch import nn
import json
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from ca_fusenet.utils.engine import validate_batch
from ca_fusenet.utils.class_mapping import ITWPolimiClassMapping
import logging

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

    mapping_classes = ITWPolimiClassMapping()
    class_names = [mapping_classes.get_class_name(i) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(15, 13))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
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

def finalize_training(
    metrics_history, best_epoch, best_val_acc, metrics_path, 
    model, val_loader, device, best_path, evaluation_params
):
    
    # Save training metrics to JSON
    _save_metrics_json(metrics_path, best_epoch, best_val_acc, metrics_history)

    # Final evaluation on the validation set using the best checkpoint
    if val_loader is not None and best_epoch >= 1 and best_path.exists():
        logger.info(f"Loading best checkpoint from {best_path} for final evaluation on the validation set...")
        
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        
        _run_final_eval(model, val_loader, device, evaluation_params)

def _save_metrics_json(path, epoch, acc, history):
    payload = {
        "best_epoch": epoch if epoch >= 1 else None,
        "best_val_acc": _to_json_float(acc),
        "history": history,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"saved_metrics path={path}")

def _run_final_eval(model, loader, device, p):
    labels, preds = _collect_val_predictions(
        model=model, loader=loader, device=device, input_key=p["input_key"]
    )
    _save_confusion_matrix(
        labels=labels,
        preds=preds,
        num_classes=p["num_classes"],
        out_npy_path=p["cm_npy_path"],
        out_plot_path=p["cm_plot_path"],
    )
    logger.info(f"saved_confusion_matrix npy={p['cm_npy_path']} plot={p['cm_plot_path']}")