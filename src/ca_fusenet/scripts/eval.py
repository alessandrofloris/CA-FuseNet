from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import logging
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
import torch
from torch import nn

from ca_fusenet.utils.engine import (
    build_dataloader,
    ensure_output_dirs,
    get_data_cfg,
    get_input_key,
    get_model_cfg,
    get_split_data_cfg,
    run_epoch,
    select_cfg,
    validate_batch,
)

logger = logging.getLogger(__name__)


def _to_json_float(value: float) -> float | None:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _collect_predictions(
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


def _resolve_class_names(
    saved_cfg: DictConfig,
    dataset: Any,
    num_classes: int,
) -> list[str]:
    def _normalize_from_sequence(values: Any) -> list[str] | None:
        if not isinstance(values, (list, tuple)):
            return None
        if len(values) < num_classes:
            return None
        return [str(values[idx]) for idx in range(num_classes)]

    def _normalize_from_mapping(values: Any) -> list[str] | None:
        if not isinstance(values, dict):
            return None
        class_names = [str(idx) for idx in range(num_classes)]
        found = False
        for idx in range(num_classes):
            if idx in values:
                class_names[idx] = str(values[idx])
                found = True
                continue
            key = str(idx)
            if key in values:
                class_names[idx] = str(values[key])
                found = True
        if found:
            return class_names
        return None

    candidate_sources = [
        getattr(dataset, "class_names", None),
        getattr(dataset, "classes", None),
        select_cfg(saved_cfg, "data.class_names", None),
        select_cfg(saved_cfg, "class_names", None),
    ]

    for source in candidate_sources:
        class_names = _normalize_from_sequence(source)
        if class_names is not None:
            return class_names
        class_names = _normalize_from_mapping(source)
        if class_names is not None:
            return class_names

    return [str(idx) for idx in range(num_classes)]


def _save_per_class_metrics_csv(
    labels: list[int],
    preds: list[int],
    *,
    num_classes: int,
    class_names: list[str],
    out_csv_path: Path,
) -> None:
    report = classification_report(
        labels,
        preds,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )

    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_index",
                "class_name",
                "precision",
                "recall",
                "f1",
                "support",
            ],
        )
        writer.writeheader()
        for class_index in range(num_classes):
            class_key = str(class_index)
            metrics = report.get(class_key, {})
            writer.writerow(
                {
                    "class_index": class_index,
                    "class_name": class_names[class_index],
                    "precision": float(metrics.get("precision", 0.0)),
                    "recall": float(metrics.get("recall", 0.0)),
                    "f1": float(metrics.get("f1-score", 0.0)),
                    "support": int(metrics.get("support", 0)),
                }
            )


def _save_confusion_matrix(
    labels: list[int],
    preds: list[int],
    *,
    num_classes: int,
    out_npy_path: Path,
    out_plot_path: Path,
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
    ax.set_title("Evaluation Confusion Matrix (Row-Normalized)")
    fig.tight_layout()
    fig.savefig(out_plot_path, dpi=300)
    plt.close(fig)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a dict.")
    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint missing key: model_state")
    if "cfg" not in checkpoint:
        raise KeyError("Checkpoint missing key: cfg")
    if "epoch" not in checkpoint:
        logger.warning("Checkpoint missing key: epoch")
    return checkpoint


def _restore_cfg(checkpoint: dict[str, Any]) -> DictConfig:
    cfg = OmegaConf.create(checkpoint["cfg"])
    if not isinstance(cfg, DictConfig):
        raise TypeError("Checkpoint cfg is not a DictConfig.")
    return cfg


def _inject_test_cfg(saved_cfg: DictConfig, eval_cfg: DictConfig) -> None:
    saved_data_cfg = get_data_cfg(saved_cfg)
    if saved_data_cfg is None:
        return
    if get_split_data_cfg(saved_data_cfg, "test") is not None:
        return
    eval_data_cfg = get_data_cfg(eval_cfg)
    if eval_data_cfg is None:
        return
    eval_test_cfg = get_split_data_cfg(eval_data_cfg, "test")
    if eval_test_cfg is None:
        return
    saved_data_cfg["test"] = eval_test_cfg


def _resolve_eval_dataset_cfg(data_cfg: DictConfig) -> tuple[DictConfig, str]:
    for split in ("test", "train", "val"):
        split_cfg = get_split_data_cfg(data_cfg, split)
        if split_cfg is not None:
            return split_cfg, split
    if "_target_" in data_cfg:
        return data_cfg, "data"
    raise ValueError("No dataset configuration found for evaluation.")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Checkpoint path 
    checkpoint_path = select_cfg(cfg, "eval.checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError("eval.checkpoint_path is required.")

    # Ensuring output directories for metrics
    out_dirs = ensure_output_dirs(cfg)
    metrics_path = out_dirs["reports"] / "metrics_eval.json"
    cm_npy_path = out_dirs["reports"] / "confusion_matrix_eval.npy"
    cm_plot_path = out_dirs["reports"] / "confusion_matrix_eval.png"
    cm_raw_npy_path = out_dirs["reports"] / "confusion_matrix_raw.npy"
    per_class_metrics_csv_path = out_dirs["reports"] / "per_class_metrics.csv"

    logger.info("eval.config=%s", OmegaConf.to_container(cfg.eval, resolve=True))
    logger.info("eval.checkpoint_path=%s", checkpoint_path)

    # Load checkpoint and restore configuration
    checkpoint = _load_checkpoint(Path(checkpoint_path))
    saved_cfg = _restore_cfg(checkpoint)
    _inject_test_cfg(saved_cfg, cfg)

    model_cfg = get_model_cfg(saved_cfg)
    data_cfg = get_data_cfg(saved_cfg)

    logger.info(
        "checkpoint.training=%s",
        OmegaConf.to_container(saved_cfg.training, resolve=True),
    )
    logger.info(
        "checkpoint.model=%s",
        OmegaConf.to_container(model_cfg, resolve=True),
    )
    logger.info(
        "checkpoint.data=%s",
        OmegaConf.to_container(data_cfg, resolve=True) if data_cfg is not None else None,
    )

    seed = int(select_cfg(saved_cfg, "seed", 42))
    torch.manual_seed(seed)

    input_key = get_input_key(saved_cfg)
    if input_key is None:
        raise ValueError("training.input_key is required in the checkpoint config.")

    # Model instantiation 
    model = hydra.utils.instantiate(model_cfg)
    num_classes = int(select_cfg(model_cfg, "num_classes", 37))
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("device=%s", device)

    # Dataset and dataloader setup
    if data_cfg is None:
        raise ValueError("No data configuration found in the checkpoint config.")
    dataset_cfg, split_name = _resolve_eval_dataset_cfg(data_cfg)
    test_dataset = hydra.utils.instantiate(dataset_cfg)
    if len(test_dataset) <= 0:
        raise ValueError(
            f"{split_name} dataset is empty; check data configuration and artifacts."
        )
    if split_name != "test":
        logger.warning(
            "no test split configured; using %s split for evaluation",
            split_name,
        )

    batch_size = int(select_cfg(cfg, "eval.batch_size", 32))
    num_workers = int(select_cfg(cfg, "eval.num_workers", 0))
    logger.info(
        "eval: batch_size=%d num_workers=%d",
        batch_size,
        num_workers,
    )

    test_loader = build_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Loss function and evaluation loop
    criterion = nn.CrossEntropyLoss()
    test_metrics = run_epoch(
        model, test_loader, device, criterion, None, train=False, input_key=input_key
    )

    logger.info("%s", test_metrics.prefixed("test_"))

    eval_labels, eval_preds = _collect_predictions(
        model=model,
        loader=test_loader,
        device=device,
        input_key=input_key,
    )
    _save_confusion_matrix(
        labels=eval_labels,
        preds=eval_preds,
        num_classes=num_classes,
        out_npy_path=cm_npy_path,
        out_plot_path=cm_plot_path,
    )
    cm_raw = confusion_matrix(
        eval_labels,
        eval_preds,
        labels=list(range(num_classes)),
    )
    np.save(cm_raw_npy_path, cm_raw)
    logger.info(
        "saved_confusion_matrix normalized_npy=%s raw_npy=%s plot=%s",
        cm_npy_path,
        cm_raw_npy_path,
        cm_plot_path,
    )

    class_names = _resolve_class_names(saved_cfg, test_dataset, num_classes)
    _save_per_class_metrics_csv(
        labels=eval_labels,
        preds=eval_preds,
        num_classes=num_classes,
        class_names=class_names,
        out_csv_path=per_class_metrics_csv_path,
    )
    logger.info("saved_per_class_metrics path=%s", per_class_metrics_csv_path)

    weighted_f1 = float(f1_score(eval_labels, eval_preds, average="weighted", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(eval_labels, eval_preds))
    mcc = float(matthews_corrcoef(eval_labels, eval_preds))

    metrics_payload = {
        "split": split_name,
        "loss": _to_json_float(test_metrics.loss),
        "acc": _to_json_float(test_metrics.acc),
        "macro_f1": _to_json_float(test_metrics.macro_f1),
        "weighted_f1": _to_json_float(weighted_f1),
        "balanced_accuracy": _to_json_float(balanced_acc),
        "matthews_corrcoef": _to_json_float(mcc),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("saved_metrics path=%s", metrics_path)


if __name__ == "__main__":
    main()
