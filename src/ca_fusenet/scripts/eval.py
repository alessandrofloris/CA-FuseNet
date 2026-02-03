from __future__ import annotations

from pathlib import Path
from typing import Any

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

from ca_fusenet.utils.engine import (
    build_dataloader,
    get_data_cfg,
    get_input_key,
    get_model_cfg,
    get_split_data_cfg,
    run_epoch,
    select_cfg,
)

logger = logging.getLogger(__name__)


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
    for split in ("test", "val", "train"):
        split_cfg = get_split_data_cfg(data_cfg, split)
        if split_cfg is not None:
            return split_cfg, split
    if "_target_" in data_cfg:
        return data_cfg, "data"
    raise ValueError("No dataset configuration found for evaluation.")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint_path = select_cfg(cfg, "eval.checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError("eval.checkpoint_path is required.")

    logger.info("eval.config=%s", OmegaConf.to_container(cfg.eval, resolve=True))
    logger.info("eval.checkpoint_path=%s", checkpoint_path)

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

    model = hydra.utils.instantiate(model_cfg)
    model.load_state_dict(checkpoint["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("device=%s", device)

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

    batch_size = int(
        select_cfg(
            cfg,
            "eval.batch_size",
            select_cfg(saved_cfg, "training.batch_size", 32),
        )
    )
    num_workers = int(
        select_cfg(
            cfg,
            "eval.num_workers",
            select_cfg(saved_cfg, "training.num_workers", 0),
        )
    )
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

    criterion = nn.CrossEntropyLoss()
    test_metrics = run_epoch(
        model, test_loader, device, criterion, None, train=False, input_key=input_key
    )

    logger.info("%s", test_metrics.prefixed("test_"))


if __name__ == "__main__":
    main()
