from __future__ import annotations

from typing import Any

import numpy as np
import torch


def cafusenet_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cafusenet_collate_fn expected non-empty batch")

    required_keys = {
        "pose",
        "bbox_xywh",
        "indicators",
        "label",
        "sample_id",
        "video_path",
        "frame",
        "tublet",
    }
    for idx, sample in enumerate(batch):
        for key in required_keys:
            if key not in sample:
                raise KeyError(f"Missing key '{key}' in sample index {idx}")

    t_values = {
        "pose": [sample["pose"].shape[1] for sample in batch],
        "bbox_xywh": [sample["bbox_xywh"].shape[0] for sample in batch],
        "indicators": [sample["indicators"].shape[0] for sample in batch],
    }
    t_set = set(t_values["pose"]) | set(t_values["bbox_xywh"]) | set(t_values["indicators"])
    if len(t_set) != 1:
        raise ValueError(f"Time dimension mismatch across batch: {t_values}")

    for idx, sample in enumerate(batch):
        pose = sample["pose"]
        bbox = sample["bbox_xywh"]
        ind = sample["indicators"]
        tublet = sample["tublet"]
        if pose.shape[0] != 3:
            raise ValueError(
                f"pose first dim expected 3; got shape={pose.shape} at sample index {idx}"
            )
        if bbox.shape[-1] != 4:
            raise ValueError(
                f"bbox_xywh last dim expected 4; got shape={bbox.shape} at sample index {idx}"
            )
        if ind.shape[-1] != 3:
            raise ValueError(
                f"indicators last dim expected 3; got shape={ind.shape} at sample index {idx}"
            )
        if tublet.shape[0] != 3:
            raise ValueError(
                f"tublet first dim expected 3; got shape={tublet.shape} at sample index {idx}"
            )

    poses = np.ascontiguousarray(np.stack([sample["pose"] for sample in batch], axis=0))
    bboxes = np.ascontiguousarray(
        np.stack([sample["bbox_xywh"] for sample in batch], axis=0)
    )
    indicators = np.ascontiguousarray(
        np.stack([sample["indicators"] for sample in batch], axis=0)
    )
    tublets = np.ascontiguousarray(
        np.stack([sample["tublet"] for sample in batch], axis=0)
    )
    
    labels = torch.as_tensor([sample["label"] for sample in batch], dtype=torch.long)
    sample_ids = [sample["sample_id"] for sample in batch]
    video_paths = [sample["video_path"] for sample in batch]
    frames = [sample["frame"] for sample in batch]

    return {
        "pose": torch.as_tensor(poses, dtype=torch.float32),
        "bbox_xywh": torch.as_tensor(bboxes, dtype=torch.float32),
        "indicators": torch.as_tensor(indicators, dtype=torch.float32),
        "tublet": torch.as_tensor(tublets, dtype=torch.float32),
        "label": labels,
        "sample_id": sample_ids,
        "video_path": video_paths,
        "frame": frames,
    }
