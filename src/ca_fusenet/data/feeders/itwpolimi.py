from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from torch.utils.data import Dataset

from ..loaders import BBoxLoader, IndicatorsLoader, JointLoader, LabelLoader


class ITWPOLIMI(Dataset):
    """Dataset serving per-sample data from dataset-level artifact stores."""

    def __init__(
        self,
        root_dir: str | Path,
        bbox_file: str,
        indicators_file: str,
        joint_file: str,
        label_file: str,
        *,
        mmap: bool = True,
        coord_space: Literal["normalized", "pixel"] = "normalized",
        num_classes: int,
        validate_samples: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.coord_space = coord_space
        self.validate_samples = validate_samples
        self.num_classes = num_classes

        self.bbox_loader = BBoxLoader(self.root_dir, self.coord_space)
        self.indicators_loader = IndicatorsLoader(self.root_dir)
        self.joint_loader = JointLoader(self.root_dir, self.coord_space)
        self.label_loader = LabelLoader(self.root_dir)

        self.bbox_store = self.bbox_loader.load_store(bbox_file, mmap=mmap)
        self.indicators_store = self.indicators_loader.load_store(indicators_file, mmap=mmap)
        self.joint_store = self.joint_loader.load_store(joint_file, mmap=mmap)
        self.labels_store = self.label_loader.load_store(label_file)

        n_bbox = self.bbox_store.n_samples
        n_ind = self.indicators_store.n_samples
        n_joint = self.joint_store.n_samples
        n_lbl = self.labels_store.n_samples
        if len({n_bbox, n_ind, n_joint, n_lbl}) != 1:
            raise ValueError(
                "N mismatch across stores: "
                f"bbox={n_bbox}, indicators={n_ind}, joint={n_joint}, labels={n_lbl}"
            )
        
        self.N = n_bbox


    def __len__(self) -> int:
        return self.N


    def __getitem__(self, idx: int) -> dict[str, Any]:
        
        # Index normalization & index error check
        if idx < 0:
            idx += self.N
        if idx < 0 or idx >= self.N:
            raise IndexError(f"Index out of range: idx={idx}, N={self.N}")

        bbox_sample = self.bbox_loader.get_sample(self.bbox_store, idx)
        ind_sample = self.indicators_loader.get_sample(self.indicators_store, idx)
        pose_sample = self.joint_loader.get_sample(self.joint_store, idx)
        label_record = self.label_loader.get_sample(self.labels_store, idx)
    
        return {
            "sample_id": str(label_record.sample_id),
            "video_path": label_record.video_path,
            "frame": label_record.frame,
            "label": label_record.label,
            "bbox_xywh": np.asarray(bbox_sample),
            "indicators": np.asarray(ind_sample),
            "pose": np.asarray(pose_sample),
        }
