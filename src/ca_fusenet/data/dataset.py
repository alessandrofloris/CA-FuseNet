from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from torch.utils.data import Dataset

from .contracts import (
    BBoxSequenceContract,
    BBoxStoreContract,
    IndicatorsContract,
    IndicatorsStoreContract,
    JointStoreContract,
    LabelStoreContract,
    PoseSequenceContract,
)
from .loaders import BBoxLoader, IndicatorsLoader, JointLoader, LabelLoader


class CAFuseNetDataset(Dataset):
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
        validate_stores: bool = True,
        validate_samples: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self.coord_space = coord_space
        self.validate_samples = validate_samples

        self.bbox_loader = BBoxLoader(self.root_dir)
        self.indicators_loader = IndicatorsLoader(self.root_dir)
        self.joint_loader = JointLoader(self.root_dir)
        self.label_loader = LabelLoader(self.root_dir)

        self.bbox_store = self.bbox_loader.load_store(bbox_file, mmap=mmap)
        self.indicators_store = self.indicators_loader.load_store(indicators_file, mmap=mmap)
        self.joint_store = self.joint_loader.load_store(joint_file, mmap=mmap)
        self.labels_store = self.label_loader.load_store(label_file)

        if validate_stores:
            BBoxStoreContract.from_array(
                self.bbox_store, coord_space=self.coord_space
            ).validate()
            IndicatorsStoreContract.from_array(self.indicators_store).validate()
            JointStoreContract.from_array(
                self.joint_store, coord_space=self.coord_space
            ).validate()
            LabelStoreContract.from_tuple(self.labels_store).validate()

        n_bbox = self.bbox_store.shape[0]
        n_ind = self.indicators_store.shape[0]
        n_joint = self.joint_store.shape[0]
        n_lbl = len(self.labels_store[0])
        if len({n_bbox, n_ind, n_joint, n_lbl}) != 1:
            raise ValueError(
                "N mismatch across stores: "
                f"bbox={n_bbox}, indicators={n_ind}, joint={n_joint}, labels={n_lbl}"
            )
        self.N = n_bbox


    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict[str, Any]:
        
        if idx < 0:
            idx += self.N
        if idx < 0 or idx >= self.N:
            raise IndexError(f"Index out of range: idx={idx}, N={self.N}")

        bbox_sample = self.bbox_loader.get_sample(self.bbox_store, idx)
        ind_sample = self.indicators_loader.get_sample(self.indicators_store, idx)
        pose_sample = self.joint_loader.get_sample(self.joint_store, idx)
        label_record = self.label_loader.get_sample(self.labels_store, idx)

        frame = None
        if isinstance(label_record.meta, dict):
            frame = label_record.meta.get("frame")

        if self.validate_samples:
            BBoxSequenceContract.from_array(
                bbox_sample, coord_space=self.coord_space
            ).validate()
            IndicatorsContract.from_array(ind_sample).validate()
            PoseSequenceContract.from_array(
                pose_sample, coord_space=self.coord_space
            ).validate()

        return {
            "sample_id": str(label_record.sample_id),
            "video_path": label_record.video_path,
            "frame": frame,
            "label": label_record.label,
            "bbox_xywh": np.asarray(bbox_sample),
            "indicators": np.asarray(ind_sample),
            "pose": np.asarray(pose_sample),
        }
