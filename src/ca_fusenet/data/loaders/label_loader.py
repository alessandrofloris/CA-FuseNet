from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseLoader, LoaderError, load_pkl

from ..contracts import LabelStoreContract


@dataclass(frozen=True)
class LabelRecord:
    sample_id: str
    label: Any
    video_path: str 
    frame: int 


class LabelLoader(BaseLoader):
    """Load dataset-level label store from a tuple of lists (sample_names, labels, frames, video_paths)."""

    def load_store(
        self, rel_path: str | Path
    ) -> LabelStoreContract:
        path = self.resolve(rel_path)

        data = load_pkl(path)
        
        return LabelStoreContract.from_tuple(data)
    

    def get_sample(
        self, store: LabelStoreContract, idx: int
    ) -> LabelRecord:
        sample_names, labels, frames, video_paths = store.sample_names, store.labels, store.frames, store.video_paths

        sample_id = str(sample_names[idx])
        raw_video = video_paths[idx]
        video_path = None if raw_video is None else str(raw_video)
        label = labels[idx]
        frame = frames[idx]

        return LabelRecord(
            sample_id=sample_id, label=label, video_path=video_path, frame=frame
        )