from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseLoader, LoaderError, load_pkl


@dataclass(frozen=True)
class LabelRecord:
    sample_id: str
    label: Any
    video_path: str 
    frame: int 


class LabelLoader(BaseLoader):
    """Load dataset-level label store from a tuple of lists."""

    def load_store(
        self, rel_path: str | Path
    ) -> tuple[list[str], list[str], list[int], list[int]]:
        path = self.resolve(rel_path)
        data = load_pkl(path)
        if not isinstance(data, (tuple, list)):
            raise LoaderError(
                "Label store expected tuple of 4 lists: "
                "(sample_names, labels, frames, video_paths); "
                f"got {type(data)} from {path}"
            )
        if len(data) != 4:
            raise LoaderError(
                "Label store expected exactly 4 elements: "
                "(ssample_names, labels, frames, video_paths); "
                f"got len={len(data)} from {path}"
            )
        
        if not isinstance(data, (tuple, list)) or len(data) != 4:
            raise LoaderError(
                "Label store expected tuple/list of 4 lists: "
                "(sample_names, labels, frames, video_paths)"
            )
        sample_names, labels, frames, video_paths = data

        for name, field in (
            ("sample_name_list", sample_names),
            ("video_path_list", video_paths),
            ("label_list", labels),
            ("frame_list", frames),
        ):
            if not isinstance(field, (list, tuple)):
                raise LoaderError(
                    f"Label store expected {name} as list/tuple; got {type(field)}"
                )

        sample_names = list(sample_names)
        video_paths = list(video_paths)
        labels = list(labels)
        frames = list(frames)
        lengths = {
            "sample_name_list": len(sample_names),
            "video_path_list": len(video_paths),
            "label_list": len(labels),
            "frame_list": len(frames),
        }
        if len(set(lengths.values())) != 1:
            raise LoaderError(
                "Label store lists must have equal lengths; "
                f"got lengths {lengths} from {path}"
            )

        return sample_names, video_paths, labels, frames

    def get_sample(
        self, store: tuple[list[str], list[str], list[int], list[int]], idx: int
    ) -> LabelRecord:
        sample_names, video_paths, labels, frames = store
        if idx < 0 or idx >= len(sample_names):
            raise LoaderError(
                f"Label store index out of range: idx={idx}, N={len(sample_names)}"
            )

        sample_id = str(sample_names[idx])
        raw_video = video_paths[idx]
        video_path = None if raw_video is None else str(raw_video)
        label = labels[idx]
        frame = frames[idx]

        return LabelRecord(
            sample_id=sample_id, label=label, video_path=video_path, frame=frame
        )