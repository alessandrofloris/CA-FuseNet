from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseLoader, LoaderError, load_pkl


@dataclass(frozen=True)
class LabelRecord:
    sample_id: str
    label: Any
    video_path: str | None
    meta: dict[str, Any]


class LabelLoader(BaseLoader):
    """Load label/metadata records from *_label.pkl."""

    def load(self, rel_path: str | Path, sample_id: str | None = None) -> LabelRecord:
        path = self.resolve(rel_path)
        data = load_pkl(path)
        if not isinstance(data, (tuple, list)):
            raise LoaderError(
                f"Label record expected tuple/list with (sample_name, video_path, label, frame); "
                f"got {type(data)} from {path}"
            )
        try:
            sample_name, video_path, label, frame = data
        except ValueError as exc:
            raise LoaderError(
                "Label record expected exactly 4 elements: "
                "(sample_name, video_path, label, frame)"
            ) from exc

        if sample_id is None:
            sample_id = str(sample_name)

        video_path = None if video_path is None else str(video_path)
        meta = {"frame": frame}

        return LabelRecord(
            sample_id=sample_id, label=label, video_path=video_path, meta=meta
        )
