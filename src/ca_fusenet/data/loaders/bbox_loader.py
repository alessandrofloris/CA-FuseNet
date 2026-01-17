from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from .base import BaseLoader, LoaderError, load_npy


class BBoxLoader(BaseLoader):
    """Load bounding box sequences as (N, T, 4) with [x1, y1, w, h]."""

    def load(self, rel_path: str | Path) -> np.ndarray:
        path = self.resolve(rel_path)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = load_npy(path)
        else:
            raise LoaderError(
                f"Unsupported bbox file extension: {path.suffix} for {path}"
            )

        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[2] == 4:
            normalized = arr
        else:
            raise LoaderError(
                "BBox data expected shape (N, T, 4) with last dim 4; "
                f"got shape={arr.shape}"
            )

        return np.ascontiguousarray(normalized)
