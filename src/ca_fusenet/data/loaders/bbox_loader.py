from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseLoader, LoaderError, load_npy


class BBoxLoader(BaseLoader):
    """Load dataset-level bounding boxes as (N, T, 4) with [x, y, w, h]."""

    def load_store(self, rel_path: str | Path, mmap: bool = True) -> np.ndarray:
        path = self.resolve(rel_path)
        if path.suffix.lower() != ".npy":
            raise LoaderError(f"BBox store only supports .npy; got {path.suffix} for {path}")
        if mmap:
            arr = load_npy(path, mmap_mode="r")
        else:
            arr = load_npy(path)

        arr = np.asarray(arr)
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise LoaderError(
                "BBox store expected shape (N, T, 4); "
                f"got shape={arr.shape} from {path}"
            )
        
        if mmap:
            return arr  
        return np.ascontiguousarray(arr)
    
    def get_sample(self, store: np.ndarray, idx: int) -> np.ndarray:
        """Return a single sample as (T, 4) with [x, y, w, h]."""
        if store.ndim != 3 or store.shape[2] != 4:
            raise LoaderError(
                "BBox store expected shape (N, T, 4); " f"got shape={store.shape}"
            )
        if idx < 0 or idx >= store.shape[0]:
            raise LoaderError(
                f"BBox store index out of range: idx={idx}, N={store.shape[0]}"
            )
        return store[idx]
