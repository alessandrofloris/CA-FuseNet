from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseLoader, LoaderError, load_npy


class IndicatorsLoader(BaseLoader):
    """Load dataset-level crowd indicators as (N, T, 3)."""

    def load_store(
        self, rel_path: str | Path, mmap: bool = True) -> np.ndarray:
        path = self.resolve(rel_path)
        if path.suffix.lower() != ".npy":
            raise LoaderError(f"Indicators store only supports .npy; got {path.suffix} for {path}")
        if mmap:
            values = load_npy(path, mmap_mode="r")
        else:
            values = load_npy(path)

        arr = np.asarray(values)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise LoaderError(
                "Indicators store expected shape (N, T, 3); "
                f"got shape={arr.shape} from {path}"
            )
        
        if mmap:
            return arr
        return np.ascontiguousarray(arr)

    def get_sample(self, store: np.ndarray, idx: int) -> np.ndarray:
        """Return indicators for a single sample as (T, 3)."""
        if store.ndim != 3 or store.shape[2] != 3:
            raise LoaderError(
                "Indicators store expected shape (N, T, 3); "
                f"got shape={store.shape}"
            )
        if idx < 0 or idx >= store.shape[0]:
            raise LoaderError(
                f"Indicators store index out of range: idx={idx}, N={store.shape[0]}"
            )
        return store[idx]
