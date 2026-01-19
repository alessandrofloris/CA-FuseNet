from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseLoader, LoaderError, load_npy


class JointLoader(BaseLoader):
    """Load dataset-level joints as (N, 3, T, V, M) with (x, y, score)."""

    def load_store(self, rel_path: str | Path, mmap: bool = True) -> np.ndarray:
        path = self.resolve(rel_path)
        if path.suffix.lower() != ".npy":
            raise LoaderError(
                f"Joint store only supports .npy; got {path.suffix} for {path}"
            )
        if mmap:
            arr = load_npy(path, mmap_mode="r")
        else:
            arr = load_npy(path)

        arr = np.asarray(arr)
        if arr.ndim != 5:
            raise LoaderError(
                "Joint store expected shape (N, 3, T, V, M) or known permutations; "
                f"got shape={arr.shape} from {path}"
            )
        
        if mmap:
            return arr
        return np.ascontiguousarray(arr)

    def get_sample(
        self, store: np.ndarray, idx: int, person_index: int = 0
    ) -> np.ndarray:
        """Return one person's pose as (T, V, 3) from the dataset store."""
        if store.ndim != 5 or store.shape[1] != 3:
            raise LoaderError(
                "Joint store expected shape (N, 3, T, V, M); "
                f"got shape={store.shape}"
            )
        if idx < 0 or idx >= store.shape[0]:
            raise LoaderError(
                f"Joint store index out of range: idx={idx}, N={store.shape[0]}"
            )
        if person_index < 0 or person_index >= store.shape[4]:
            raise LoaderError(
                "Joint store person_index out of range: "
                f"person_index={person_index}, M={store.shape[4]}"
            )
        sample = store[idx, :, :, :, 0]
        return np.transpose(sample, (1, 2, 0))
