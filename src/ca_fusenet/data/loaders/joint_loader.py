from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseLoader, LoaderError, load_npy

from ..contracts import JointStoreContract


class JointLoader(BaseLoader):
    """Load dataset-level joints as (N, C, T, V, M) with C = (x, y, score)."""

    def __init__(self, root: Path, coord_space: str):
        super().__init__(root)
        self.coord_space = coord_space

    def load_store(self, rel_path: str | Path, mmap: bool = True) -> JointStoreContract:
        path = self.resolve(rel_path)
        
        if path.suffix.lower() != ".npy":
            raise LoaderError(
                f"Joint store only supports .npy; got {path.suffix} for {path}"
            )
    
        if mmap:
            arr = load_npy(path, mmap_mode="r")
        else:
            arr = load_npy(path)

        return JointStoreContract.from_array(arr, coord_space=self.coord_space, mmap=mmap)

    def get_sample(
        self, store: JointStoreContract, idx: int
    ) -> np.ndarray:
        """Return one person's pose as (C, T, V) from the dataset store."""

        arr = store.joint

        sample = arr[idx, :, :, :, 0]
        return sample
