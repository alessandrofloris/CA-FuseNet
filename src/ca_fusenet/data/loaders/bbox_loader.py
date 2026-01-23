from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseLoader, LoaderError, load_npy

from ..contracts import BBoxStoreContract


class BBoxLoader(BaseLoader):
    """Load dataset-level bounding boxes as (N, T, K) with K = [x, y, w, h]."""

    def __init__(self, root: Path, coord_space: str):
        super().__init__(root)
        self.coord_space = coord_space


    def load_store(self, rel_path: str | Path, mmap: bool = True) -> BBoxStoreContract:
        path = self.resolve(rel_path)

        if path.suffix.lower() != ".npy":
            raise LoaderError(f"BBox store only supports .npy; got {path.suffix} for {path}")
        
        if mmap:
            arr = load_npy(path, mmap_mode="r")
        else:
            arr = load_npy(path)
        
        return BBoxStoreContract.from_array(arr, coord_space=self.coord_space, mmap=mmap)

    def get_sample(self, store: BBoxStoreContract, idx: int) -> np.ndarray:
        """Return a single sample as (T, 4) with [x, y, w, h]."""
        
        arr = store.bboxes
        
        return arr[idx]