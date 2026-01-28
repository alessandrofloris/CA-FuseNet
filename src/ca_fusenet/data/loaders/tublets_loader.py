from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseLoader, LoaderError, load_npy

from ..contracts import TubletsStoreContract


class TubletsLoader(BaseLoader):
    """Load tublets as (N, C, T, H, W)."""

    def __init__(self, root: Path):
        super().__init__(root)

    def load_store(self, rel_path: str | Path, mmap: bool = True) -> TubletsStoreContract:
        path = self.resolve(rel_path)
        
        if path.suffix.lower() != ".npy":
            raise LoaderError(
                f"Tublets store only supports .npy; got {path.suffix} for {path}"
            )
    
        if mmap:
            arr = load_npy(path, mmap_mode="r")
        else:
            arr = load_npy(path)

        return TubletsStoreContract.from_array(arr, mmap=mmap)

    def get_sample(
        self, store: TubletsStoreContract, idx: int
    ) -> np.ndarray:
        """Return one tublet as (C, T, H, W)"""

        arr = store.tublets

        sample = arr[idx, :, :, :, :]
        return sample.astype(np.float32) / 255.0 # uint8 to float32 normalization
