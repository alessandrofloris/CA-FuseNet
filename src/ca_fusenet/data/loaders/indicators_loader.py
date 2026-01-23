from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseLoader, LoaderError, load_npy

from ..contracts import IndicatorsStoreContract

class IndicatorsLoader(BaseLoader):
    """Load dataset-level crowd indicators as (N, T, 3)."""

    def load_store(
        self, rel_path: str | Path, mmap: bool = True) -> IndicatorsStoreContract:
        path = self.resolve(rel_path)
        
        if path.suffix.lower() != ".npy":
            raise LoaderError(f"Indicators store only supports .npy; got {path.suffix} for {path}")
        
        if mmap:
            values = load_npy(path, mmap_mode="r")
        else:
            values = load_npy(path)

        return IndicatorsStoreContract.from_array(values, mmap=mmap)        

    def get_sample(self, store: IndicatorsStoreContract, idx: int) -> np.ndarray:
        """Return indicators for a single sample as (T, C)."""
        
        arr = store.values
        
        return arr[idx]