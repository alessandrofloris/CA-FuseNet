from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import BaseLoader, LoaderError, load_npy


class JointLoader(BaseLoader):
    """Load joint sequences shaped as (N, C, T, J, M) with (x, y, v)."""

    def load(self, rel_path: str | Path) -> np.ndarray:
        path = self.resolve(rel_path)
        arr = load_npy(path)
        if arr.ndim != 5:
            raise LoaderError(
                "Joint data expected 5D array with shape (N, C, T, J, M); "
                f"got shape={arr.shape}"
            )
        '''
        if arr.shape[2] == 3:
            return arr
        if arr.shape[1] == 3:
            return np.transpose(arr, (0, 2, 1))
        if arr.shape[0] == 3:
            return np.transpose(arr, (1, 2, 0))
        raise LoaderError(
            "Joint data expected shape (T, J, 3), (T, 3, J), or (3, T, J); "
            f"got shape={arr.shape}"
        )
        '''
