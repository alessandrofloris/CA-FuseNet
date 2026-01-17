from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseLoader, LoaderError, load_npy


def _ensure_first_dim_is_N(arr: np.ndarray, path: Path) -> None:
    if arr.ndim != 3:
        raise LoaderError(
            "Indicators store expected shape (N, T, K) or (N, K); "
            f"got shape={arr.shape} from {path}"
        )
    if arr.shape[0] < 1:
        raise LoaderError(
            f"Indicators store expected N>=1; got N={arr.shape[0]} from {path}"
        )


def _validate_names(names: list[str] | None, K: int) -> None:
    if names is None:
        return
    if len(names) != K:
        raise LoaderError(
            f"Indicator names expected length {K}; got length {len(names)}"
        )
    if not all(isinstance(name, str) for name in names):
        raise LoaderError("Indicator names must be strings")


class IndicatorsLoader(BaseLoader):
    """Load dataset-level crowd indicators as (N, T, K)."""

    def load_store(
        self, rel_path: str | Path, mmap: bool = True
    ) -> tuple[np.ndarray, list[str] | None]:
        path = self.resolve(rel_path)
        if path.suffix.lower() != ".npy":
            raise LoaderError(
                f"Indicators store only supports .npy; got {path.suffix} for {path}"
            )
        if mmap:
            try:
                values: Any = np.load(path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                raise LoaderError(
                    f"Failed to load indicators store: {path} ({type(exc).__name__}: {exc})"
                ) from exc
        else:
            values = load_npy(path)

        arr = np.asarray(values)
        _ensure_first_dim_is_N(arr, path)

        if arr.ndim == 3:
            if arr.shape[2] < 1:
                raise LoaderError(
                    f"Indicators store expected K>=1; got K={arr.shape[2]} from {path}"
                )
        else:
            if arr.shape[1] < 1 or arr.shape[2] < 1:
                raise LoaderError(
                    f"Indicators store expected T,K>=1; got T={arr.shape[1]}, K={arr.shape[2]} from {path}"
                )

        names: list[str] | None = None
        _validate_names(names, arr.shape[2])
        return arr, names

    def get_sample(self, store: np.ndarray, idx: int) -> np.ndarray:
        """Return indicators for a single sample as (T, K)."""
        if store.ndim != 3:
            raise LoaderError(
                "Indicators store expected shape (N, T, K); "
                f"got shape={store.shape}"
            )
        return store[idx]
