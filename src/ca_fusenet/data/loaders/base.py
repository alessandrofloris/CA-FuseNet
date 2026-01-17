'''
Docstring for ca_fusenet.data.loaders.base

Defines the foundation of the entire CA-FuseNet data loading system.
'''
from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any
import pickle

import numpy as np


class LoaderError(RuntimeError):
    """Raised when loading an offline artifact fails."""

    def __init__(self, msg: str) -> None:
        super().__init__(f"Loader error: {msg}")


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise LoaderError(f"File not found: {path}")
    if not path.is_file():
        raise LoaderError(f"Path is not a file: {path}")


def load_npy(path: Path) -> np.ndarray:
    ensure_file(path)
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        raise LoaderError(
            f"Failed to load npy: {path} ({type(exc).__name__}: {exc})"
        ) from exc


def load_pkl(path: Path) -> Any:
    ensure_file(path)
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        raise LoaderError(
            f"Failed to load pkl: {path} ({type(exc).__name__}: {exc})"
        ) from exc


class BaseLoader(ABC):
    '''
        Defines the rules for all other loaders
    '''
    def __init__(self, root: str | Path) -> None:
        self.root: Path = Path(root).expanduser()

    def resolve(self, rel_path: str | Path) -> Path:
        rel = Path(rel_path)
        if rel.is_absolute():
            return rel
        return self.root / rel
