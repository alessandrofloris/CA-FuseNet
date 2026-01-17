"""
Docstring for ca_fusenet.data.contracts

Data contracts for CA-FuseNet artifacts.
"""

from dataclasses import dataclass
from typing import List, Optional, Literal

import numpy as np

_EPS = 1e-6


class ContractError(ValueError):
    """Raised when a data contract is violated."""

    def __init__(self, msg: str) -> None:
        super().__init__(f"Data contract error: {msg}")


def ensure(condition: bool, msg: str) -> None:
    if not condition:
        raise ContractError(msg)


def is_finite_numpy(arr: np.ndarray) -> bool:
    try:
        return bool(np.all(np.isfinite(arr)))
    except TypeError:
        return False


def _shape_str(arr: object) -> str:
    if isinstance(arr, np.ndarray):
        return f"ndim={arr.ndim} shape={tuple(arr.shape)}"
    return f"type={type(arr)}"


def _ensure_ndarray(arr: object, label: str) -> np.ndarray:
    ensure(isinstance(arr, np.ndarray), f"{label} expected numpy.ndarray; got {type(arr)}")
    return arr


def _ensure_float_dtype(arr: np.ndarray, label: str) -> None:
    ensure(
        arr.dtype in (np.float32, np.float64),
        f"{label} expected dtype float32 or float64; got {arr.dtype}",
    )


def _range_check(arr: np.ndarray, low: float, high: float, eps: float, label: str) -> None:
    min_val = float(arr.min())
    max_val = float(arr.max())
    ensure(
        min_val >= low - eps and max_val <= high + eps,
        f"{label} expected values in [{low}, {high}] (eps={eps}); got min={min_val}, max={max_val}",
    )


@dataclass
class PoseSequenceContract:
    """Pose joint sequence for one person: (T, J, 3) with (x, y, v)."""

    joint: np.ndarray
    coord_space: Literal["normalized", "pixel"] = "normalized"

    @classmethod
    def from_array(
        cls,
        joint: np.ndarray,
        coord_space: Literal["normalized", "pixel"] = "normalized",
    ) -> "PoseSequenceContract":
        arr = np.asarray(joint)
        _ensure_float_dtype(arr, "PoseSequenceContract.joint")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        obj = cls(arr, coord_space=coord_space)
        obj.validate()
        return obj

    def validate(self, expected_T: int | None = None) -> None:
        arr = _ensure_ndarray(self.joint, "PoseSequenceContract.joint")
        ensure(
            arr.ndim == 3,
            "PoseSequenceContract.joint expected shape (T, J, 3); got "
            + _shape_str(arr),
        )
        T, J, C = arr.shape
        ensure(T >= 1, f"PoseSequenceContract.joint expected T>=1; got T={T}")
        ensure(J >= 1, f"PoseSequenceContract.joint expected J>=1; got J={J}")
        ensure(
            C == 3,
            f"PoseSequenceContract.joint expected last dim C==3; got C={C} shape={arr.shape}",
        )
        if expected_T is not None:
            ensure(
                T == expected_T,
                f"PoseSequenceContract.joint expected T=={expected_T}; got T={T}",
            )
        _ensure_float_dtype(arr, "PoseSequenceContract.joint")
        ensure(is_finite_numpy(arr), "PoseSequenceContract.joint contains NaN or inf")
        ensure(
            self.coord_space in ("normalized", "pixel"),
            f"PoseSequenceContract.coord_space must be 'normalized' or 'pixel'; got {self.coord_space}",
        )
        if self.coord_space == "normalized":
            xy = arr[..., :2]
            _range_check(xy, 0.0, 1.0, _EPS, "PoseSequenceContract.joint[x,y]")
        v = arr[..., 2]
        _range_check(v, 0.0, 1.0, _EPS, "PoseSequenceContract.joint[v]")


@dataclass
class BBoxSequenceContract:
    """Bounding boxes aligned to T: (T, 4) with [x1, y1, w, h]."""

    bboxes: np.ndarray
    coord_space: Literal["normalized", "pixel"] = "normalized"

    @classmethod
    def from_array(
        cls,
        bboxes: np.ndarray,
        coord_space: Literal["normalized", "pixel"] = "normalized",
    ) -> "BBoxSequenceContract":
        arr = np.asarray(bboxes)
        _ensure_float_dtype(arr, "BBoxSequenceContract.bboxes")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        obj = cls(arr, coord_space=coord_space)
        obj.validate()
        return obj

    def validate(self, expected_T: int | None = None) -> None:
        arr = _ensure_ndarray(self.bboxes, "BBoxSequenceContract.bboxes")
        ensure(
            arr.ndim == 2,
            "BBoxSequenceContract.bboxes expected shape (T, 4); got " + _shape_str(arr),
        )
        T, C = arr.shape
        ensure(T >= 1, f"BBoxSequenceContract.bboxes expected T>=1; got T={T}")
        ensure(
            C == 4,
            f"BBoxSequenceContract.bboxes expected last dim 4; got C={C} shape={arr.shape}",
        )
        if expected_T is not None:
            ensure(
                T == expected_T,
                f"BBoxSequenceContract.bboxes expected T=={expected_T}; got T={T}",
            )
        _ensure_float_dtype(arr, "BBoxSequenceContract.bboxes")
        ensure(is_finite_numpy(arr), "BBoxSequenceContract.bboxes contains NaN or inf")
        ensure(
            self.coord_space in ("normalized", "pixel"),
            f"BBoxSequenceContract.coord_space must be 'normalized' or 'pixel'; got {self.coord_space}",
        )
        if self.coord_space == "normalized":
            _range_check(arr, 0.0, 1.0, _EPS, "BBoxSequenceContract.bboxes")


@dataclass
class IndicatorsContract:
    """Crowd/context indicators aligned to T or clip-level: (T, K) or (K,)."""

    values: np.ndarray
    names: Optional[List[str]] = None

    @classmethod
    def from_array(
        cls, values: np.ndarray, names: Optional[List[str]] = None
    ) -> "IndicatorsContract":
        arr = np.asarray(values)
        _ensure_float_dtype(arr, "IndicatorsContract.values")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        obj = cls(arr, names=names)
        obj.validate()
        return obj

    def validate(self, expected_T: int | None = None) -> None:
        arr = _ensure_ndarray(self.values, "IndicatorsContract.values")
        ensure(
            arr.ndim in (1, 2),
            "IndicatorsContract.values expected shape (K,) or (T, K); got "
            + _shape_str(arr),
        )
        if arr.ndim == 1:
            K = arr.shape[0]
            ensure(K >= 1, f"IndicatorsContract.values expected K>=1; got K={K}")
        else:
            T, K = arr.shape
            ensure(T >= 1, f"IndicatorsContract.values expected T>=1; got T={T}")
            ensure(K >= 1, f"IndicatorsContract.values expected K>=1; got K={K}")
            if expected_T is not None:
                ensure(
                    T == expected_T,
                    f"IndicatorsContract.values expected T=={expected_T}; got T={T}",
                )
        _ensure_float_dtype(arr, "IndicatorsContract.values")
        ensure(is_finite_numpy(arr), "IndicatorsContract.values contains NaN or inf")
        if self.names is not None:
            ensure(
                len(self.names) == K,
                f"IndicatorsContract.names expected length {K}; got {len(self.names)}",
            )
            ensure(
                all(isinstance(n, str) for n in self.names),
                "IndicatorsContract.names must be strings",
            )


@dataclass
class TubeletContract:
    """RGB tubelet frames: (T, 3, H, W) with uint8 or float32 values."""

    frames: np.ndarray

    @classmethod
    def from_array(cls, frames: np.ndarray) -> "TubeletContract":
        arr = np.asarray(frames)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        obj = cls(arr)
        obj.validate()
        return obj

    def validate(self, expected_T: int | None = None) -> None:
        arr = _ensure_ndarray(self.frames, "TubeletContract.frames")
        ensure(
            arr.ndim == 4,
            "TubeletContract.frames expected shape (T, 3, H, W); got " + _shape_str(arr),
        )
        T, C, H, W = arr.shape
        ensure(T >= 1, f"TubeletContract.frames expected T>=1; got T={T}")
        ensure(C == 3, f"TubeletContract.frames expected C==3; got C={C}")
        ensure(H >= 1 and W >= 1, f"TubeletContract.frames expected H,W>=1; got H={H}, W={W}")
        if expected_T is not None:
            ensure(
                T == expected_T,
                f"TubeletContract.frames expected T=={expected_T}; got T={T}",
            )
        ensure(
            arr.dtype in (np.uint8, np.float32),
            f"TubeletContract.frames expected dtype uint8 or float32; got {arr.dtype}",
        )
        if arr.dtype == np.uint8:
            min_val = int(arr.min())
            max_val = int(arr.max())
            ensure(
                min_val >= 0 and max_val <= 255,
                f"TubeletContract.frames expected uint8 in [0, 255]; got min={min_val}, max={max_val}",
            )
        else:
            ensure(is_finite_numpy(arr), "TubeletContract.frames contains NaN or inf")
            _range_check(arr, 0.0, 1.0, _EPS, "TubeletContract.frames")
