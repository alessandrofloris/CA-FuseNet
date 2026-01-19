"""Data contracts for CA-FuseNet artifacts."""

from dataclasses import dataclass
from typing import Any, Literal, Optional

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


def _min_check(arr: np.ndarray, min_value: float, eps: float, label: str) -> None:
    min_val = float(arr.min())
    ensure(
        min_val >= min_value - eps,
        f"{label} expected values >= {min_value} (eps={eps}); got min={min_val}",
    )


def _first_non_str(values: list[object]) -> tuple[int, object] | None:
    for idx, value in enumerate(values):
        if not isinstance(value, str):
            return idx, value
    return None


def _first_non_int(values: list[object]) -> tuple[int, object] | None:
    for idx, value in enumerate(values):
        if not isinstance(value, (int, np.integer)):
            return idx, value
    return None


@dataclass
class PoseSequenceContract:
    """Per-sample pose sequence: (T, V, 3) with (x, y, score)."""

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
            "PoseSequenceContract.joint expected shape (T, V, 3); got "
            + _shape_str(arr),
        )
        T, V, C = arr.shape
        ensure(T >= 1, f"PoseSequenceContract.joint expected T>=1; got T={T}")
        ensure(V == 17, f"PoseSequenceContract.joint expected V==17; got V={V}")
        ensure(
            C == 3,
            f"PoseSequenceContract.joint expected last dim C==3; got C={C} shape={arr.shape}",
        )
        if expected_T is not None:
            ensure(
                T == expected_T,
                f"PoseSequenceContract.joint expected T=={expected_T}; got T={T}",
            )
        ensure(is_finite_numpy(arr), "PoseSequenceContract.joint contains NaN or inf")
        ensure(
            self.coord_space in ("normalized", "pixel"),
            f"PoseSequenceContract.coord_space must be 'normalized' or 'pixel'; got {self.coord_space}",
        )

        if self.coord_space == "pixel":
            x = arr[..., 0]
            y = arr[..., 1]
            _range_check(x, 0.0, 1920.0, _EPS, "PoseSequenceContract.joint[x]")
            _range_check(y, 0.0, 1080.0, _EPS, "PoseSequenceContract.joint[y]")
        else:
            xy = arr[..., :2]
            _range_check(xy, 0.0, 1.0, _EPS, "PoseSequenceContract.joint[x,y]")
        score = arr[..., 2]
        _range_check(score, 0.0, 1.0, _EPS, "PoseSequenceContract.joint[score]")


@dataclass
class BBoxSequenceContract:
    """Per-sample bounding boxes: (T, 4) with [x, y, w, h]."""

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
        ensure(is_finite_numpy(arr), "BBoxSequenceContract.bboxes contains NaN or inf")
        ensure(
            self.coord_space in ("normalized", "pixel"),
            f"BBoxSequenceContract.coord_space must be 'normalized' or 'pixel'; got {self.coord_space}",
        )

        x = arr[..., 0]
        y = arr[..., 1]
        w = arr[..., 2]
        h = arr[..., 3]
        
        if self.coord_space == "pixel":
            _min_check(w, 0.0, _EPS, "BBoxSequenceContract.bboxes[w]")
            _min_check(h, 0.0, _EPS, "BBoxSequenceContract.bboxes[h]")
            _range_check(x+w, 0.0, 1920.0, _EPS, "BBoxSequenceContract.bboxes[x+w]")
            _range_check(y+h, 0.0, 1080.0, _EPS, "BBoxSequenceContract.bboxes[y+h]")
        else:
            _range_check(arr, 0.0, 1.0, _EPS, "BBoxSequenceContract.bboxes")
            # TODO: Needs more checks for normalized boxes


@dataclass
class IndicatorsContract:
    """Per-sample indicators: (T, 3)."""

    values: np.ndarray

    @classmethod
    def from_array(
        cls, values: np.ndarray) -> "IndicatorsContract":
        arr = np.asarray(values)
        _ensure_float_dtype(arr, "IndicatorsContract.values")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        obj = cls(arr)
        obj.validate()
        return obj

    def validate(self, expected_T: int | None = None) -> None:
        arr = _ensure_ndarray(self.values, "IndicatorsContract.values")
        ensure(
            arr.ndim == 2,
            "IndicatorsContract.values expected shape (T, 3); got "
            + _shape_str(arr),
        )
        T, K = arr.shape
        ensure(T >= 1, f"IndicatorsContract.values expected T>=1; got T={T}")
        ensure(
            K == 3,
            f"IndicatorsContract.values expected K==3; got K={K} shape={arr.shape}",
        )
        if expected_T is not None:
            ensure(
                T == expected_T,
                f"IndicatorsContract.values expected T=={expected_T}; got T={T}",
            )
        ensure(is_finite_numpy(arr), "IndicatorsContract.values contains NaN or inf")

        area = arr[:,0]; vis = arr[:,1]; motion = arr[:,2]
        _range_check(area, 0.0, 1.0, _EPS, "Indicators[area]")
        _range_check(vis, 0.0, 1.0, _EPS, "Indicators[visibility]")
        _min_check(motion, 0.0, _EPS, "Indicators[motion_proxy]")

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


@dataclass
class BBoxStoreContract:
    """Dataset-level bounding boxes: (N, T, 4) with [x, y, w, h] (XYWH)."""

    bboxes: np.ndarray
    coord_space: Literal["normalized", "pixel"] = "normalized"

    @classmethod
    def from_array(
        cls,
        bboxes: np.ndarray,
        coord_space: Literal["normalized", "pixel"] = "normalized",
    ) -> "BBoxStoreContract":
        arr = np.asarray(bboxes)
        _ensure_float_dtype(arr, "BBoxStoreContract.bboxes")
        #if arr.dtype != np.float32:
        #    arr = arr.astype(np.float32)
        obj = cls(arr, coord_space=coord_space)
        obj.validate()
        return obj

    def validate(self, expected_T: int | None = None) -> None:
        arr = _ensure_ndarray(self.bboxes, "BBoxStoreContract.bboxes")
        ensure(
            arr.ndim == 3,
            "BBoxStoreContract.bboxes expected shape (N, T, 4); got " + _shape_str(arr),
        )
        N, T, C = arr.shape
        ensure(N >= 1, f"BBoxStoreContract.bboxes expected N>=1; got N={N}")
        ensure(T >= 1, f"BBoxStoreContract.bboxes expected T>=1; got T={T}")
        ensure(
            C == 4,
            f"BBoxStoreContract.bboxes expected last dim 4; got C={C} shape={arr.shape}",
        )
        if expected_T is not None:
            ensure(
                T == expected_T,
                f"BBoxStoreContract.bboxes expected T=={expected_T}; got T={T}",
            )
        ensure(is_finite_numpy(arr), "BBoxStoreContract.bboxes contains NaN or inf")
        ensure(
            self.coord_space in ("normalized", "pixel"),
            f"BBoxStoreContract.coord_space must be 'normalized' or 'pixel'; got {self.coord_space}",
        )
        w = arr[..., 2]
        h = arr[..., 3]
        _min_check(w, 0.0, _EPS, "BBoxStoreContract.bboxes[w]")
        _min_check(h, 0.0, _EPS, "BBoxStoreContract.bboxes[h]")
        if self.coord_space == "normalized":
            _range_check(arr, 0.0, 1.0, _EPS, "BBoxStoreContract.bboxes")


@dataclass
class IndicatorsStoreContract:
    """Dataset-level indicators: (N, T, 3)."""

    values: np.ndarray
    names: Optional[list[str]] = None

    @classmethod
    def from_array(
        cls, values: np.ndarray, names: Optional[list[str]] = None
    ) -> "IndicatorsStoreContract":
        arr = np.asarray(values)
        _ensure_float_dtype(arr, "IndicatorsStoreContract.values")
        #if arr.dtype != np.float32:
        #    arr = arr.astype(np.float32)
        obj = cls(arr, names=names)
        obj.validate()
        return obj

    def validate(self, expected_T: int | None = None) -> None:
        arr = _ensure_ndarray(self.values, "IndicatorsStoreContract.values")
        ensure(
            arr.ndim == 3,
            "IndicatorsStoreContract.values expected shape (N, T, 3); got "
            + _shape_str(arr),
        )
        N, T, K = arr.shape
        ensure(N >= 1, f"IndicatorsStoreContract.values expected N>=1; got N={N}")
        ensure(T >= 1, f"IndicatorsStoreContract.values expected T>=1; got T={T}")
        ensure(
            K == 3,
            f"IndicatorsStoreContract.values expected K==3; got K={K} shape={arr.shape}",
        )
        if expected_T is not None:
            ensure(
                T == expected_T,
                f"IndicatorsStoreContract.values expected T=={expected_T}; got T={T}",
            )
        _ensure_float_dtype(arr, "IndicatorsStoreContract.values")
        ensure(is_finite_numpy(arr), "IndicatorsStoreContract.values contains NaN or inf")
        if self.names is not None:
            ensure(
                len(self.names) == 3,
                f"IndicatorsStoreContract.names expected length 3; got {len(self.names)}",
            )
            ensure(
                all(isinstance(n, str) for n in self.names),
                "IndicatorsStoreContract.names must be strings",
            )


@dataclass
class JointStoreContract:
    """Dataset-level joint store: (N, 3, T, V, M) with (x, y, score)."""

    joint: np.ndarray
    coord_space: Literal["normalized", "pixel"] = "normalized"

    @classmethod
    def from_array(
        cls,
        joint: np.ndarray,
        coord_space: Literal["normalized", "pixel"] = "normalized",
    ) -> "JointStoreContract":
        arr = np.asarray(joint)
        _ensure_float_dtype(arr, "JointStoreContract.joint")
        #if arr.dtype != np.float32:
        #    arr = arr.astype(np.float32)
        obj = cls(arr, coord_space=coord_space)
        obj.validate()
        return obj

    def validate(self, expected_T: int | None = None) -> None:
        arr = _ensure_ndarray(self.joint, "JointStoreContract.joint")
        ensure(
            arr.ndim == 5,
            "JointStoreContract.joint expected shape (N, 3, T, V, M); got "
            + _shape_str(arr),
        )
        N, C, T, V, M = arr.shape
        ensure(N >= 1, f"JointStoreContract.joint expected N>=1; got N={N}")
        ensure(T >= 1, f"JointStoreContract.joint expected T>=1; got T={T}")
        ensure(V >= 1, f"JointStoreContract.joint expected V>=1; got V={V}")
        ensure(M >= 1, f"JointStoreContract.joint expected M>=1; got M={M}")
        ensure(
            C == 3,
            f"JointStoreContract.joint expected C==3; got C={C} shape={arr.shape}",
        )
        if expected_T is not None:
            ensure(
                T == expected_T,
                f"JointStoreContract.joint expected T=={expected_T}; got T={T}",
            )
        _ensure_float_dtype(arr, "JointStoreContract.joint")
        ensure(is_finite_numpy(arr), "JointStoreContract.joint contains NaN or inf")
        ensure(
            self.coord_space in ("normalized", "pixel"),
            f"JointStoreContract.coord_space must be 'normalized' or 'pixel'; got {self.coord_space}",
        )
        if self.coord_space == "normalized":
            xy = arr[:, :2, :, :, :]
            _range_check(xy, 0.0, 1.0, _EPS, "JointStoreContract.joint[x,y]")
        score = arr[:, 2, :, :, :]
        _range_check(score, 0.0, 1.0, _EPS, "JointStoreContract.joint[score]")


@dataclass
class LabelStoreContract:
    """Dataset-level labels: tuple of lists (sample_name, video_path, label, frame)."""

    sample_names: list[str]
    video_paths: list[str]
    labels: list[Any]
    frames: list[int]

    @classmethod
    def from_tuple(cls, store: object) -> "LabelStoreContract":
        ensure(
            isinstance(store, (tuple, list)),
            f"LabelStoreContract expected tuple/list of 4 lists; got {type(store)}",
        )
        ensure(
            len(store) == 4,
            f"LabelStoreContract expected 4 elements; got len={len(store)}",
        )
        sample_names, video_paths, labels, frames = store
        ensure(
            isinstance(sample_names, (list, tuple)),
            f"LabelStoreContract.sample_names expected list/tuple; got {type(sample_names)}",
        )
        ensure(
            isinstance(video_paths, (list, tuple)),
            f"LabelStoreContract.video_paths expected list/tuple; got {type(video_paths)}",
        )
        ensure(
            isinstance(labels, (list, tuple)),
            f"LabelStoreContract.labels expected list/tuple; got {type(labels)}",
        )
        ensure(
            isinstance(frames, (list, tuple)),
            f"LabelStoreContract.frames expected list/tuple; got {type(frames)}",
        )
        obj = cls(
            list(sample_names),
            list(video_paths),
            list(labels),
            list(frames),
        )
        obj.validate()
        return obj

    def validate(self) -> None:
        N = len(self.sample_names)
        ensure(
            len(self.video_paths) == N,
            f"LabelStoreContract.video_paths expected length {N}; got {len(self.video_paths)}",
        )
        ensure(
            len(self.labels) == N,
            f"LabelStoreContract.labels expected length {N}; got {len(self.labels)}",
        )
        ensure(
            len(self.frames) == N,
            f"LabelStoreContract.frames expected length {N}; got {len(self.frames)}",
        )
        non_str = _first_non_str(list(self.sample_names))
        if non_str is not None:
            idx, value = non_str
            raise ContractError(
                f"LabelStoreContract.sample_names expected str at index {idx}; got {type(value)}"
            )
        non_path = _first_non_str(list(self.video_paths))
        if non_path is not None:
            idx, value = non_path
            raise ContractError(
                f"LabelStoreContract.video_paths expected str at index {idx}; got {type(value)}"
            )
        non_int = _first_non_int(list(self.frames))
        if non_int is not None:
            idx, value = non_int
            raise ContractError(
                f"LabelStoreContract.frames expected int at index {idx}; got {type(value)}"
            )
