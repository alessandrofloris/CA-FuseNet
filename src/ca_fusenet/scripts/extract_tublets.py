from __future__ import annotations

from pathlib import Path
import sys
import json
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path


def _ensure_repo_src_on_path() -> None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[3]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _unique_frame_indices(frames: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    indices: list[int] = []
    for idx, frame in enumerate(frames):
        frame_id = int(frame)
        if frame_id not in seen:
            seen.add(frame_id)
            indices.append(idx)
    return indices


def _select_frame_indices(frames: list[int], target_len: int) -> list[int]:
    if not frames:
        return [0] * target_len

    unique_indices = _unique_frame_indices(frames)
    if not unique_indices:
        return [0] * target_len

    unique_frames = [frames[i] for i in unique_indices]

    # Prefer a contiguous window if possible.
    best_start = 0
    best_len = 1
    cur_start = 0
    cur_len = 1
    for i in range(1, len(unique_frames)):
        if unique_frames[i] == unique_frames[i - 1] + 1:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
            cur_start = i
            cur_len = 1
    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start

    if best_len >= target_len:
        selected = unique_indices[best_start : best_start + target_len]
    else:
        selected = unique_indices[:target_len]

    if not selected:
        selected = [0]
    while len(selected) < target_len:
        selected.append(selected[-1])
    return selected


def _frame_index_base(frames: list[int]) -> int:
    if not frames:
        return 0
    return 1 if min(frames) >= 1 else 0


def _load_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _crop_and_resize(
    frame: np.ndarray,
    bbox_xywh: np.ndarray,
    out_h: int,
    out_w: int,
    coord_space: str,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, w, h = (float(v) for v in bbox_xywh)
    if coord_space == "normalized":
        x *= width
        y *= height
        w *= width
        h *= height
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _estimate_sizes(num_samples: int, tubelet_len: int, out_h: int, out_w: int) -> dict[str, float]:
    bytes_per_sample = 3 * tubelet_len * out_h * out_w
    total_bytes = num_samples * bytes_per_sample
    return {
        "bytes_per_sample": bytes_per_sample,
        "total_gb": total_bytes / (1024**3),
        "batch_gb": bytes_per_sample / (1024**3),
    }


def _get_rss_gb() -> float | None:
    try:
        import psutil
    except Exception:
        return None
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def _load_state(state_path: Path) -> dict[str, object]:
    if not state_path.exists():
        return {"next_index": 0, "errors": []}
    try:
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"next_index": 0, "errors": []}


def _save_state(state_path: Path, next_index: int, errors: list[dict[str, object]]) -> None:
    payload = {"next_index": next_index, "errors": errors}
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _next_unprocessed(done: np.ndarray) -> int:
    missing = np.where(done == 0)[0]
    if missing.size == 0:
        return done.shape[0]
    return int(missing[0])


def _process_one_sample(
    *,
    idx: int,
    video_path: Path,
    frames: list[int],
    bbox_seq: np.ndarray,
    tubelet_len: int,
    out_h: int,
    out_w: int,
    coord_space: str,
) -> tuple[int, np.ndarray | None, str | None]:
    try:
        if not video_path.exists():
            return idx, None, f"Video not found: {video_path}"

        if len(frames) != bbox_seq.shape[0]:
            return (
                idx,
                None,
                f"Frame/bbox length mismatch: frames={len(frames)} bbox={bbox_seq.shape[0]}",
            )

        frame_base = _frame_index_base(frames)
        selected_indices = _select_frame_indices(frames, tubelet_len)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return idx, None, f"Failed to open video: {video_path}"

        tubelet = np.zeros((3, tubelet_len, out_h, out_w), dtype=np.uint8)
        last_frame = None
        for t, frame_idx in enumerate(selected_indices):
            raw_frame_id = int(frames[frame_idx]) - frame_base
            raw_frame_id = max(raw_frame_id, 0)
            frame = _load_frame(cap, raw_frame_id)
            if frame is None:
                frame = last_frame
            if frame is None:
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or out_h
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or out_w
                frame = np.zeros((height, width, 3), dtype=np.uint8)

            bbox_xywh = bbox_seq[frame_idx]
            crop = _crop_and_resize(frame, bbox_xywh, out_h, out_w, coord_space)
            tubelet[:, t] = np.transpose(crop, (2, 0, 1))
            last_frame = frame

        cap.release()
        return idx, tubelet, None
    except Exception as exc:  # pragma: no cover - runtime safety
        return idx, None, f"{type(exc).__name__}: {exc}"


@hydra.main(version_base=None, config_path="../configs", config_name="preprocessing/tublets")
def main(cfg: DictConfig) -> None:
    _ensure_repo_src_on_path()

    from ca_fusenet.data.loaders import BBoxLoader, LabelLoader
    from ca_fusenet.data.loaders.base import load_npy
    from ca_fusenet.data.contracts import BBoxStoreContract

    print(OmegaConf.to_yaml(cfg))
    
    cfg = OmegaConf.to_container(cfg, resolve=True)

    data_root = Path(to_absolute_path(cfg["preprocessing"]["data_root"])).expanduser()
    video_root = Path(to_absolute_path(cfg["preprocessing"]["video_root"])).expanduser()
    label_store_path = cfg["preprocessing"]["label_store"]
    bbox_store_path = cfg["preprocessing"]["bbox_store"]
    output_store_path = cfg["preprocessing"]["output_store"]
    tubelet_len = int(cfg["preprocessing"]["tubelet_length"])
    out_h = int(cfg["preprocessing"]["output_height"])
    out_w = int(cfg["preprocessing"]["output_width"])
    coord_space = str(cfg.get("bbox_coord_space", "pixel")).lower()
    batch_size = max(1, int(cfg.get("batch_size", 8)))
    resume = bool(cfg.get("resume", False))
    num_workers = int(cfg.get("num_workers", 0))
    parallel = bool(cfg.get("parallel", False))
    if num_workers <= 0:
        parallel = False

    label_loader = LabelLoader(data_root)
    labels_store = label_loader.load_store(label_store_path)

    bbox_loader = BBoxLoader(data_root, coord_space)
    bbox_store = bbox_loader.load_store(bbox_store_path, mmap=True)


    bboxes = bbox_store.bboxes
    num_samples = len(labels_store.sample_names)
    if bboxes.shape[0] != num_samples:
        raise ValueError(
            f"N mismatch between labels ({num_samples}) and bboxes ({bboxes.shape[0]})"
        )

    output_path = Path(output_store_path)
    if not output_path.is_absolute():
        output_path = data_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shape = (num_samples, 3, tubelet_len, out_h, out_w)
    estimates = _estimate_sizes(num_samples, tubelet_len, out_h, out_w)
    print(
        "Tubelet store shape "
        f"{shape} (~{estimates['total_gb']:.2f} GB total, "
        f"~{estimates['batch_gb'] * batch_size:.2f} GB per batch)"
    )

    state_path = output_path.with_suffix(output_path.suffix + ".state.json")
    done_path = output_path.with_suffix(output_path.suffix + ".done.npy")

    if output_path.exists():
        if resume:
            tubelets = np.lib.format.open_memmap(output_path, mode="r+")
        else:
            tubelets = np.lib.format.open_memmap(
                output_path, mode="w+", dtype=np.uint8, shape=shape
            )
    else:
        tubelets = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.uint8, shape=shape
        )
    if tubelets.shape != shape:
        raise ValueError(
            f"Existing output shape {tubelets.shape} does not match expected {shape}."
        )

    if done_path.exists() and resume:
        done = np.lib.format.open_memmap(done_path, mode="r+")
        if done.shape != (num_samples,):
            done = np.lib.format.open_memmap(
                done_path, mode="w+", dtype=np.uint8, shape=(num_samples,)
            )
    else:
        done = np.lib.format.open_memmap(
            done_path, mode="w+", dtype=np.uint8, shape=(num_samples,)
        )

    if resume:
        state = _load_state(state_path)
        next_index = int(state.get("next_index", 0))
        if next_index == 0:
            next_index = _next_unprocessed(done)
        errors: list[dict[str, object]] = list(state.get("errors", []))
    else:
        next_index = 0
        errors = []
        done[:] = 0
        done.flush()

    if next_index >= num_samples:
        print("All samples already processed.")
        return

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None  # type: ignore

    ranges = range(next_index, num_samples, batch_size)
    iterator = tqdm(ranges, desc="Extracting tubelets", unit="batch") if tqdm else ranges

    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, num_samples)
        batch_indices = list(range(batch_start, batch_end))
        tasks = []
        for idx in batch_indices:
            if done[idx] != 0:
                continue
            record = label_loader.get_sample(labels_store, idx)
            video_rel = record.video_path
            path = Path(video_rel)
            video_rel = str(path.parent / f"b_{path.name}")
            
            if video_rel is None:
                errors.append({"idx": idx, "error": "Missing video_path"})
                done[idx] = 1
                tubelets[idx] = 0
                continue
            frames = list(record.frame)
            bbox_seq = bboxes[idx]
            tasks.append(
                {
                    "idx": idx,
                    "video_path": video_root / video_rel,
                    "frames": frames,
                    "bbox_seq": bbox_seq,
                }
            )

        if parallel and tasks:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_map = {
                    executor.submit(
                        _process_one_sample,
                        idx=task["idx"],
                        video_path=task["video_path"],
                        frames=task["frames"],
                        bbox_seq=task["bbox_seq"],
                        tubelet_len=tubelet_len,
                        out_h=out_h,
                        out_w=out_w,
                        coord_space=coord_space,
                    ): task["idx"]
                    for task in tasks
                }
                for future in as_completed(future_map):
                    idx, tubelet, error = future.result()
                    if error:
                        errors.append({"idx": idx, "error": error})
                        tubelets[idx] = 0
                    else:
                        tubelets[idx] = tubelet
                    done[idx] = 1
        else:
            for task in tasks:
                idx, tubelet, error = _process_one_sample(
                    idx=task["idx"],
                    video_path=task["video_path"],
                    frames=task["frames"],
                    bbox_seq=task["bbox_seq"],
                    tubelet_len=tubelet_len,
                    out_h=out_h,
                    out_w=out_w,
                    coord_space=coord_space,
                )
                if error:
                    errors.append({"idx": idx, "error": error})
                    tubelets[idx] = 0
                else:
                    tubelets[idx] = tubelet
                done[idx] = 1

        tubelets.flush()
        done.flush()
        _save_state(state_path, _next_unprocessed(done), errors)

        if tqdm and hasattr(iterator, "set_postfix"):
            rss_gb = _get_rss_gb()
            if rss_gb is not None:
                iterator.set_postfix({"rss_gb": f"{rss_gb:.2f}"})

    print(f"Saved tubelets to {output_path} with shape {tubelets.shape}")


if __name__ == "__main__":
    main()
