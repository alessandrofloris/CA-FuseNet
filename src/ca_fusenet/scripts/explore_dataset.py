from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

# Make local package imports work when running this file directly from project root.
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ca_fusenet.utils.class_mapping import ITWPolimiClassMapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore a pickled dataset and count samples per action class."
    )
    parser.add_argument("pkl_path", type=Path, help="Path to input .pkl file")
    parser.add_argument(
        "--sort",
        choices=("id", "name"),
        default="id",
        help="Sort output by class id or class name (default: id)",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except Exception as pickle_error:
            try:
                import pandas as pd  # type: ignore
            except ImportError:
                raise RuntimeError(
                    "Failed to load pickle with standard pickle module. "
                    "Install pandas for fallback loading."
                ) from pickle_error

    try:
        return pd.read_pickle(path)
    except Exception as pandas_error:
        raise RuntimeError(
            f"Failed to load '{path}' as pickle (pickle and pandas fallback both failed)."
        ) from pandas_error


def iter_samples(data: Any) -> Iterable[Any]:
    return data[1]


def main() -> int:
    args = parse_args()

    if not args.pkl_path.exists():
        print(f"Error: file not found: {args.pkl_path}", file=sys.stderr)
        return 1

    data = load_pickle(args.pkl_path)
    samples = iter_samples(data)

    mapping = ITWPolimiClassMapping().get_mapping()
    counts: Counter[int] = Counter()
    total_samples = 0
    unknown_ids_warned: set[int] = set()
    
    for id_action in samples:
        class_id = id_action
        total_samples += 1
        counts[class_id] += 1

        if class_id not in mapping and class_id not in unknown_ids_warned:
            warnings.warn(
                f"id_action={class_id} has no class-mapping entry; shown as unknown.",
                RuntimeWarning,
            )
            unknown_ids_warned.add(class_id)
    
    if args.sort == "name":
        ordered = sorted(
            counts.items(),
            key=lambda kv: mapping.get(kv[0], f"unknown (id={kv[0]})").lower(),
        )
    else:
        ordered = sorted(counts.items(), key=lambda kv: kv[0])

    print(f"Dataset: {args.pkl_path}")
    print("Class counts:")
    for class_id, count in ordered:
        class_name = mapping.get(class_id, f"unknown (id={class_id})")
        print(f"  {class_id:>3} | {class_name:<35} | {count}")
    print(f"Total samples: {total_samples}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
