#!/usr/bin/env python3

"""Convert a LeRobot meta/stats.json file into a LingBot norm_stats file.

The LingBot training chain in this repository expects a JSON file shaped like:

{
  "norm_stats": { ... },
  "count": 1234
}

LeRobot datasets already store statistics in ``meta/stats.json``, but that file
is flat and therefore cannot be consumed directly by the current LingBot
dataset wrappers. This script wraps the existing stats, preserves all keys, and
performs lightweight validation for the state/action fields used by training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_STATE_KEY = "observation.state"
DEFAULT_ACTION_KEY = "action"
REQUIRED_BOUNDS_KEYS = ("q01", "q99")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot meta/stats.json into a LingBot-compatible norm_stats file.",
    )
    parser.add_argument(
        "stats_json",
        type=Path,
        help="Path to the source LeRobot meta/stats.json file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the converted norm stats JSON. "
            "Defaults to assets/norm_stats/<dataset_name>.json under this repo."
        ),
    )
    parser.add_argument(
        "--state-key",
        default=DEFAULT_STATE_KEY,
        help=f"State key expected by training. Default: {DEFAULT_STATE_KEY}",
    )
    parser.add_argument(
        "--action-key",
        default=DEFAULT_ACTION_KEY,
        help=f"Action key expected by training. Default: {DEFAULT_ACTION_KEY}",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input stats file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input is not valid JSON: {path}") from exc


def _infer_default_output(stats_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_name = stats_path.resolve().parents[1].name
    return repo_root / "assets" / "norm_stats" / f"{dataset_name}.json"


def _extract_count(stats: dict[str, Any], state_key: str, action_key: str) -> int:
    for key in (state_key, action_key):
        feature_stats = stats.get(key)
        if not isinstance(feature_stats, dict):
            continue
        raw_count = feature_stats.get("count")
        if isinstance(raw_count, list) and raw_count:
            return int(raw_count[0])
        if isinstance(raw_count, (int, float)):
            return int(raw_count)
    raise ValueError(
        f"Could not infer sample count from keys {state_key!r} / {action_key!r}. "
        "Expected a `count` field in one of them."
    )


def _validate_feature(stats: dict[str, Any], key: str) -> None:
    if key not in stats:
        raise KeyError(f"Required feature key missing from stats.json: {key}")

    feature_stats = stats[key]
    if not isinstance(feature_stats, dict):
        raise TypeError(f"Expected stats for {key!r} to be a JSON object, got {type(feature_stats).__name__}")

    for bounds_key in REQUIRED_BOUNDS_KEYS:
        if bounds_key not in feature_stats:
            raise KeyError(
                f"Stats for {key!r} are missing {bounds_key!r}. "
                "Current LingBot configs use bounds_99-style normalization."
            )


def _build_output(stats: dict[str, Any], state_key: str, action_key: str) -> dict[str, Any]:
    _validate_feature(stats, state_key)
    _validate_feature(stats, action_key)
    count = _extract_count(stats, state_key, action_key)
    return {
        "norm_stats": stats,
        "count": count,
    }


def main() -> None:
    args = _parse_args()
    stats_path = args.stats_json.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else _infer_default_output(stats_path)

    stats = _load_json(stats_path)
    wrapped = _build_output(stats, args.state_key, args.action_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(wrapped, indent=2))

    print(f"Source: {stats_path}")
    print(f"Output: {output_path}")
    print(f"Count: {wrapped['count']}")
    print(f"State key: {args.state_key}")
    print(f"Action key: {args.action_key}")
    if output_path.parent.name == "norm_stats":
        print(f"Training config value: assets/norm_stats/{output_path.name}")
    else:
        print(f"Training config value: {output_path}")


if __name__ == "__main__":
    main()
