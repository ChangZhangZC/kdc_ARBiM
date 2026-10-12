"""Batch 02: Zarr merge rejection and non-destructive failure tests."""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "post_training" / "src", ROOT / "third_party" / "lerobot" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from post_rl.data.zarr_merge import merge_zarr_datasets
from smoke_14_merge_zarr import create_input


def reject(config, error, message):
    try:
        merge_zarr_datasets(config)
    except error as exc:
        assert message.lower() in str(exc).lower(), str(exc)
    else:
        raise AssertionError(f"Expected {error.__name__}: {message}")


def test_invalid_options():
    with tempfile.TemporaryDirectory(prefix="arbim_zarr_bad_options_") as name:
        root = pathlib.Path(name)
        source = create_input(root, "original")
        second = create_input(root, "rollout")
        target = root / "out.zarr"
        config = {
            "zarr_sources": [str(source), str(second)],
            "zarr_output_path": str(target),
        }
        reject({**config, "return_gamma": 0}, ValueError, "return_gamma")
        reject({**config, "overwrite": True}, ValueError, "refuses overwrite")
        reject({**config, "zarr_sources": [str(source)]}, ValueError, "two")
        reject({**config, "zarr_sources": [str(source), str(source)]},
               ValueError, "duplicate")
        reject({**config, "zarr_output_path": str(source)},
               ValueError, "output Zarr")
        reject({**config, "zarr_output_path": str(source / "child.zarr")},
               ValueError, "output Zarr")
        assert not target.exists()
        merge_zarr_datasets(config)
        reject(config, FileExistsError, "already exists")
        assert target.is_dir()
        print("PASS config, overwrite refusal and duplicate/path safety")


def test_incompatible_reward_and_depth():
    with tempfile.TemporaryDirectory(prefix="arbim_zarr_bad_reward_") as name:
        root = pathlib.Path(name)
        v2 = create_input(root, "v2")
        shaped = create_input(root, "shaped", reward_mode="episode_end_shaped")
        depth = create_input(root, "depth", use_depth=True)
        dest = root / "merged.zarr"
        cfg = {"zarr_sources": [str(v2), str(shaped)],
               "zarr_output_path": str(dest)}
        reject(cfg, ValueError, "reward_mode")
        reject({**cfg, "zarr_sources": [str(v2), str(depth)]},
               ValueError, "array")
        assert not dest.exists()
        print("PASS mixed Reward semantics and incompatible Depth rejected")


def test_corrupt_source():
    with tempfile.TemporaryDirectory(prefix="arbim_zarr_corrupt_") as name:
        root = pathlib.Path(name)
        one = create_input(root, "original")
        two = create_input(root, "round")
        target = root / "out.zarr"
        cfg = {"zarr_sources": [str(one), str(two)],
               "zarr_output_path": str(target)}
        src = zarr.open_group(str(two), mode="a")
        # Invalidate a terminal next_index; the merge must fail *before* writes.
        value = int(src["data"]["next_index"][13])
        src["data"]["next_index"][13] = 14
        reject(cfg, ValueError, "next_index")
        assert not target.exists()
        src["data"]["next_index"][13] = value
        # Reward V2 terminal reward must not be accepted.
        original = float(src["data"]["reward"][13, 0])
        src["data"]["reward"][13, 0] = 1.0
        reject(cfg, ValueError, "Reward V2")
        assert not target.exists()
        src["data"]["reward"][13, 0] = original
        # Offset metadata must describe byte-stream boundaries exactly.
        before = int(src["data"]["head_rgb_jpeg_offsets"][1])
        src["data"]["head_rgb_jpeg_offsets"][1] = 0
        reject(cfg, ValueError, "JPEG offsets")
        assert not target.exists()
        src["data"]["head_rgb_jpeg_offsets"][1] = before
        merge_zarr_datasets(cfg)
        print("PASS corrupt Zarr rejected without partial output")


def main():
    test_invalid_options()
    test_incompatible_reward_and_depth()
    test_corrupt_source()
    print("SMOKE 15 merge_zarr negative cases PASSED")


if __name__ == "__main__":
    main()
