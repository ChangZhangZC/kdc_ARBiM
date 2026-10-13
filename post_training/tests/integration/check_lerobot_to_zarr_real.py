"""Real-data Batch 01 check: convert LeRobot directly and compare to Reward V2 Zarr.

Usage:
 python post_training/tests/integration/check_lerobot_to_zarr_real.py \
   --source-npy data/sim_task1.npy \
   --output data/sim_task1_direct_test.zarr \
   --reference-zarr data/sim_task1.zarr \
   --expected-episodes 600 --expected-frames 90144

Only reads --source-npy's manifest; NEVER writes to the source NPY/reference Zarr.
Will refuse to overwrite an existing output Zarr.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[3]
for parent in (REPO_ROOT, REPO_ROOT / "post_training" / "src", REPO_ROOT / "third_party" / "lerobot" / "src"):
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
from post_rl.data.data_prepare import run_lerobot_to_zarr


def compare_reference(output, ref_path, frames, block_size=4096):
    reference = zarr.open_group(str(ref_path), mode="r")
    expected_ends = np.asarray(reference["meta"]["episode_ends"][:], dtype=np.int64)
    actual_ends = np.asarray(output["meta"]["episode_ends"][:], dtype=np.int64)
    np.testing.assert_array_equal(actual_ends, expected_ends)
    for key in ("state", "action", "done", "timeout", "reward", "return", "next_index"):
        left = output["data"][key]
        right = reference["data"][key]
        assert left.shape == right.shape, (key, left.shape, right.shape)
        for start in range(0, frames, block_size):
            end = min(frames, start + block_size)
            a = np.asarray(left[start:end])
            b = np.asarray(right[start:end])
            if key in {"return", "state", "action"}:
                np.testing.assert_allclose(
                    a, b, rtol=1e-5, atol=1e-5,
                    err_msg=f"{key} mismatch on rows [{start}:{end}]",
                )
            else:
                np.testing.assert_array_equal(
                    a, b, err_msg=f"{key} mismatch on rows [{start}:{end}]"
                )
        print(f"PASS: reference {key}")
    print("PASS: reference Episode boundaries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npy", default=None, help="Existing Processed NPY header, used ONLY to find LeRobot source_root")
    parser.add_argument("--lerobot-root", default=None, help="Override with source LeRobot dataset folder")
    parser.add_argument("--output", required=True, help="NEW output Zarr path; no overwrite permitted")
    parser.add_argument("--reference-zarr", default=None, help="Already migrated Reward V2 Zarr for verification")
    parser.add_argument("--expected-episodes", type=int, default=None)
    parser.add_argument("--expected-frames", type=int, default=None)
    parser.add_argument("--use-depth", action="store_true")
    args = parser.parse_args()

    original_manifest = None
    if args.source_npy:
        with open(args.source_npy, "rb") as source:
            original_manifest = np.load(source, allow_pickle=True).item()
    lerobot_root = args.lerobot_root or (original_manifest or {}).get("source_root")
    if not lerobot_root:
        parser.error("--lerobot-root or an NPY with source_root is required")
    src = Path(lerobot_root).expanduser().resolve()
    if not (src / "meta").is_dir():
        raise FileNotFoundError(f"LeRobot metadata not found: {src / 'meta'}")
    out = Path(args.output).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists, refusing overwrite: {out}")
    if args.reference_zarr and Path(args.reference_zarr).resolve() == out:
        raise ValueError("Reference and output paths must differ")

    config = {
        "lerobot_root": str(src),
        "zarr_output_path": str(out),
        "overwrite": False,
        "use_depth": args.use_depth,
        "reward_mode": "last_gripper_release_sparse",
        "reward_left_gripper_index": 7,
        "reward_right_gripper_index": 15,
        "reward_left_open_side": "auto",
        "reward_right_open_side": "auto",
        "reward_open_infer_frames": 5,
        "reward_min_dwell": 3,
        "reward_expected_cycles": 1,
        "stream_batch_size": 50,
        "jpeg_quality": 95,
    }
    print(f"LeRobot: {src}")
    print(f"Output:  {out}")
    run_lerobot_to_zarr(config)
    root = zarr.open_group(str(out), mode="r")
    data = root["data"]
    ends = np.asarray(root["meta"]["episode_ends"][:])
    frames = int(data["action"].shape[0])
    episodes = len(ends)
    assert ends[-1] == frames
    assert data["state"].shape == (frames, 16)
    assert data["action"].shape == (frames, 16)
    assert args.expected_frames is None or frames == args.expected_frames
    assert args.expected_episodes is None or episodes == args.expected_episodes
    rewards = np.asarray(data["reward"][:]).reshape(-1)
    done = np.asarray(data["done"][:]).reshape(-1)
    assert int(done.sum()) == episodes
    assert int(np.count_nonzero(rewards == 1.0)) == episodes
    assert np.count_nonzero(rewards) == episodes
    assert np.all(rewards[ends - 1] == 0.0)
    assert root.attrs["reward_mode"] == "last_gripper_release_sparse"
    assert root.attrs["source_manifest"][0]["kind"] == "teleop_lerobot"
    print(f"PASS: direct Zarr frames={frames}, episodes={episodes}, rewards={rewards.sum()}")

    if args.reference_zarr:
        compare_reference(root, Path(args.reference_zarr).resolve(), frames)
    print("REAL LeRobot -> Zarr TEST PASSED")


if __name__ == "__main__":
    main()
