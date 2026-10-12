"""End-to-end real-data check: Staging -> LeRobot -> Zarr -> merged IL/RL.

Use a *new* --output-dir. Input Teleop LeRobot, existing Reward V2 Zarr, and
Data Wheel Staging are never modified. Can be time-consuming: video re-encode
is necessary to build a merged LeRobot dataset for ACT.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "post_training" / "src", ROOT / "third_party" / "lerobot" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from post_rl.data import data_prepare as dp
from post_rl.data.staging_to_lerobot import (
    convert_staging_to_lerobot, discover_staging, load_staging,
)
from post_rl.data.zarr_merge import merge_zarr_datasets
from post_rl.data.lerobot_merge import merge_lerobot_datasets


def inspect_staging(paths):
    valid = []
    invalid = []
    for path in paths:
        item = load_staging(path)
        if item["reward_status"] == "valid":
            valid.append((path, item))
        else:
            invalid.append(path)
    if not valid:
        raise ValueError("No valid-reward staging episodes")
    return valid, invalid


def assert_reward_boundaries(group, starts, accepted):
    data = group["data"]
    reward = np.asarray(data["reward"][:], dtype=np.float32).reshape(-1)
    done = np.asarray(data["done"][:], dtype=bool).reshape(-1)
    expected_episode_ends = []
    expected_reward_indices = []
    cursor = int(starts)
    for _, stage in accepted:
        expected_reward_indices.append(cursor + int(stage["reward_event"]["frame_offset"]))
        cursor += int(stage["num_steps"])
        expected_episode_ends.append(cursor)
    # Starts refer to the end of the original Teleop source.
    observed_reward = np.flatnonzero(reward[int(starts):]) + int(starts)
    observed_done = np.flatnonzero(done[int(starts):]) + int(starts)
    np.testing.assert_array_equal(observed_reward, expected_reward_indices)
    np.testing.assert_array_equal(observed_done, np.asarray(expected_episode_ends) - 1)
    np.testing.assert_array_equal(
        np.asarray(group["meta"]["episode_ends"][:])[-len(accepted):],
        expected_episode_ends,
    )
    return expected_episode_ends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teleop-lerobot", required=True)
    parser.add_argument("--teleop-zarr", required=True, help="Already validated Reward V2 Teleop Zarr")
    parser.add_argument("--staging", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory; must not exist")
    parser.add_argument("--expected-teleop-episodes", type=int, default=600)
    parser.add_argument("--expected-teleop-frames", type=int, default=90144)
    parser.add_argument("--task", default="ARBiM Data Wheel")
    parser.add_argument("--use-depth", action="store_true")
    parser.add_argument(
        "--verify-combined-lerobot-to-zarr", action="store_true",
        help="Expensive: convert merged LeRobot back to Zarr to cross-check reward provenance",
    )
    args = parser.parse_args()

    teleop = Path(args.teleop_lerobot).expanduser().resolve()
    base = Path(args.teleop_zarr).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    if not (teleop / "meta").is_dir() or not base.is_dir():
        raise FileNotFoundError("Teleop LeRobot or baseline Reward V2 Zarr missing")
    if out.exists():
        raise FileExistsError(f"Output directory exists: {out}")
    if out == teleop or out in teleop.parents or teleop in out.parents:
        raise ValueError("Do not nest output inside Teleop LeRobot source")
    if out == base or out in base.parents or base in out.parents:
        raise ValueError("Do not nest output inside baseline Zarr source")
    paths = discover_staging(args.staging)
    if any(out == path or out in path.parents for path in paths):
        raise ValueError("Do not nest output inside staging source files")
    accepted, invalid = inspect_staging(paths)
    baseline = zarr.open_group(str(base), mode="r")
    base_ends = np.asarray(baseline["meta"]["episode_ends"][:], dtype=np.int64)
    assert baseline.attrs.get("reward_mode") == "last_gripper_release_sparse", (
        "Baseline must have explicit Reward V2 mode (migrated Zarr)"
    )
    assert int(base_ends[-1]) == args.expected_teleop_frames
    assert len(base_ends) == args.expected_teleop_episodes
    rollout_frames = sum(int(item["num_steps"]) for _, item in accepted)

    out.mkdir(parents=True)
    wheel_lev = out / "rollout_lerobot"
    wheel_zarr = out / "rollout.zarr"
    merged_zarr = out / "combined.zarr"
    merged_il = out / "combined_lerobot"

    print(f"[STEP 1] Staging → LeRobot: valid={len(accepted)} skipped_invalid={len(invalid)}", flush=True)
    converted = convert_staging_to_lerobot(
        paths, teleop, wheel_lev, "lerobot/arbim_rollout_data_contract",
        args.task, use_depth=args.use_depth, require_valid_reward=True,
    )
    assert converted["num_episodes"] == len(accepted)
    assert converted["num_frames"] == rollout_frames

    print("[STEP 2] LeRobot → reward-preserving Zarr", flush=True)
    wheel_info = dp.run_lerobot_to_zarr({
        "lerobot_root": str(wheel_lev),
        "zarr_output_path": str(wheel_zarr),
        "overwrite": False,
        "use_depth": args.use_depth,
        "reward_source": "recorded",
        "reward_mode": "last_gripper_release_sparse",
        "stream_batch_size": 50,
        "jpeg_quality": 95,
    })
    assert wheel_info["num_frames"] == rollout_frames
    wheel = zarr.open_group(str(wheel_zarr), mode="r")
    assert_reward_boundaries(wheel, 0, accepted)

    print("[STEP 3] Merge Teleop Zarr + Rollout Zarr", flush=True)
    merged = merge_zarr_datasets({
        "zarr_sources": [{"name": "teleop", "path": str(base)},
                         {"name": "rollout", "path": str(wheel_zarr)}],
        "zarr_output_path": str(merged_zarr),
        "return_gamma": 0.99,
        "overwrite": False,
    })
    assert merged["num_frames"] == args.expected_teleop_frames + rollout_frames
    assert merged["num_episodes"] == args.expected_teleop_episodes + len(accepted)
    combined = zarr.open_group(str(merged_zarr), mode="r")
    assert_reward_boundaries(combined, args.expected_teleop_frames, accepted)
    np.testing.assert_array_equal(
        combined["data"]["reward"][:args.expected_teleop_frames],
        baseline["data"]["reward"][:],
    )
    np.testing.assert_array_equal(
        combined["meta"]["episode_ends"][:args.expected_teleop_episodes],
        base_ends,
    )

    print("[STEP 4] Merge Teleop LeRobot + Rollout LeRobot for IL", flush=True)
    il = merge_lerobot_datasets(
        [teleop, wheel_lev], merged_il, "lerobot/arbim_combined_data_contract",
        use_depth=args.use_depth, batch_size=25, fallback_task=args.task,
    )
    assert il["num_episodes"] == merged["num_episodes"]
    assert il["num_frames"] == merged["num_frames"]

    if args.verify_combined_lerobot_to_zarr:
        print("[STEP 5] Verify merged LeRobot Reward provenance (slow)", flush=True)
        check_path = out / "combined_from_lerobot.zarr"
        dp.run_lerobot_to_zarr({
            "lerobot_root": str(merged_il),
            "zarr_output_path": str(check_path),
            "overwrite": False,
            "use_depth": args.use_depth,
            "reward_source": "auto",
            "reward_mode": "last_gripper_release_sparse",
            "stream_batch_size": 50,
            "jpeg_quality": 95,
        })
        rebuilt = zarr.open_group(str(check_path), mode="r")
        np.testing.assert_array_equal(
            np.asarray(rebuilt["data"]["reward"][:]),
            np.asarray(combined["data"]["reward"][:]),
        )
        np.testing.assert_array_equal(
            np.asarray(rebuilt["meta"]["episode_ends"][:]),
            np.asarray(combined["meta"]["episode_ends"][:]),
        )

    print(f"PASS: Staging valid={len(accepted)}, skipped invalid={len(invalid)}")
    print(f"PASS: total Episodes={merged['num_episodes']}, Frames={merged['num_frames']}")
    print(f"PASS: IL LeRobot={merged_il}")
    print(f"PASS: RL Zarr={merged_zarr}")
    print("REAL DATA CONTRACT END-TO-END PASSED")


if __name__ == "__main__":
    main()
