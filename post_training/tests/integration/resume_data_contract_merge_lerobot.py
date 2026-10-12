"""Resume ARBiM real E2E at Step 5/6 without rebuilding existing Zarrs.

Requires outputs previously created by check_data_contract_real.py through
Step 4/6. Reads prior datasets read-only and creates only a NEW merged IL
LeRobot folder inside --work-root. Never deletes incomplete prior outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for folder in (
    REPO_ROOT, REPO_ROOT / "post_training" / "src",
    REPO_ROOT / "third_party" / "lerobot" / "src"
):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from check_data_contract_real import resolve_lerobot, verify_zarr, compare_reference
from post_rl.data.data_prepare import load_lerobot_dataset, _group_episode_indices
from post_rl.data.lerobot_merge import merge_lerobot_datasets, _episode_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", help="Original Teleop LeRobot dataset")
    parser.add_argument("--source-npy", help="Historic NPY: read manifest to resolve LeRobot root")
    parser.add_argument("--work-root", required=True, help="EXISTING E2E work directory")
    parser.add_argument("--reference-zarr", help="Optional old Reward V2 Zarr to compare")
    parser.add_argument(
        "--output-name", default="merged_il_lerobot_resume_01",
        help="Name of a NEW subdirectory in work-root; must not already exist",
    )
    parser.add_argument("--expected-teleop-episodes", type=int, default=600)
    parser.add_argument("--expected-teleop-frames", type=int, default=90144)
    parser.add_argument("--expected-rollout-episodes", type=int, default=4)
    parser.add_argument("--expected-rollout-frames", type=int, default=600)
    parser.add_argument("--fallback-task", default="ARBiM Data Wheel")
    parser.add_argument("--zarr-use-depth", action="store_true",
                        help="Opt in to Depth in Zarr (default RGB-only)")
    args = parser.parse_args()

    original = resolve_lerobot(args)
    root = Path(args.work_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Existing work-root directory not found: {root}")
    if (
        not args.output_name or args.output_name in (".", "..")
        or Path(args.output_name).name != args.output_name
    ):
        raise ValueError("--output-name must be a simple new folder name")
    dest = root / args.output_name
    if dest.exists():
        raise FileExistsError(f"Merge output already exists; refusing overwrite: {dest}")

    wheel_lerobot = root / "rollout_lerobot"
    wheel_provenance = wheel_lerobot / "meta" / "arbim_staging_provenance.json"
    teleop_zarr = root / "teleop_reward_v2.zarr"
    wheel_zarr = root / "rollout_reward_v2.zarr"
    merged_zarr = root / "merged_offline_rl.zarr"
    for dataset in (wheel_lerobot, teleop_zarr, wheel_zarr, merged_zarr):
        if not dataset.is_dir():
            raise FileNotFoundError(f"Required completed E2E dataset missing: {dataset}")
    if not wheel_provenance.is_file():
        raise FileNotFoundError(f"Missing Staging reward provenance: {wheel_provenance}")

    base, base_ends, base_rewards = verify_zarr(
        teleop_zarr, args.expected_teleop_frames, args.expected_teleop_episodes,
        use_depth=args.zarr_use_depth,
    )
    wheel, wheel_ends, wheel_rewards = verify_zarr(
        wheel_zarr, args.expected_rollout_frames, args.expected_rollout_episodes,
        use_depth=args.zarr_use_depth,
    )
    total_frames = args.expected_teleop_frames + args.expected_rollout_frames
    total_episodes = args.expected_teleop_episodes + args.expected_rollout_episodes
    combined, ends, combined_rewards = verify_zarr(
        merged_zarr, total_frames, total_episodes, use_depth=args.zarr_use_depth,
    )
    np.testing.assert_array_equal(ends[:len(base_ends)], base_ends)
    np.testing.assert_array_equal(
        ends[len(base_ends):], wheel_ends + args.expected_teleop_frames
    )
    np.testing.assert_array_equal(
        combined_rewards[:args.expected_teleop_frames], base_rewards
    )
    np.testing.assert_array_equal(
        combined_rewards[args.expected_teleop_frames:], wheel_rewards
    )
    manifest = json.loads(wheel_provenance.read_text(encoding="utf-8"))
    if not manifest.get("use_depth"):
        raise ValueError("Archived Rollout LeRobot must preserve Depth")
    if manifest.get("format") != "arbim_lerobot_staging_provenance_v1":
        raise ValueError("Rollout LeRobot has unexpected Reward provenance")
    records = manifest.get("episodes", [])
    if len(records) != args.expected_rollout_episodes:
        raise ValueError("Recorded Rollout Episode count mismatch")
    starts = np.r_[0, wheel_ends[:-1]]
    expected_indices = []
    for start, row in zip(starts, records, strict=True):
        if row.get("reward_status") != "valid":
            raise ValueError("Recorded Rollout includes invalid Reward V2")
        expected_indices.append(int(start) + int(row["reward_index"]))
    np.testing.assert_array_equal(
        np.flatnonzero(wheel_rewards), expected_indices
    )
    assert len(combined.attrs["merge_inputs"]) == 2
    if args.reference_zarr:
        print("[Preflight] Comparing preserved Teleop Zarr with Reward V2 reference", flush=True)
        compare_reference(base, Path(args.reference_zarr).expanduser().resolve())
    print(
        f"[Preflight PASS] Reusing stages 1–4: frames={total_frames}, "
        f"episodes={total_episodes}; Rollout reward indices={expected_indices}",
        flush=True,
    )

    # Detect task metadata incompatibilities before allocating or encoding 90k frames.
    teleop = load_lerobot_dataset(original)
    rollout = load_lerobot_dataset(wheel_lerobot)
    for label, dataset in (("Teleop", teleop), ("Rollout", rollout)):
        indices = _group_episode_indices(dataset)
        if not indices:
            raise ValueError(f"{label}: no LeRobot Episodes")
        first_episode = next(iter(indices.values()))
        task = _episode_task(dataset, first_episode[0], args.fallback_task)
        print(f"[Preflight PASS] {label} task: {task!r}", flush=True)
    del teleop, rollout

    print(f"\n[5/6] Resume LeRobot IL merge → {dest}", flush=True)
    result = merge_lerobot_datasets(
        [original, wheel_lerobot], dest, "arbim/il_contract_resume",
        use_depth=True, batch_size=25, fallback_task=args.fallback_task,
    )
    assert result["num_episodes"] == total_episodes
    assert result["num_frames"] == total_frames

    print("\n[6/6] Load merged LeRobot and check IL/RL contracts", flush=True)
    final_il = load_lerobot_dataset(result["lerobot_root"])
    assert len(final_il) == total_frames
    assert int(final_il.meta.total_episodes) == total_episodes
    assert combined["data"]["next_index"].shape == (total_frames,)
    print("\nDATA CONTRACT REAL E2E RESUME PASSED", flush=True)
    print(f"LeRobot IL: {result['lerobot_root']}", flush=True)
    print(f"Offline RL: {merged_zarr}", flush=True)


if __name__ == "__main__":
    main()
