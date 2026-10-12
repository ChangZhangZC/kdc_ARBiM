"""ARBiM complete real-data contract test (expensive, non-destructive).

Builds NEW LeRobot and Zarr artifacts under a fresh --work-root, exercises
Staging→LeRobot, both LeRobot→Zarr routes, Zarr merge and LeRobot merge.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT, ROOT / "post_training" / "src", ROOT / "third_party" / "lerobot" / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from post_rl.data.data_prepare import load_lerobot_dataset, run_lerobot_to_zarr
from post_rl.data.staging_to_lerobot import convert_staging_to_lerobot
from post_rl.data.zarr_merge import merge_zarr_datasets
from post_rl.data.lerobot_merge import merge_lerobot_datasets


def resolve_lerobot(args):
    if args.lerobot_root:
        result = Path(args.lerobot_root).expanduser().resolve()
    elif args.source_npy:
        with open(args.source_npy, "rb") as fp:
            manifest = np.load(fp, allow_pickle=True).item()
        if manifest.get("__format__") != "arbim_processed_npy_stream_v2":
            raise ValueError("Old NPY has unexpected format")
        result = Path(manifest["source_root"]).expanduser().resolve()
    else:
        raise ValueError("Provide --lerobot-root or --source-npy")
    if not (result / "meta").is_dir():
        raise FileNotFoundError(f"LeRobot meta/ directory missing: {result}")
    return result


def verify_zarr(path, frames, episodes, reward_v2=True, use_depth=False):
    root = zarr.open_group(str(path), mode="r")
    data = root["data"]
    ep_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if len(ep_ends) != episodes or ep_ends[-1] != frames:
        raise AssertionError(f"{path}: unexpected episode boundaries")
    if data["state"].shape != (frames, 16) or data["action"].shape != (frames, 16):
        raise AssertionError(f"{path}: action/state contract mismatch")
    for name in ("head_rgb", "wrist_left_rgb", "wrist_right_rgb"):
        assert len(data[f"{name}_jpeg_offsets"]) == frames + 1
    for name in ("head_depth", "wrist_left_depth", "wrist_right_depth"):
        if (name in data) != use_depth or (f"next_{name}" in data) != use_depth:
            raise AssertionError(f"{path}: Depth storage for {name} differs from use_depth={use_depth}")
    reward = np.asarray(data["reward"][:]).reshape(-1)
    done = np.asarray(data["done"][:]).reshape(-1)
    returns = np.asarray(data["return"][:]).reshape(-1)
    assert int(done.sum()) == episodes
    assert np.isfinite(reward).all() and np.isfinite(returns).all()
    if reward_v2:
        assert root.attrs["reward_mode"] == "last_gripper_release_sparse"
        assert np.count_nonzero(reward) == episodes
        assert int(np.count_nonzero(reward == 1)) == episodes
        assert np.all(reward[ep_ends - 1] == 0)
    return root, ep_ends, reward


def compare_reference(created, reference):
    src = zarr.open_group(str(reference), mode="r")
    np.testing.assert_array_equal(
        created["meta"]["episode_ends"][:],
        src["meta"]["episode_ends"][:],
    )
    length = created["data"]["action"].shape[0]
    for name in ("action", "state", "done", "timeout", "reward", "return"):
        a, b = created["data"][name], src["data"][name]
        assert a.shape == b.shape
        for start in range(0, length, 2048):
            stop = min(start + 2048, length)
            np.testing.assert_allclose(
                a[start:stop], b[start:stop], rtol=1e-5, atol=1e-5,
                err_msg=f"Reference mismatch: {name}, frames {start}:{stop}",
            )
        print(f"[PASS] reference {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", help="Original Teleop LeRobot dataset")
    parser.add_argument("--source-npy", help="Optional historical NPY to resolve source_root only")
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--reference-zarr", help="Previously migrated Reward V2 Teleop Zarr")
    parser.add_argument("--reuse-teleop-zarr",
                        help="Already validated Teleop Reward V2 Zarr: skip expensive Step 2 rebuild")
    parser.add_argument("--work-root", required=True, help="NEW output directory")
    parser.add_argument("--expected-teleop-frames", type=int, default=90144)
    parser.add_argument("--expected-teleop-episodes", type=int, default=600)
    parser.add_argument("--task", default="ARBiM Data Wheel")
    parser.add_argument("--zarr-use-depth", action="store_true",
                        help="Opt in to writing Depth in Zarr (default RGB-only)")
    args = parser.parse_args()

    original = resolve_lerobot(args)
    output = Path(args.work_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"work-root already exists: {output}")
    stage = Path(args.staging_dir).expanduser().resolve()
    if not stage.is_dir():
        raise FileNotFoundError(f"Staging directory missing: {stage}")
    if output == original or original in output.parents or stage in output.parents:
        raise ValueError("work-root must be separate from input datasets")

    print("\n[1/6] Staging -> Data Wheel LeRobot (Reward V2-valid subset)")
    result = convert_staging_to_lerobot(
        [stage], original, output / "rollout_lerobot",
        "arbim/flywheel_contract_check", args.task,
        use_depth=True, require_valid_reward=True,
    )
    import json
    manifest = json.loads(Path(result["provenance"]).read_text(encoding="utf-8"))
    assert manifest["use_depth"], "Staging -> LeRobot must archive Depth"
    rollout_count = result["num_episodes"]
    rollout_frames = result["num_frames"]
    assert rollout_count > 0 and rollout_frames > 0
    print(f"  Data Wheel: {rollout_count} episodes, {rollout_frames} frames")

    common = {
        "overwrite": False, "use_depth": args.zarr_use_depth,
        "stream_batch_size": 50, "jpeg_quality": 95,
        "reward_mode": "last_gripper_release_sparse",
        "reward_source": "auto",
        "reward_left_gripper_index": 7, "reward_right_gripper_index": 15,
        "reward_left_open_side": "auto", "reward_right_open_side": "auto",
        "reward_open_infer_frames": 5, "reward_min_dwell": 3,
        "reward_expected_cycles": 1,
    }
    if args.reuse_teleop_zarr:
        print("\n[2/6] Reuse previously validated Teleop Zarr (no rebuild)")
        teleop_zarr = Path(args.reuse_teleop_zarr).expanduser().resolve()
        if not teleop_zarr.is_dir():
            raise FileNotFoundError(f"reused Teleop Zarr missing: {teleop_zarr}")
        if teleop_zarr == output or output in teleop_zarr.parents:
            raise ValueError("Reused Teleop Zarr must be external to the new work-root")
    else:
        print("\n[2/6] Original Teleop LeRobot -> Zarr")
        teleop_zarr = output / "teleop_reward_v2.zarr"
        run_lerobot_to_zarr({
            **common, "lerobot_root": str(original), "zarr_output_path": str(teleop_zarr),
        })
    teleop, _, _ = verify_zarr(
        teleop_zarr, args.expected_teleop_frames, args.expected_teleop_episodes,
        use_depth=args.zarr_use_depth,
    )
    if args.reference_zarr:
        compare_reference(teleop, Path(args.reference_zarr).expanduser().resolve())

    print("\n[3/6] Data Wheel LeRobot -> Zarr, preserving recorded Reward V2")
    rollout_zarr = output / "rollout_reward_v2.zarr"
    run_lerobot_to_zarr({
        **common, "lerobot_root": result["lerobot_root"],
        "zarr_output_path": str(rollout_zarr),
    })
    wheel, wheel_ends, wheel_reward = verify_zarr(
        rollout_zarr, rollout_frames, rollout_count, use_depth=args.zarr_use_depth,
    )
    # Verify recorded reward indices per episode, not just the reward sum.
    import json
    provenance = json.loads(Path(result["provenance"]).read_text(encoding="utf-8"))
    starts = [0, *wheel_ends[:-1].tolist()]
    recorded_indices = [
        starts[i] + entry["reward_index"]
        for i, entry in enumerate(provenance["episodes"])
    ]
    np.testing.assert_array_equal(
        np.flatnonzero(wheel_reward), recorded_indices,
    )
    assert wheel.attrs["reward_event"]["source"] == "staging_recorded_v1"

    print("\n[4/6] Teleop + Rollout Zarr merge")
    merged_zarr = output / "merged_offline_rl.zarr"
    merge_result = merge_zarr_datasets({
        "zarr_sources": [str(teleop_zarr), str(rollout_zarr)],
        "zarr_output_path": str(merged_zarr),
        "overwrite": False, "return_gamma": 0.99,
    })
    total_frames = args.expected_teleop_frames + rollout_frames
    total_episodes = args.expected_teleop_episodes + rollout_count
    assert merge_result["num_frames"] == total_frames
    assert merge_result["num_episodes"] == total_episodes
    merged, ends, rewards = verify_zarr(
        merged_zarr, total_frames, total_episodes, use_depth=args.zarr_use_depth
    )
    np.testing.assert_array_equal(
        rewards[args.expected_teleop_frames:], wheel_reward,
    )
    np.testing.assert_array_equal(
        ends[-rollout_count:],
        wheel_ends + args.expected_teleop_frames,
    )
    assert len(merged.attrs["merge_inputs"]) == 2

    print("\n[5/6] Teleop + Rollout LeRobot merge for ACT IL")
    merged_lerobot_result = merge_lerobot_datasets(
        [original, result["lerobot_root"]],
        output / "merged_il_lerobot", "arbim/il_contract_check",
        use_depth=True, batch_size=25, fallback_task=args.task,
    )
    assert merged_lerobot_result["num_frames"] == total_frames
    assert merged_lerobot_result["num_episodes"] == total_episodes

    print("\n[6/6] Read back merged LeRobot and final Zarr contracts")
    combined_il = load_lerobot_dataset(merged_lerobot_result["lerobot_root"])
    assert len(combined_il) == total_frames
    assert int(combined_il.meta.total_episodes) == total_episodes
    assert set(("observation.depth_h", "observation.depth_l",
                "observation.depth_r")).issubset(combined_il.meta.features)
    for frame_id in (0, args.expected_teleop_frames):
        frame = combined_il[frame_id]  # Decode one original and one Data Wheel video sample.
        for key in ("observation.depth_h", "observation.depth_l", "observation.depth_r"):
            depth = np.asarray(frame[key])
            assert depth.ndim == 3 and depth.shape[0] == 3, (key, depth.shape)
            assert np.isfinite(depth).all(), (frame_id, key)
    print("[PASS] merged LeRobot depth video decodes at Teleop/Rollout boundaries", flush=True)
    assert merged["data"]["next_index"].shape == (total_frames,)
    print("\nDATA CONTRACT FULL REAL-DATA TEST PASSED")
    print(f"LeRobot IL: {merged_lerobot_result['lerobot_root']}")
    print(f"Offline RL: {merged_zarr}")


if __name__ == "__main__":
    main()
