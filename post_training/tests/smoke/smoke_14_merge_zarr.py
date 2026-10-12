"""Batch 02: verify exact multi-Zarr concatenation (RGB, Depth, terminal, indices)."""
from __future__ import annotations

import pathlib
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import zarr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
for folder in (REPO_ROOT, REPO_ROOT / "post_training" / "src",
               REPO_ROOT / "third_party" / "lerobot" / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from post_rl.data import data_prepare as dp
from post_rl.data.zarr_merge import merge_zarr_datasets
from smoke_12_lerobot_to_zarr import FakeLeRobot


RGB = ("head_rgb", "wrist_left_rgb", "wrist_right_rgb")


def create_input(folder, label, *, use_depth=False, reward_mode="last_gripper_release_sparse"):
    dest = folder / f"{label}.zarr"
    cfg = {
        "lerobot_root": str(folder / f"{label}_dataset"),
        "zarr_output_path": str(dest),
        "overwrite": False,
        "use_depth": use_depth,
        "reward_mode": reward_mode,
        "reward_open_infer_frames": 3,
        "reward_min_dwell": 3,
        "reward_expected_cycles": 1,
        "stream_batch_size": 5,
        "jpeg_quality": 95,
    }
    with patch.object(dp, "load_lerobot_dataset", return_value=FakeLeRobot(use_depth)):
        dp.run_lerobot_to_zarr(cfg)
    return dest


def create_merged(folder, sources):
    target = folder / "merged.zarr"
    result = merge_zarr_datasets({
        "zarr_output_path": str(target),
        "zarr_sources": [str(source) for source in sources],
        "return_gamma": 0.99,
        "overwrite": False,
    })
    assert result["zarr_path"] == str(target.resolve())
    return target, result


def check_sources(sources, dest, *, use_depth):
    output = zarr.open_group(str(dest), mode="r")
    groups = [zarr.open_group(str(path), mode="r") for path in sources]
    n = sum(int(g["data"]["action"].shape[0]) for g in groups)
    expected_ends = []
    cursor = 0
    for g in groups:
        expected_ends.extend((np.asarray(g["meta"]["episode_ends"][:]) + cursor).tolist())
        cursor += len(g["data"]["action"])
    assert n == 54
    assert expected_ends == [14, 27, 41, 54]
    np.testing.assert_array_equal(
        np.asarray(output["meta"]["episode_ends"][:]), expected_ends
    )
    assert output.attrs["reward_mode"] == "last_gripper_release_sparse"
    assert len(output.attrs["merge_inputs"]) == len(groups)
    assert len(output.attrs["source_manifest"]) == len(groups)
    assert output.attrs["rgb_shapes"] == groups[0].attrs["rgb_shapes"]

    merged_data = output["data"]
    for name in ("state", "next_state", "action", "next_action",
                 "reward", "done", "timeout"):
        expected = np.concatenate([np.asarray(g["data"][name][:]) for g in groups], axis=0)
        np.testing.assert_array_equal(
            merged_data[name][:], expected, err_msg=name
        )
    cursors = np.cumsum([0] + [len(g["data"]["action"]) for g in groups])
    expected_next = np.concatenate([
        np.asarray(g["data"]["next_index"][:]) + int(cursors[index])
        for index, g in enumerate(groups)
    ])
    np.testing.assert_array_equal(merged_data["next_index"][:], expected_next)
    for name in RGB:
        src_arrays = [g["data"] for g in groups]
        original_payload = b"".join(
            bytes(np.asarray(g[f"{name}_jpeg_data"][:], dtype=np.uint8))
            for g in src_arrays
        )
        actual_payload = bytes(np.asarray(
            merged_data[f"{name}_jpeg_data"][:], dtype=np.uint8
        ))
        assert actual_payload == original_payload, name
        offsets = [0]
        byte_offset = 0
        for g in src_arrays:
            local_offsets = np.asarray(g[f"{name}_jpeg_offsets"][:], dtype=np.int64)
            offsets.extend((local_offsets[1:] + byte_offset).tolist())
            byte_offset = offsets[-1]
        np.testing.assert_array_equal(
            merged_data[f"{name}_jpeg_offsets"][:], offsets,
            err_msg=f"{name}_jpeg_offsets"
        )
    if use_depth:
        for name in ("head_depth", "wrist_left_depth", "wrist_right_depth"):
            for key in (name, f"next_{name}"):
                expected = np.concatenate(
                    [np.asarray(g["data"][key][:]) for g in groups]
                )
                np.testing.assert_array_equal(merged_data[key][:], expected)
    else:
        assert "head_depth" not in merged_data

    rewards = np.asarray(merged_data["reward"][:]).reshape(-1)
    done = np.asarray(merged_data["done"][:]).reshape(-1)
    returns = np.asarray(merged_data["return"][:]).reshape(-1)
    expected = np.zeros(n, dtype=np.float32)
    running = 0.0
    for i in range(n - 1, -1, -1):
        running = rewards[i] + 0.99 * running * float(not done[i])
        expected[i] = running
    np.testing.assert_allclose(returns, expected, rtol=1e-6, atol=1e-6)
    assert np.flatnonzero(rewards).tolist() == [9, 22, 36, 49]
    assert np.flatnonzero(done).tolist() == [13, 26, 40, 53]


def test_normal(use_depth):
    with tempfile.TemporaryDirectory(prefix="arbim_merge_zarr_") as directory:
        root = pathlib.Path(directory)
        files = [
            create_input(root, "teleop", use_depth=use_depth),
            create_input(root, "rollout", use_depth=use_depth),
        ]
        snapshots = [
            np.asarray(zarr.open_group(str(p), mode="r")["data"]["reward"][:]).copy()
            for p in files
        ]
        destination, result = create_merged(root, files)
        assert result["num_frames"] == 54 and result["num_episodes"] == 4
        check_sources(files, destination, use_depth=use_depth)
        for path, before in zip(files, snapshots):
            current = zarr.open_group(str(path), mode="r")["data"]["reward"][:]
            np.testing.assert_array_equal(current, before)
        print(f"PASS multi-Zarr merge use_depth={use_depth}")


def main():
    test_normal(False)
    test_normal(True)
    print("SMOKE 14 multi-Zarr merge PASSED")


if __name__ == "__main__":
    main()
