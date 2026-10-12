"""Batch 01 regression: direct LeRobot -> Zarr equals legacy LeRobot -> NPY -> Zarr."""
from __future__ import annotations

import pathlib
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import zarr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_RL_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
for path in (REPO_ROOT, POST_RL_SRC, LEROBOT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from post_rl.data import data_prepare as dp


class FakeLeRobot:
    """Small in-memory LeRobot contract; forces the per-frame decode fallback."""

    def __init__(self, use_depth=False):
        self.use_depth = use_depth
        self.episodes = [14, 13]
        self.frames = []
        for episode, length in enumerate(self.episodes):
            action = np.zeros((length, 16), dtype=np.float32)
            left_close = (3, 7) if episode == 0 else (2, 6)
            right_close = (4, 9) if episode == 0 else (3, 8)
            action[left_close[0]:left_close[1], 7] = 1.0
            action[right_close[0]:right_close[1], 15] = 1.0
            for step in range(length):
                sample = {
                    "episode_index": episode,
                    "observation.state": np.full((16,), step / 10, dtype=np.float32),
                    "action": action[step],
                }
                for key in dp.RGB_FEATURE_TO_BUFFER:
                    sample[key] = np.full(
                        (3, 16, 16), (step % 10) / 10, dtype=np.float32
                    )
                if use_depth:
                    for key in dp.DEPTH_FEATURE_TO_BUFFER:
                        sample[key] = np.full((3, 16, 16), 0.25, dtype=np.float32)
                self.frames.append(sample)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


def compare_data(a, b):
    for key in a.keys():
        assert key in b, f"missing key in direct Zarr: {key}"
        np.testing.assert_array_equal(
            a[key][:], b[key][:], err_msg=f"direct/legacy mismatch: {key}"
        )


def run_case(use_depth):
    ds = FakeLeRobot(use_depth=use_depth)
    with tempfile.TemporaryDirectory(prefix="arbim_direct_zarr_") as tmp:
        root = pathlib.Path(tmp)
        processed = root / "legacy.npy"
        old_zarr = root / "legacy.zarr"
        direct_zarr = root / "direct.zarr"
        config = {
            "lerobot_root": str(root / "synthetic_lerobot"),
            "reward_mode": "last_gripper_release_sparse",
            "reward_left_gripper_index": 7,
            "reward_right_gripper_index": 15,
            "reward_open_infer_frames": 3,
            "reward_min_dwell": 3,
            "reward_expected_cycles": 1,
            "stream_batch_size": 6,
            "use_depth": use_depth,
            "jpeg_quality": 95,
            "overwrite": False,
        }
        with patch.object(dp, "load_lerobot_dataset", return_value=ds):
            old_cfg = dict(
                config, processed_npy_output=str(processed)
            )
            npy_result = dp.process_raw_teleop_to_npy(old_cfg)
            assert npy_result["num_frames"] == 27
            legacy_result = dp.run_build_db({
                **config,
                "zarr_output_path": str(old_zarr),
                "teleop_sources": [{"name": "teleop", "path": str(processed)}],
            })
            direct_result = dp.run_lerobot_to_zarr({
                **config,
                "zarr_output_path": str(direct_zarr),
            })

        assert legacy_result["num_frames"] == direct_result["num_frames"] == 27
        assert legacy_result["num_episodes"] == direct_result["num_episodes"] == 2

        legacy = zarr.open_group(str(old_zarr), mode="r")
        direct = zarr.open_group(str(direct_zarr), mode="r")
        compare_data(legacy["data"], direct["data"])
        np.testing.assert_array_equal(
            legacy["meta"]["episode_ends"][:],
            direct["meta"]["episode_ends"][:],
        )
        rewards = direct["data"]["reward"][:].reshape(-1)
        ends = direct["meta"]["episode_ends"][:]
        assert np.flatnonzero(rewards).tolist() == [9, 22]
        assert ends.tolist() == [14, 27]
        assert direct.attrs["source_manifest"][0]["kind"] == "teleop_lerobot"
        assert direct.attrs["source_manifest"][0]["reward_mode"] == (
            "last_gripper_release_sparse"
        )
        # Direct conversion must refuse accidental overwrite.
        with patch.object(dp, "load_lerobot_dataset", return_value=ds):
            try:
                dp.run_lerobot_to_zarr({
                    **config,
                    "zarr_output_path": str(direct_zarr),
                })
            except FileExistsError:
                pass
            else:
                raise AssertionError("direct Zarr conversion overwrote existing output")
        print(f"PASS use_depth={use_depth}: frames=27, episodes=2")


def main():
    run_case(False)
    run_case(True)
    print("SMOKE direct LeRobot -> Zarr PASSED")


if __name__ == "__main__":
    main()
