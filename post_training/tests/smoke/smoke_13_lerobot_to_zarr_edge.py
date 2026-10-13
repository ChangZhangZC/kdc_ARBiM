"""Batch 01: edge cases for direct LeRobot -> Zarr, no real dataset required."""
from __future__ import annotations

import pathlib
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import zarr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
for folder in (REPO_ROOT, REPO_ROOT / "post_training" / "src", REPO_ROOT / "third_party" / "lerobot" / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from post_rl.data import data_prepare as dp
from smoke_12_lerobot_to_zarr import FakeLeRobot


def cfg(tmp, **overrides):
    settings = {
        "lerobot_root": str(tmp / "synthetic_lerobot"),
        "zarr_output_path": str(tmp / "result.zarr"),
        "stream_batch_size": 5,
        "jpeg_quality": 95,
        "use_depth": False,
        "reward_mode": "last_gripper_release_sparse",
        "reward_open_infer_frames": 3,
        "reward_min_dwell": 3,
        "reward_expected_cycles": 1,
    }
    settings.update(overrides)
    return settings


def expect_error(error_type, callback, contains):
    try:
        callback()
    except error_type as exc:
        assert contains in str(exc), (contains, str(exc))
    else:
        raise AssertionError(f"Expected {error_type.__name__}: {contains}")


def test_shaped_rewards():
    # This is the legacy mode; it must remain identical to the old formula.
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        ds = FakeLeRobot()
        config = cfg(
            tmp, reward_mode="episode_end_shaped", smooth_penalty=0.01,
            lambda_penalty=0.05, max_episode_len=2000,
        )
        with patch.object(dp, "load_lerobot_dataset", return_value=ds):
            dp.run_lerobot_to_zarr(config)
        root = zarr.open_group(config["zarr_output_path"], mode="r")
        rewards = np.asarray(root["data"]["reward"][:]).reshape(-1)
        action = np.stack([frame["action"] for frame in ds.frames])
        expected = np.zeros_like(rewards)
        ends = [14, 27]
        start = 0
        for end in ends:
            for index in range(start, end):
                terminal = index == end - 1
                expected[index] = float(terminal)
                if terminal:
                    expected[index] -= 0.05 * (end - start) / 2000
                if index > start:
                    expected[index] -= 0.01 * np.linalg.norm(
                        action[index] - action[index - 1]
                    )
            start = end
        np.testing.assert_allclose(rewards, expected, rtol=1e-6, atol=1e-7)
        assert root.attrs["reward_mode"] == "episode_end_shaped"
        assert "reward_event" not in root.attrs
        assert root["meta"]["episode_ends"][:].tolist() == [14, 27]
        print("PASS: shaped reward, terminal and source metadata")


def test_bad_rewards_and_depth():
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        ds = FakeLeRobot()
        for item in ds.frames:
            item["action"] = np.zeros(16, dtype=np.float32)
        config = cfg(tmp)
        with patch.object(dp, "load_lerobot_dataset", return_value=ds):
            expect_error(
                ValueError, lambda: dp.run_lerobot_to_zarr(config),
                "dynamic range",
            )
        assert not pathlib.Path(config["zarr_output_path"]).exists()
        print("PASS: invalid reward fails before creating output")
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        config = cfg(tmp, use_depth=True)
        with patch.object(dp, "load_lerobot_dataset", return_value=FakeLeRobot()):
            expect_error(
                KeyError, lambda: dp.run_lerobot_to_zarr(config),
                "observation.depth",
            )
        assert not pathlib.Path(config["zarr_output_path"]).exists()
        print("PASS: incomplete depth rejected")


def test_invalid_config_and_no_overwrite():
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        ds = FakeLeRobot()
        bad = [
            ({"stream_batch_size": 0}, "stream_batch_size"),
            ({"jpeg_quality": 101}, "jpeg_quality"),
            ({"reward_mode": "unknown"}, "reward_mode"),
            ({"max_episode_len": 0}, "max_episode_len"),
        ]
        with patch.object(dp, "load_lerobot_dataset", return_value=ds):
            for overrides, msg in bad:
                expect_error(
                    ValueError, lambda overrides=overrides: dp.run_lerobot_to_zarr(
                        cfg(tmp, **overrides)
                    ), msg
                )
            config = cfg(tmp)
            dp.run_lerobot_to_zarr(config)
            # Omitted overwrite means "no overwrite" in new mode.
            expect_error(
                FileExistsError,
                lambda: dp.run_lerobot_to_zarr(config),
                "already exists",
            )
        print("PASS: config validation and default no-overwrite")


def test_returns_and_next_indices():
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        config = cfg(tmp)
        with patch.object(dp, "load_lerobot_dataset", return_value=FakeLeRobot()):
            dp.run_lerobot_to_zarr(config)
        root = zarr.open_group(config["zarr_output_path"], mode="r")
        group = root["data"]
        reward = np.asarray(group["reward"][:]).reshape(-1)
        returns = np.asarray(group["return"][:]).reshape(-1)
        done = np.asarray(group["done"][:]).reshape(-1)
        next_indices = np.asarray(group["next_index"][:]).reshape(-1)
        expected_returns = np.empty_like(returns)
        running = 0.0
        for i in reversed(range(len(reward))):
            running = float(reward[i]) + 0.99 * running * (not done[i])
            expected_returns[i] = running
        np.testing.assert_allclose(returns, expected_returns, rtol=1e-6, atol=1e-6)
        assert np.flatnonzero(done).tolist() == [13, 26]
        np.testing.assert_array_equal(next_indices[:-1][~done[:-1]], np.arange(1, 27)[~done[:-1]])
        assert next_indices[13] == 13 and next_indices[26] == 26
        assert root.attrs["source_manifest"][0]["kind"] == "teleop_lerobot"
        print("PASS: discounted returns and next indices")


def main():
    test_shaped_rewards()
    test_bad_rewards_and_depth()
    test_invalid_config_and_no_overwrite()
    test_returns_and_next_indices()
    print("SMOKE direct LeRobot -> Zarr edge cases PASSED")


if __name__ == "__main__":
    main()
