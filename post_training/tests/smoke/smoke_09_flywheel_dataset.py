"""Batch 05 integration: Teleop NPY + Data Wheel Rollout NPY -> Zarr -> PPO chunks.

Uses synthetic teleop-like Processed NPY V2 and real Recorder/Converter output.
Run in the ARBiM environment (requires zarr, LeRobot, torch, numba, cv2).
"""
import pathlib
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "post_training" / "src", ROOT / "third_party" / "lerobot" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kuavo_deploy.src.eval.rollout_recorder import CAMERA_KEYS, RolloutRecorder
from post_rl.data import data_prepare as dp
from post_rl.data.offline_buffer import LazyJpegZarrArray, OfflineBuffer
from post_rl.data.offline_dataset import OfflineDataset
from post_rl.data.rollout_to_npy import convert_staging_to_npy


TELEOP_LENGTHS = (60, 64)
ROLLOUT_LENGTHS = (66, 67)
ALL_LENGTHS = TELEOP_LENGTHS + ROLLOUT_LENGTHS
EXPECTED_ENDS = np.cumsum(ALL_LENGTHS).tolist()
EXPECTED_REWARDS = [30, 60 + 30, 60 + 64 + 21, 60 + 64 + 66 + 21]


def _jpeg(value):
    bgr = np.full((16, 16, 3), value % 256, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", bgr)
    assert success
    return encoded.tobytes()


def _raw_obs(value):
    obs = {"observation.state": np.full((1, 16), value, dtype=np.float32)}
    for key in CAMERA_KEYS:
        obs[key] = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
    return obs


def _write_teleop_processed(path, chunk_size=17):
    """Make a realistic streamed *processed* demonstration fixture, not LeRobot."""
    header = {
        "__format__": dp.STREAM_NPY_FORMAT,
        "num_frames": sum(TELEOP_LENGTHS),
        "num_episodes": len(TELEOP_LENGTHS),
        "use_depth": False,
        "state_shape": (16,),
        "action_shape": (16,),
        "rgb_storage": "jpeg",
        "jpeg_quality": 95,
        "rgb_shapes": {key: (3, 16, 16) for key in CAMERA_KEYS},
        "rgb_dtypes": {key: "uint8" for key in CAMERA_KEYS},
        "reward_mode": "last_gripper_release_sparse",
    }
    with path.open("wb") as handle:
        np.save(handle, header, allow_pickle=True)
        chunk = {key: [] for key in ("agent_pos", "action", "rgb", "reward", "done", "timeout")}
        for episode, length in enumerate(TELEOP_LENGTHS, start=1):
            for step in range(length):
                terminal = step == length - 1
                state = np.full((16,), episode * 1000 + step, dtype=np.float32)
                action = np.full((16,), float(step), dtype=np.float32)
                chunk["agent_pos"].append(state)
                chunk["action"].append(action)
                chunk["rgb"].append({key: _jpeg(episode * 50 + step) for key in CAMERA_KEYS})
                chunk["reward"].append(float(step == 30))
                chunk["done"].append(terminal)
                chunk["timeout"].append(terminal)
                if len(chunk["action"]) == chunk_size:
                    np.save(handle, chunk, allow_pickle=True)
                    chunk = {key: [] for key in chunk}
        if chunk["action"]:
            np.save(handle, chunk, allow_pickle=True)


def _write_rollout_stage(folder, episode, length):
    recorder = RolloutRecorder(folder, episode=episode)
    recorder.start_episode(_raw_obs(episode * 1000))
    actions = np.zeros((length, 16), dtype=np.float32)
    actions[8:18, 7] = 1.0
    actions[12:21, 15] = 1.0
    for step, action in enumerate(actions):
        recorder.append_step(action, _raw_obs(episode * 1000 + step + 1))
    path = recorder.finish_episode(True)
    assert path is not None
    return path


def _make_sources(folder):
    teleop = folder / "teleop.npy"
    _write_teleop_processed(teleop)
    staging_dir = folder / "staging"
    for episode, length in zip((3, 4), ROLLOUT_LENGTHS, strict=True):
        _write_rollout_stage(staging_dir, episode, length)
    rollout = folder / "rollout.npy"
    result = convert_staging_to_npy([staging_dir], rollout, batch_size=19)
    assert result["num_frames"] == sum(ROLLOUT_LENGTHS)
    assert result["num_episodes"] == 2
    assert result["skipped_invalid"] == 0
    return teleop, rollout


def _config(teleop, rollout, output):
    return {
        "mode": "build_db",
        "use_depth": False,
        "stream_batch_size": 23,
        "jpeg_quality": 95,
        "overwrite": False,
        "zarr_output_path": str(output),
        "teleop_sources": [
            {"name": "teleop_main", "kind": "teleop_npy", "path": str(teleop)},
            {"name": "flywheel_01", "kind": "rollout_npy", "path": str(rollout)},
        ],
    }


class TestFlywheelDatasetIntegration(unittest.TestCase):
    def test_sources_zarr_buffer_and_non_crossing_action_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            teleop, rollout = _make_sources(root)
            info_teleop = dp._read_source_header(teleop, use_depth=False)
            info_rollout = dp._read_source_header(rollout, use_depth=False)
            dp._validate_source_shapes([info_teleop, info_rollout], use_depth=False)

            dest = root / "merged.zarr"
            result = dp.run_build_db(_config(teleop, rollout, dest))
            self.assertEqual(result["num_frames"], sum(ALL_LENGTHS))
            self.assertEqual(result["num_episodes"], 4)
            self.assertEqual(result["episode_ends"].tolist(), EXPECTED_ENDS)

            z = zarr.open_group(str(dest), mode="r")
            manifest = z.attrs["source_manifest"]
            self.assertEqual([entry["name"] for entry in manifest], ["teleop_main", "flywheel_01"])
            self.assertEqual([entry["kind"] for entry in manifest], ["teleop_npy", "rollout_npy"])
            self.assertEqual([entry["reward_mode"] for entry in manifest],
                             ["last_gripper_release_sparse"] * 2)

            d = z["data"]
            done = np.asarray(d["done"][:], dtype=bool).reshape(-1)
            np.testing.assert_array_equal(np.flatnonzero(done) + 1, EXPECTED_ENDS)
            reward = np.asarray(d["reward"][:], dtype=np.float32).reshape(-1)
            np.testing.assert_array_equal(np.flatnonzero(reward), EXPECTED_REWARDS)
            self.assertEqual(float(reward.sum()), 4.0)

            next_indices = np.asarray(d["next_index"][:], dtype=np.int64)
            for start, end in zip((0, *EXPECTED_ENDS[:-1]), EXPECTED_ENDS, strict=True):
                np.testing.assert_array_equal(next_indices[start:end - 1],
                                              np.arange(start + 1, end, dtype=np.int64))
                self.assertEqual(int(next_indices[end - 1]), end - 1)
                self.assertEqual(float(d["return"][end - 1, 0]), 0.0)

            buffer = OfflineBuffer(device=torch.device("cpu"), gamma=0.99, use_depth=False)
            buffer.load_zarr(str(dest))
            self.assertEqual(len(buffer), sum(ALL_LENGTHS))
            np.testing.assert_array_equal(buffer.episode_ends, EXPECTED_ENDS)
            np.testing.assert_allclose(buffer["reward"].reshape(-1), reward)
            self.assertIsInstance(buffer["head_rgb"], LazyJpegZarrArray)
            self.assertEqual(buffer["head_rgb"][0].shape, (3, 16, 16))
            self.assertEqual(buffer["next_head_rgb"][EXPECTED_ENDS[0] - 1].shape,
                             (3, 16, 16))
            np.testing.assert_array_equal(buffer["head_rgb"][EXPECTED_ENDS[0] - 1],
                                          buffer["next_head_rgb"][EXPECTED_ENDS[0] - 1])

            # Both Dataset and SequenceSampler operate on one episode at a time.
            dataset = OfflineDataset(buffer, horizon=50, pad_before=0, pad_after=0,
                                     val_ratio=0.0, use_depth=False)
            self.assertEqual(len(dataset), sum(length - 50 + 1 for length in ALL_LENGTHS))
            for row in dataset.sampler.indices:
                start, stop, left_padding, right_padding = map(int, row)
                self.assertEqual((left_padding, right_padding), (0, 50))
                self.assertEqual(stop - start, 50)
                self.assertTrue(any(ep_start <= start < stop <= ep_end
                                    for ep_start, ep_end in zip(
                                        (0, *EXPECTED_ENDS[:-1]), EXPECTED_ENDS,
                                        strict=True)))

            for sample_id in (0, len(dataset) // 2, len(dataset) - 1):
                batch = dataset[sample_id]
                self.assertEqual(tuple(batch["action"].shape), (50, 16))
                self.assertEqual(tuple(batch["reward"].shape), (50, 1))
                self.assertEqual(tuple(batch["obs"]["state"].shape), (50, 16))
                self.assertEqual(tuple(batch["obs"]["head_rgb"].shape), (50, 3, 16, 16))
                self.assertEqual(tuple(batch["next_obs"]["state"].shape), (50, 16))

    def test_unknown_source_kind_rejected_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            teleop, rollout = _make_sources(root)
            output = root / "reject.zarr"
            config = _config(teleop, rollout, output)
            config["teleop_sources"][1]["kind"] = "invalid"
            with self.assertRaisesRegex(ValueError, "unknown processed NPY source kind"):
                dp.run_build_db(config)
            self.assertFalse(output.exists())

    def test_kind_defaults_to_teleop_for_legacy_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            teleop = root / "teleop.npy"
            _write_teleop_processed(teleop)
            dest = root / "legacy.zarr"
            cfg = {
                "zarr_output_path": str(dest),
                "teleop_sources": [{"name": "teleop_main", "path": str(teleop)}],
                "use_depth": False, "overwrite": False, "stream_batch_size": 19,
            }
            result = dp.run_build_db(cfg)
            self.assertEqual(result["num_episodes"], 2)
            z = zarr.open_group(str(dest), mode="r")
            self.assertEqual(z.attrs["source_manifest"][0]["kind"], "teleop_npy")


if __name__ == "__main__":
    unittest.main()
