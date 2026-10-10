"""Standalone smoke tests for Batch 02 rollout staging (no ROS required)."""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kuavo_deploy.src.eval.rollout_recorder import (
    CAMERA_KEYS,
    STAGING_FORMAT,
    RolloutRecorder,
)


def observation(value):
    data = {
        "observation.state": np.full((1, 16), value, dtype=np.float32)
    }
    for key in CAMERA_KEYS:
        image = np.zeros((1, 3, 16, 16), dtype=np.float32)
        image[:, 0] = value / 10
        image[:, 1] = 0.4
        image[:, 2] = 0.8
        data[key] = image
    return data


class TestRolloutRecorder(unittest.TestCase):
    def test_success_persists_full_transitions_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RolloutRecorder(tmp, episode=2)
            first = observation(1)
            second = observation(2)
            last = observation(3)
            recorder.start_episode(first)
            first["observation.state"][:] = 99  # Simulate in-place preprocessing.
            action0 = np.arange(16, dtype=np.float32)
            recorder.append_step(action0, second)
            action0[:] = -42
            recorder.append_step(np.ones(16, dtype=np.float32), last)
            path = recorder.finish_episode(True)
            self.assertTrue(path.is_file())
            data = np.load(path, allow_pickle=True).item()
            self.assertEqual(data["__format__"], STAGING_FORMAT)
            self.assertEqual(data["num_steps"], 2)
            self.assertEqual(data["actions"].shape, (2, 16))
            self.assertEqual(len(data["observations"]), 3)
            self.assertEqual(data["observations"][0]["agent_pos"][0], 1)
            self.assertEqual(data["observations"][1]["agent_pos"][0], 2)
            self.assertEqual(data["observations"][2]["agent_pos"][0], 3)
            self.assertEqual(data["actions"][0, 0], 0)
            self.assertEqual(data["actions"][0, 15], 15)
            for sample in data["observations"]:
                self.assertEqual(set(sample["rgb"]), set(CAMERA_KEYS))
                for jpeg in sample["rgb"].values():
                    decoded = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    self.assertEqual(decoded.shape, (16, 16, 3))

    def test_failure_does_not_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RolloutRecorder(tmp, episode=3)
            recorder.start_episode(observation(0))
            recorder.append_step(np.zeros(16), observation(1))
            self.assertIsNone(recorder.finish_episode(False))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_abort_clears_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RolloutRecorder(tmp, episode=4)
            recorder.start_episode(observation(0))
            recorder.abort()
            recorder.start_episode(observation(1))
            recorder.append_step(np.ones(16), observation(2))
            path = recorder.finish_episode(True)
            data = np.load(path, allow_pickle=True).item()
            self.assertEqual(data["num_steps"], 1)

    def test_action_dimension_change_rejected(self):
        recorder = RolloutRecorder("unused", episode=5)
        recorder.start_episode(observation(0))
        recorder.append_step(np.ones(16), observation(1))
        with self.assertRaises(ValueError):
            recorder.append_step(np.ones(14), observation(2))
        recorder.abort()


if __name__ == "__main__":
    unittest.main()
