"""Standalone smoke tests for successful rollout capture and Reward V2 (no ROS)."""

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
    _annotate_reward_v2,
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
            self.assertEqual(data["reward_status"], "invalid")
            self.assertIsNone(data["reward"])
            self.assertIn("release", data["reward_error"])
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


    @staticmethod
    def _valid_actions():
        # Grippers start open (0), close (1), then release (0).
        # Left release at t=6; right release at t=8.
        action = np.zeros((14, 16), dtype=np.float32)
        action[3:6, 7] = 1.0
        action[4:8, 15] = 1.0
        return action

    def test_reward_v2_exactly_one_sparse_success_event(self):
        actions = self._valid_actions()
        reward, event = _annotate_reward_v2(actions)
        self.assertEqual(reward.shape, (14,))
        self.assertEqual(np.flatnonzero(reward).tolist(), [8])
        self.assertEqual(float(reward.sum()), 1.0)
        self.assertEqual(event["left_release"], 6)
        self.assertEqual(event["right_release"], 8)
        self.assertEqual(event["frame_offset"], 8)

        with tempfile.TemporaryDirectory() as tmp:
            recorder = RolloutRecorder(tmp, episode=7)
            recorder.start_episode(observation(0))
            for i, command in enumerate(actions):
                recorder.append_step(command, observation(i + 1))
            path = recorder.finish_episode(True)
            payload = np.load(path, allow_pickle=True).item()
            self.assertEqual(payload["reward_status"], "valid")
            self.assertIsNone(payload["reward_error"])
            self.assertEqual(payload["reward_mode"], "last_gripper_release_sparse")
            self.assertEqual(np.flatnonzero(payload["reward"]).tolist(), [8])
            self.assertEqual(payload["num_steps"], len(payload["reward"]))
            self.assertEqual(len(payload["observations"]), len(payload["reward"]) + 1)

    def test_invalid_reward_keeps_successful_staging_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RolloutRecorder(tmp, episode=8)
            recorder.start_episode(observation(0))
            for i in range(6):
                recorder.append_step(np.zeros(16), observation(i + 1))
            path = recorder.finish_episode(True)
            payload = np.load(path, allow_pickle=True).item()
            self.assertTrue(payload["success"])
            self.assertEqual(payload["reward_status"], "invalid")
            self.assertIsNone(payload["reward"])
            self.assertIsNone(payload["reward_event"])
            self.assertTrue(payload["reward_error"])

    def test_second_gripper_release_cycle_is_rejected(self):
        actions = self._valid_actions()
        # Extend left gripper with a second closed-to-open cycle.
        extra = np.zeros((10, 16), dtype=np.float32)
        extra[3:6, 7] = 1.0
        double_release = np.concatenate((actions, extra), axis=0)
        with self.assertRaisesRegex(ValueError, "one release per gripper"):
            _annotate_reward_v2(double_release)

    def test_reward_not_added_to_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RolloutRecorder(tmp, episode=9)
            recorder.start_episode(observation(0))
            for i, action in enumerate(self._valid_actions()):
                recorder.append_step(action, observation(i + 1))
            self.assertIsNone(recorder.finish_episode(False))
            self.assertEqual(list(Path(tmp).iterdir()), [])



if __name__ == "__main__":
    unittest.main()
