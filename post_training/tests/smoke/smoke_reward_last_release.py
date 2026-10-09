from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_RL_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
for path in (REPO_ROOT, POST_RL_SRC, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from post_rl.data import data_prepare as dp


class _FakeDataset:
    def __init__(self, action: np.ndarray):
        self._action = np.asarray(action, dtype=np.float32)
        self.hf_dataset = {"action": [row for row in self._action]}

    def __len__(self):
        return len(self._action)

    def __getitem__(self, index):
        return {"action": self._action[index]}


def _episode_action(length: int, left_close, right_close) -> np.ndarray:
    action = np.zeros((length, 16), dtype=np.float32)
    left_start, left_end = left_close
    right_start, right_end = right_close
    action[left_start:left_end, 7] = 1.0
    action[right_start:right_end, 15] = 1.0
    return action


def main() -> None:
    ep0 = _episode_action(
        14,
        left_close=(3, 7),
        right_close=(4, 9),
    )
    ep1 = _episode_action(
        13,
        left_close=(2, 6),
        right_close=(3, 8),
    )
    action = np.concatenate([ep0, ep1], axis=0)
    dataset = _FakeDataset(action)
    episodes = {
        0: list(range(0, len(ep0))),
        1: list(range(len(ep0), len(ep0) + len(ep1))),
    }

    offsets, metadata = dp._last_gripper_release_offsets(
        dataset,
        episodes,
        {
            "reward_left_gripper_index": 7,
            "reward_right_gripper_index": 15,
            "reward_left_open_side": "auto",
            "reward_right_open_side": "auto",
            "reward_open_infer_frames": 3,
            "reward_min_dwell": 3,
            "reward_expected_cycles": 1,
        },
    )

    assert offsets == {0: 9, 1: 8}, offsets
    assert metadata["left_thresholds"]["open_side"] == "low"
    assert metadata["right_thresholds"]["open_side"] == "low"

    for episode_id, frames in episodes.items():
        rewards = np.zeros(len(frames), dtype=np.float32)
        rewards[offsets[episode_id]] = 1.0
        assert rewards.sum() == 1.0
        assert rewards[-1] == 0.0
        assert offsets[episode_id] < len(frames) - 1

    print("SMOKE reward last-release PASSED")


if __name__ == "__main__":
    main()
