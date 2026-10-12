"""Batch 03: Data Wheel Staging -> LeRobot feature/video contract and provenance."""
from __future__ import annotations

import pathlib
import sys
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "post_training" / "src", ROOT / "third_party" / "lerobot" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from kuavo_deploy.src.eval.rollout_recorder import CAMERA_KEYS, DEPTH_KEYS
from post_rl.data import data_prepare as dp
from post_rl.data import staging_to_lerobot as converter
from smoke_08_flywheel_npy import save_rollout


class FakeMeta:
    def __init__(self):
        self.features = {
            "observation.state": {"dtype": "float32", "shape": (16,), "names": {"state_names": [str(i) for i in range(16)]}},
            "action": {"dtype": "float32", "shape": (16,), "names": {"action_names": [str(i) for i in range(16)]}},
            **{key: {"dtype": "video", "shape": (3, 16, 16), "names": ["channels", "height", "width"]}
               for key in (*CAMERA_KEYS, *DEPTH_KEYS)},
        }
        self.info = {"fps": 10, "robot_type": "kuavo4pro"}


class FakeWriter:
    instances = []

    def __init__(self, root, features, fps, robot_type, **kwargs):
        self.path = pathlib.Path(root)
        self.path.mkdir(parents=True)
        (self.path / "meta").mkdir()
        self.meta = FakeMeta()
        self.features = features
        self.fps = fps
        self.robot_type = robot_type
        self.episodes = []
        self.current = []
        self.hf_dataset = None
        self.__class__.instances.append(self)

    def add_frame(self, frame):
        # Match the real LeRobotDataset.add_frame(frame) API: task belongs
        # inside the frame dict, not in a separate keyword argument.
        assert frame["task"] == "sort toy"
        self.current.append(frame)

    def save_episode(self):
        self.episodes.append(self.current)
        self.current = []

    def create_hf_dataset(self):
        return []


class FakeLeRobot:
    def __init__(self, repo_id, root):
        self.meta = FakeMeta()

    @staticmethod
    def create(**kwargs):
        return FakeWriter(**kwargs)


def create_sources(root, *, depth):
    stage = root / "staging"
    first = save_rollout(stage, 1, use_depth=depth)
    invalid = save_rollout(stage, 2, valid=False, use_depth=depth)
    second = save_rollout(stage, 3, use_depth=depth)
    template = root / "template"
    (template / "meta").mkdir(parents=True)
    return stage, template, first, invalid, second


def test_convert(depth):
    with tempfile.TemporaryDirectory(prefix="arbim_staging_lerobot_") as tmp:
        root = pathlib.Path(tmp)
        stage, template, first, invalid, second = create_sources(root, depth=depth)
        FakeWriter.instances.clear()
        with patch.object(converter, "LeRobotDataset", FakeLeRobot):
            result = converter.convert_staging_to_lerobot(
                [stage], template, root / "il_dataset", "lerobot/arbim_il",
                "sort toy", use_depth=depth,
            )
        assert result["num_episodes"] == 3
        assert result["num_frames"] == 35  # 14 + 7 + 14
        writer = FakeWriter.instances[-1]
        assert writer.fps == 10 and writer.robot_type == "kuavo4pro"
        assert len(writer.episodes) == 3
        assert len(writer.episodes[0]) == 14
        assert len(writer.episodes[1]) == 7
        assert len(writer.episodes[2]) == 14
        for frame in writer.episodes[0]:
            assert frame["observation.state"].shape == (16,)
            assert frame["action"].shape == (16,)
            for key in CAMERA_KEYS:
                assert frame[key].shape == (16, 16, 3)
                assert frame[key].dtype == np.uint8
            if depth:
                for key in DEPTH_KEYS:
                    assert frame[key].shape == (16, 16, 3)
                    assert frame[key].dtype == np.uint8
            else:
                assert not any(key in frame for key in DEPTH_KEYS)
        assert len(writer.features) == (8 if depth else 5)
        assert writer.current == []
        # IL includes reward-invalid successful episodes, retaining provenance.
        import json
        provenance = json.loads(pathlib.Path(result["provenance"]).read_text())
        assert [ep["reward_status"] for ep in provenance["episodes"]] == [
            "valid", "invalid", "valid"
        ]
        with patch.object(converter, "LeRobotDataset", FakeLeRobot):
            result_rl = converter.convert_staging_to_lerobot(
                [stage], template, root / "rl_dataset", "lerobot/arbim_rl",
                "sort toy", use_depth=depth, require_valid_reward=True,
            )
        assert result_rl["num_episodes"] == 2
        assert result_rl["skipped_invalid"] == 1
        from collections import OrderedDict
        episodes = OrderedDict([(0, list(range(14))), (1, list(range(14)))])
        offsets, meta = dp._staging_reward_offsets(
            result_rl["lerobot_root"], episodes,
            {"reward_source": "recorded"}
        )
        # The test rollout's later (right) gripper release begins at index 8.
        # Verify both the known event and byte-for-byte provenance preservation.
        expected_offsets = {
            0: int(np.load(first, allow_pickle=True).item()["reward_event"]["frame_offset"]),
            1: int(np.load(second, allow_pickle=True).item()["reward_event"]["frame_offset"]),
        }
        assert expected_offsets == {0: 8, 1: 8}
        assert offsets == expected_offsets
        assert meta["source"] == "staging_recorded_v1"
        with patch.object(converter, "LeRobotDataset", FakeLeRobot):
            try:
                converter.convert_staging_to_lerobot(
                    [stage], template, root / "rl_dataset",
                    "lerobot/arbim_rl", "sort toy", use_depth=depth,
                )
            except FileExistsError:
                pass
            else:
                raise AssertionError("existing LeRobot dataset must not be overwritten")
        assert first.exists() and invalid.exists() and second.exists()
        print(f"PASS staging to LeRobot use_depth={depth}, IL=3, RL=2")


def main():
    test_convert(False)
    test_convert(True)
    print("SMOKE 16 Staging -> LeRobot PASSED")


if __name__ == "__main__":
    main()
