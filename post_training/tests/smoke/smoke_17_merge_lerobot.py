"""Batch 04: LeRobot episode concatenation without on-disk video rewrites in test."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "post_training" / "src", ROOT / "third_party" / "lerobot" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from post_rl.data import data_prepare as dp
from post_rl.data import lerobot_merge as merger
from smoke_12_lerobot_to_zarr import FakeLeRobot
from smoke_16_staging_to_lerobot import FakeLeRobot as FakeVideoWriter, FakeWriter, FakeMeta


class FakeReader(FakeLeRobot):
    def __init__(self, *, depth, repo_id):
        super().__init__(use_depth=depth)
        self.repo_id = repo_id
        self.meta = FakeMeta()
        self.meta.tasks = {0: "sort toy"}


def run_case(depth):
    with tempfile.TemporaryDirectory(prefix="arbim_lerobot_merge_") as name:
        root = pathlib.Path(name)
        sources = [root / "teleop", root / "flywheel"]
        for path in sources:
            (path / "meta").mkdir(parents=True)
        data = {
            str(sources[0].resolve()): FakeReader(depth=depth, repo_id="teleop"),
            str(sources[1].resolve()): FakeReader(depth=depth, repo_id="flywheel"),
        }
        # Mirror the schema written by staging_to_lerobot.py, not a partial
        # hand-written provenance file. The fake actions release at 9 and 8.
        staging_episodes = [
            {
                "lerobot_episode_index": i,
                "num_frames": length,
                "success": True,
                "reward_mode": "last_gripper_release_sparse",
                "reward_status": "valid",
                "reward_index": reward_idx,
                "reward_event": {"frame_offset": reward_idx},
            }
            for i, (length, reward_idx) in enumerate(((14, 9), (13, 8)))
        ]
        (sources[1] / "meta" / "arbim_staging_provenance.json").write_text(
            json.dumps({
                "format": "arbim_lerobot_staging_provenance_v1",
                "episodes": staging_episodes,
            }),
            encoding="utf-8",
        )
        FakeWriter.instances.clear()
        with (
            patch.object(dp, "load_lerobot_dataset", side_effect=lambda path: data[str(pathlib.Path(path).resolve())]),
            patch.object(merger, "LeRobotDataset", FakeVideoWriter),
        ):
            result = merger.merge_lerobot_datasets(
                sources, root / "combined", "lerobot/arbim_combined",
                use_depth=depth, batch_size=6, fallback_task="sort toy",
            )
        assert result["num_frames"] == 54
        assert result["num_episodes"] == 4
        writer = FakeWriter.instances[-1]
        assert type(writer.fps) is int and writer.fps == 10
        assert len(writer.episodes) == 4
        assert [len(ep) for ep in writer.episodes] == [14, 13, 14, 13]
        for ep in writer.episodes:
            assert all(frame["task"] == "sort toy" for frame in ep)
            assert all(frame["action"].shape == (16,) for frame in ep)
            assert all(frame["observation.state"].shape == (16,) for frame in ep)
            assert all(frame["observation.images.head_cam_h"].shape == (16, 16, 3) for frame in ep)
            if depth:
                for frame in ep:
                    assert frame["observation.depth_h"].dtype == np.uint8
                    assert frame["observation.depth_h"].shape == (16, 16, 3)
        metadata = json.loads(pathlib.Path(result["provenance"]).read_text(encoding="utf-8"))
        assert metadata["total_frames"] == 54
        assert [item["output_episode_index"] for item in metadata["episodes"]] == [0, 1, 2, 3]
        assert [item["source_episode_index"] for item in metadata["episodes"]] == [0, 1, 0, 1]
        assert "staging_reward" not in metadata["episodes"][0]
        assert metadata["episodes"][2]["staging_reward"] == staging_episodes[0]
        assert metadata["episodes"][3]["staging_reward"] == staging_episodes[1]
        assert sources[0].exists() and sources[1].exists()
        print(f"PASS merge LeRobot use_depth={depth}, episodes=4, frames=54")


def test_rejections():
    with tempfile.TemporaryDirectory(prefix="arbim_lerobot_bad_") as name:
        root = pathlib.Path(name)
        one, two = root / "one", root / "two"
        (one / "meta").mkdir(parents=True)
        (two / "meta").mkdir(parents=True)
        data = {
            str(one.resolve()): FakeReader(depth=False, repo_id="one"),
            str(two.resolve()): FakeReader(depth=False, repo_id="two"),
        }
        target = root / "merged"
        with (
            patch.object(dp, "load_lerobot_dataset", side_effect=lambda path: data[str(pathlib.Path(path).resolve())]),
            patch.object(merger, "LeRobotDataset", FakeVideoWriter),
        ):
            for input_list, output, keyword in [
                ([one], target, "at least two"),
                ([one, one], target, "duplicate"),
                ([one, two], one, "already exists"),
            ]:
                try:
                    merger.merge_lerobot_datasets(input_list, output, "lerobot/test")
                except (ValueError, FileExistsError) as exc:
                    assert keyword in str(exc), (keyword, str(exc))
                else:
                    raise AssertionError(f"Expected rejection for {keyword}")
            # PyAV cannot encode at a float FPS. Do not silently round 10.5.
            data[str(two.resolve())].meta.info["fps"] = 10.5
            try:
                merger.merge_lerobot_datasets([one, two], target, "lerobot/test")
            except ValueError as exc:
                assert "FPS must be an integer" in str(exc), str(exc)
            else:
                raise AssertionError("Fractional FPS reached video writer")
        assert not target.exists()
        print("PASS LeRobot merge invalid inputs")


def main():
    run_case(False)
    run_case(True)
    test_rejections()
    print("SMOKE 17 merge LeRobot PASSED")


if __name__ == "__main__":
    main()
