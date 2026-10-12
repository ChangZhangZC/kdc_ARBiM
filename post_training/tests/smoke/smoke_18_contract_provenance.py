"""Final contract: propagate Reward V2 through mixed and repeated LeRobot merges."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "post_training" / "src", ROOT / "third_party" / "lerobot" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from post_rl.data import data_prepare as dp
from smoke_12_lerobot_to_zarr import FakeLeRobot


def test_mixed_provenance():
    with tempfile.TemporaryDirectory(prefix="arbim_merged_reward_") as directory:
        root = Path(directory)
        (root / "meta").mkdir()
        manifest = {
            "format": "arbim_lerobot_merge_provenance_v1",
            "episodes": [
                {"output_episode_index": 0, "frames": 14, "source_root": "teleop"},
                {"output_episode_index": 1, "frames": 13, "source_root": "flywheel",
                 "staging_reward": {
                     "reward_status": "valid",
                     "reward_mode": "last_gripper_release_sparse",
                     "reward_index": 8,
                     "reward_event": {"frame_offset": 8},
                 }},
            ],
        }
        (root / "meta" / "arbim_merge_provenance.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        ds = FakeLeRobot()
        episodes = dp._group_episode_indices(ds)
        kwargs = {
            "reward_source": "auto",
            "reward_open_infer_frames": 3,
            "reward_min_dwell": 3,
            "reward_expected_cycles": 1,
        }
        offsets, info = dp._staging_reward_offsets(root, episodes, kwargs, ds)
        assert offsets == {0: 9, 1: 8}, offsets
        assert info["source"] == "mixed_lerobot_reward_v2"
        assert info["recorded_count"] == 1 and info["computed_count"] == 1
        assert info["episodes"][0]["source"] == "computed"
        assert info["episodes"][1]["source"] == "recorded"

        # Explicit recorded mode must fail for a mixed dataset.
        try:
            dp._staging_reward_offsets(root, episodes, {**kwargs, "reward_source": "recorded"}, ds)
        except ValueError as exc:
            assert "requires all episodes" in str(exc)
        else:
            raise AssertionError("recorded mode accepted an unannotated Teleop episode")

        # Invalid successful Staging remains suitable for IL, not for Reward V2 RL.
        manifest["episodes"][1]["staging_reward"]["reward_status"] = "invalid"
        (root / "meta" / "arbim_merge_provenance.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        try:
            dp._staging_reward_offsets(root, episodes, kwargs, ds)
        except ValueError as exc:
            assert "invalid recorded Reward" in str(exc)
        else:
            raise AssertionError("invalid recorded Reward entered RL Zarr")
        print("PASS: mixed provenance, reward source isolation, invalid Reward rejection")


def test_nested_provenance():
    with tempfile.TemporaryDirectory(prefix="arbim_nested_reward_") as directory:
        root = Path(directory)
        (root / "meta").mkdir()
        ds = FakeLeRobot()
        episodes = dp._group_episode_indices(ds)
        rows = [
            {"output_episode_index": 0, "frames": 14},
            {"output_episode_index": 1, "frames": 13, "staging_reward": {
                "reward_status": "valid",
                "reward_index": 8,
                "reward_event": {"frame_offset": 8},
            }},
        ]
        (root / "meta" / "arbim_merge_provenance.json").write_text(
            json.dumps({"format": "arbim_lerobot_merge_provenance_v1", "episodes": rows}),
            encoding="utf-8"
        )
        offsets, metadata = dp._staging_reward_offsets(root, episodes, {
            "reward_source": "auto", "reward_open_infer_frames": 3,
            "reward_min_dwell": 3,
        }, ds)
        assert offsets == {0: 9, 1: 8}
        assert metadata["recorded_count"] == 1
        print("PASS: nested Staging provenance accepted")


def main():
    test_mixed_provenance()
    test_nested_provenance()
    print("SMOKE 18 merged LeRobot Reward provenance PASSED")


if __name__ == "__main__":
    main()
