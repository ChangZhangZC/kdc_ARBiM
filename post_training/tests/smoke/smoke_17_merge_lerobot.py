"""Batch 04: fast official LeRobot merge, full Depth and Reward V2 provenance."""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
for folder in (ROOT, ROOT / "post_training" / "src", ROOT / "third_party" / "lerobot" / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from post_rl.data import data_prepare as dp
from post_rl.data import lerobot_merge as merger
from smoke_12_lerobot_to_zarr import FakeLeRobot
from smoke_16_staging_to_lerobot import FakeMeta


class FakeReader(FakeLeRobot):
    def __init__(self, *, depth=True, repo_id="test", combined=False):
        super().__init__(use_depth=depth)
        self.repo_id = repo_id
        self.meta = FakeMeta()
        if not depth:
            for key in dp.DEPTH_FEATURE_TO_BUFFER:
                del self.meta.features[key]
        self.meta.tasks = pd.DataFrame({"task_index": [0]}, index=["sort toy"])
        if combined:
            original = self.frames
            self.frames = [
                {**frame, "episode_index": ep + 2}
                for ep in range(2)
                for frame in original
                if frame["episode_index"] == ep
            ]
            self.frames = original + self.frames
            self.episodes = [14, 13, 14, 13]
        self.hf_dataset = [{"task_index": np.int64(0)} for _ in self.frames]

    @property
    def root(self):
        return pathlib.Path(self.repo_id)


def make_sources(root, *, depth=True):
    paths = [root / "teleop", root / "wheel"]
    for path in paths:
        (path / "meta").mkdir(parents=True)
    datasets = {str(path.resolve()): FakeReader(depth=depth, repo_id=path.name) for path in paths}
    return paths, datasets


def write_staging_provenance(path):
    records = [
        {
            "lerobot_episode_index": i,
            "num_frames": length,
            "success": True,
            "reward_mode": "last_gripper_release_sparse",
            "reward_status": "valid",
            "reward_index": offset,
            "reward_event": {"frame_offset": offset},
        }
        for i, (length, offset) in enumerate(((14, 9), (13, 8)))
    ]
    (path / "meta" / "arbim_staging_provenance.json").write_text(
        json.dumps({"format": merger.STAGING_FORMAT, "episodes": records}), encoding="utf-8"
    )
    return records


# Builtin len() on SimpleNamespace does not use an instance-level __len__.
class MergedFake:
    def __init__(self, datasets):
        self.meta = SimpleNamespace(
            total_episodes=sum(len(ds.episodes) for ds in datasets),
            features=copy.deepcopy(datasets[0].meta.features),
        )
        self.length = sum(len(ds) for ds in datasets)

    def __len__(self):
        return self.length


def run_case(depth=True):
    with tempfile.TemporaryDirectory(prefix="arbim_official_merge_") as name:
        root = pathlib.Path(name)
        paths, datasets = make_sources(root, depth=depth)
        stage_rows = write_staging_provenance(paths[1])
        called = []

        def fake_official(*, datasets, output_repo_id, output_dir):
            called.append(output_repo_id)
            target = pathlib.Path(output_dir)
            target.mkdir()
            (target / "meta").mkdir()
            return MergedFake(datasets)

        with (
            patch.object(dp, "load_lerobot_dataset", side_effect=lambda path: datasets[str(pathlib.Path(path).resolve())]),
            patch.object(merger, "merge_datasets", side_effect=fake_official),
            patch.object(dp, "_frame_arrays", side_effect=AssertionError("Decoded video unexpectedly")),
            patch.object(dp, "_load_lerobot_frame_batch", side_effect=AssertionError("Decoded video unexpectedly")),
        ):
            output = root / "combined"
            result = merger.merge_lerobot_datasets(paths, output, "arbim/combined", use_depth=depth)
        assert called == ["arbim/combined"]
        assert result["num_frames"] == 54 and result["num_episodes"] == 4
        payload = json.loads(pathlib.Path(result["provenance"]).read_text())
        assert payload["merge_backend"] == "lerobot.dataset_tools.merge_datasets+v0.4.2_shard_fix"
        assert payload["use_depth"] == depth
        assert [r["output_episode_index"] for r in payload["episodes"]] == [0, 1, 2, 3]
        assert "staging_reward" not in payload["episodes"][0]
        assert payload["episodes"][2]["staging_reward"] == stage_rows[0]
        assert payload["episodes"][3]["staging_reward"] == stage_rows[1]
        print(f"PASS official LeRobot merge Depth={depth}, 4 episodes, no decode")


def test_task_metadata_mapping():
    reader = FakeReader(depth=True)
    reader.meta.tasks = pd.DataFrame({"task_index": [4, 2]}, index=["sort toy", "put away"])
    reader.hf_dataset[0] = {"task_index": np.int64(2)}
    assert merger._episode_task(reader, 0, "fallback") == "put away"
    reader.meta.tasks = {2: "legacy task"}
    assert merger._episode_task(reader, 0, "fallback") == "legacy task"
    reader.meta.tasks = pd.DataFrame({"task_index": [0]}, index=["unused"])
    assert merger._episode_task(reader, 0, "fallback") == "fallback"


def test_rejections_and_mismatch():
    with tempfile.TemporaryDirectory(prefix="arbim_bad_archive_") as name:
        root = pathlib.Path(name)
        paths, datasets = make_sources(root, depth=True)
        target = root / "combined"
        with patch.object(dp, "load_lerobot_dataset", side_effect=lambda p: datasets[str(pathlib.Path(p).resolve())]):
            for inputs, output, fragment in (
                ([paths[0]], target, "at least two"),
                ([paths[0], paths[0]], target, "duplicate"),
                (paths, paths[0], "already exists"),
            ):
                try:
                    merger.merge_lerobot_datasets(inputs, output, "arbim/invalid")
                except (FileExistsError, ValueError) as exc:
                    assert fragment in str(exc), str(exc)
                else:
                    raise AssertionError(f"Expected {fragment!r} rejection")
            dataset = datasets[str(paths[1].resolve())]
            for key in dp.DEPTH_FEATURE_TO_BUFFER:
                del dataset.meta.features[key]
            try:
                merger.merge_lerobot_datasets(paths, target, "arbim/invalid")
            except ValueError as exc:
                assert "missing required LeRobot feature" in str(exc)
            else:
                raise AssertionError("Missing archived Depth was accepted")
        assert not target.exists()
        print("PASS full Depth and invalid-input rejections")


def test_repeated_merge_provenance():
    with tempfile.TemporaryDirectory(prefix="arbim_merge_roundtrip_") as name:
        root = pathlib.Path(name)
        paths, datasets = make_sources(root, depth=True)
        staging_rows = write_staging_provenance(paths[1])
        merged_path = root / "merged"
        next_path = root / "next"
        (merged_path / "meta").mkdir(parents=True)
        (next_path / "meta").mkdir(parents=True)
        combined = FakeReader(depth=True, repo_id="merged", combined=True)
        new_rollout = FakeReader(depth=True, repo_id="next")
        datasets[str(merged_path.resolve())] = combined
        datasets[str(next_path.resolve())] = new_rollout

        def fake_official(*, datasets, output_repo_id, output_dir):
            target = pathlib.Path(output_dir)
            target.mkdir()
            (target / "meta").mkdir()
            return MergedFake(datasets)

        with (
            patch.object(dp, "load_lerobot_dataset", side_effect=lambda p: datasets[str(pathlib.Path(p).resolve())]),
            patch.object(merger, "merge_datasets", side_effect=fake_official),
        ):
            first_output = root / "round1"
            first = merger.merge_lerobot_datasets(paths, first_output, "arbim/round1")
            datasets[str(first_output.resolve())] = combined
            second = merger.merge_lerobot_datasets(
                [first_output, next_path], root / "round2", "arbim/round2"
            )
        assert first["num_episodes"] == 4 and second["num_episodes"] == 6
        assert second["num_frames"] == 81
        manifest = json.loads(pathlib.Path(second["provenance"]).read_text())
        assert manifest["episodes"][2]["staging_reward"] == staging_rows[0]
        assert manifest["episodes"][3]["staging_reward"] == staging_rows[1]
        print("PASS recursively merged Reward V2 provenance")


def main():
    test_task_metadata_mapping()
    run_case(True)
    run_case(False)  # explicit compatibility for old RGB-only archives
    test_rejections_and_mismatch()
    test_repeated_merge_provenance()
    print("SMOKE 17 official LeRobot merge PASSED")


if __name__ == "__main__":
    main()
