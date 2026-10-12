"""Regression for v0.4.2 LeRobot file-level merge video-shard index corruption.

Exercises two different source video files copied to separate destination files
and verifies episode metadata points to the right file at local timestamps.
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT, ROOT / "post_training" / "src", ROOT / "third_party" / "lerobot" / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from lerobot.datasets.utils import DEFAULT_VIDEO_PATH
from post_rl.data import lerobot_v042_fast_merge as compat


CAM = "observation.images.head_cam_h"


def video_path(root, chunk, file):
    return Path(root) / DEFAULT_VIDEO_PATH.format(
        video_key=CAM, chunk_index=chunk, file_index=file,
    )


def test_shards_and_local_timestamps():
    with tempfile.TemporaryDirectory(prefix="arbim_video_shards_") as temp:
        base = Path(temp)
        source = base / "src"
        destination = base / "dst"
        for file in (0, 1):
            path = video_path(source, 0, file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"source shard {file}".encode("utf-8"))

        source_meta = SimpleNamespace(
            root=source,
            episodes={
                f"videos/{CAM}/chunk_index": [0, 0],
                f"videos/{CAM}/file_index": [0, 1],
            },
        )
        dest_meta = SimpleNamespace(
            root=destination,
            info={"total_episodes": 600, "total_frames": 90144},
        )
        states = {CAM: {
            "chunk": 0, "file": 0, "latest_duration": 9014.4, "episode_duration": 0,
        }}
        with patch.object(compat.aggregate, "get_video_duration_in_s", return_value=12.0):
            result = compat._aggregate_videos(source_meta, dest_meta, states, 100, 1000)

        mapping = result[CAM]["src_to_dst"]
        assert mapping == {(0, 0): (0, 0), (0, 1): (0, 1)}, mapping
        assert result[CAM]["src_to_offset"] == {(0, 0): 0.0, (0, 1): 0.0}
        assert video_path(destination, 0, 0).read_bytes() == b"source shard 0"
        assert video_path(destination, 0, 1).read_bytes() == b"source shard 1"

        original = pd.DataFrame({
            "episode_index": [0, 1],
            "meta/episodes/chunk_index": [0, 0],
            "meta/episodes/file_index": [0, 0],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 1],
            f"videos/{CAM}/chunk_index": [0, 0],
            f"videos/{CAM}/file_index": [0, 1],
            f"videos/{CAM}/from_timestamp": [0.0, 0.0],
            f"videos/{CAM}/to_timestamp": [10.0, 12.0],
            "dataset_from_index": [0, 100],
            "dataset_to_index": [100, 220],
        })
        patched = compat._update_meta_data(
            original, dest_meta,
            meta_idx={"chunk": 0, "file": 0},
            data_idx={"src_to_dst": {(0, 0): (0, 4), (0, 1): (0, 5)}},
            videos_idx=result,
        )
        assert patched["episode_index"].tolist() == [600, 601]
        assert patched["dataset_from_index"].tolist() == [90144, 90244]
        assert patched["data/file_index"].tolist() == [4, 5]
        assert patched[f"videos/{CAM}/file_index"].tolist() == [0, 1]
        assert patched[f"videos/{CAM}/from_timestamp"].tolist() == [0.0, 0.0]
        assert patched[f"videos/{CAM}/to_timestamp"].tolist() == [10.0, 12.0]
        assert original["data/file_index"].tolist() == [0, 1]
    print("PASS multi-video file shard mappings and timestamp offsets")


def test_scoped_compat_is_restored():
    # The official orchestration is used, but patched only inside the call.
    original_video = compat.aggregate.aggregate_videos
    original_data = compat.aggregate.aggregate_data
    original_metadata = compat.aggregate.update_meta_data

    def fake_merger(*, datasets, output_repo_id, output_dir):
        assert compat.aggregate.aggregate_videos is compat._aggregate_videos
        assert compat.aggregate.aggregate_data is compat._aggregate_data
        assert compat.aggregate.update_meta_data is compat._update_meta_data
        return "merged"

    with patch.object(compat, "merge_datasets", side_effect=fake_merger):
        assert compat.merge_v042_datasets(
            [], output_repo_id="arbim/test", output_dir=Path("/tmp/not_written")
        ) == "merged"

    assert compat.aggregate.aggregate_videos is original_video
    assert compat.aggregate.aggregate_data is original_data
    assert compat.aggregate.update_meta_data is original_metadata
    print("PASS official merger context and restoration")


if __name__ == "__main__":
    test_shards_and_local_timestamps()
    test_scoped_compat_is_restored()
    print("SMOKE 19 fast LeRobot shard mapping PASSED")
