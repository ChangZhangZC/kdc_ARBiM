"""Compatibility fix for the LeRobot v0.4.2 file-level dataset merger.

LeRobot v0.4.2 aggregate.py incorrectly assigns the *last* destination
video shard to every episode from one source dataset. Its duration offsets
also use a cross-shard cumulative duration. These bugs corrupt videos even
when the total frame/episode counts and feature schemas are correct.

Keep the official merge_datasets() orchestration, Parquet writing and task
mapping, but provide source-to-destination shard maps for videos and Parquet.
One source video file becomes one destination video file: direct copy2, no
decode, encode, or timestamp remux. This intentionally favors correctness
over packing many videos into larger MP4 files.

Remove these scoped patches once the pinned LeRobot version contains upstream's
src_to_dst / per-destination-file offset fixes.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import shutil
from unittest.mock import patch

import pandas as pd

import lerobot.datasets.aggregate as aggregate
from lerobot.datasets.dataset_tools import merge_datasets


def _aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size):
    """Copy each complete source video shard into a dedicated destination shard."""
    from lerobot.datasets.utils import DEFAULT_VIDEO_PATH, update_chunk_file_indices

    for video_key, state in videos_idx.items():
        state["episode_duration"] = 0.0
        state["src_to_dst"] = {}
        state["src_to_offset"] = {}
        chunk_idx = int(state["chunk"])
        file_idx = int(state["file"])
        pairs = sorted(set(zip(
            src_meta.episodes[f"videos/{video_key}/chunk_index"],
            src_meta.episodes[f"videos/{video_key}/file_index"],
            strict=False,
        )))
        for src_chunk, src_file in pairs:
            src_key = (int(src_chunk), int(src_file))
            src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key, chunk_index=src_key[0], file_index=src_key[1],
            )
            dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx,
            )
            if dst_path.exists():
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx,
                )
            if not src_path.is_file() or dst_path.exists():
                raise FileNotFoundError(f"Cannot safely copy unique video shard {src_path} to {dst_path}")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            state["src_to_dst"][src_key] = (int(chunk_idx), int(file_idx))
            state["src_to_offset"][src_key] = 0.0
            state["episode_duration"] += aggregate.get_video_duration_in_s(src_path)
        state["chunk"] = int(chunk_idx)
        state["file"] = int(file_idx)
    return videos_idx


def _aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size):
    """Preserve the official Parquet writer, recording each resulting shard."""
    from lerobot.datasets.utils import DEFAULT_DATA_PATH
    pairs = sorted(set(zip(
        src_meta.episodes["data/chunk_index"],
        src_meta.episodes["data/file_index"],
        strict=False,
    )))
    mapping = {}
    for src_chunk, src_file in pairs:
        src_key = (int(src_chunk), int(src_file))
        src_path = src_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=src_key[0], file_index=src_key[1],
        )
        if not src_path.is_file():
            raise FileNotFoundError(f"Source Parquet shard missing: {src_path}")
        df = pd.read_parquet(src_path)
        df = aggregate.update_data_df(df, src_meta, dst_meta)
        data_idx = aggregate.append_or_create_parquet_file(
            df, src_path, data_idx, data_files_size_in_mb, chunk_size,
            DEFAULT_DATA_PATH, contains_images=len(dst_meta.image_keys) > 0,
            aggr_root=dst_meta.root,
        )
        mapping[src_key] = (int(data_idx["chunk"]), int(data_idx["file"]))
    data_idx["src_to_dst"] = mapping
    return data_idx


def _update_meta_data(df, dst_meta, meta_idx, data_idx, videos_idx):
    """Correctly map each Episode to its copied video and Parquet shard."""
    df = df.copy()
    # Leave the pinned v0.4.2 meta/episodes shard logic unchanged.
    df["meta/episodes/chunk_index"] += meta_idx["chunk"]
    df["meta/episodes/file_index"] += meta_idx["file"]

    mapping = data_idx.get("src_to_dst", {})
    for index in df.index:
        data_key = (int(df.at[index, "data/chunk_index"]), int(df.at[index, "data/file_index"]))
        if data_key not in mapping:
            raise ValueError(f"Missing destination Parquet mapping for {data_key}")
        chunk, file = mapping[data_key]
        df.at[index, "data/chunk_index"] = chunk
        df.at[index, "data/file_index"] = file

    for video_key, state in videos_idx.items():
        mapping = state.get("src_to_dst", {})
        for index in df.index:
            chunk_key = f"videos/{video_key}/chunk_index"
            file_key = f"videos/{video_key}/file_index"
            src_key = (int(df.at[index, chunk_key]), int(df.at[index, file_key]))
            if src_key not in mapping:
                raise ValueError(f"Missing destination video mapping: {video_key} {src_key}")
            chunk, file = mapping[src_key]
            df.at[index, chunk_key] = chunk
            df.at[index, file_key] = file
            # Each source file was copied unmodified and starts at timestamp zero.
            # Preserve the source per-episode from/to_timestamp exactly.

    df["dataset_from_index"] += dst_meta.info["total_frames"]
    df["dataset_to_index"] += dst_meta.info["total_frames"]
    df["episode_index"] += dst_meta.info["total_episodes"]
    return df


def _aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx):
    """Keep meta/episodes location fields aligned with the actual Parquet writer.

    When a source metadata file is appended to an existing destination shard,
    simply adding its original file index gives the wrong destination path.
    Predict the exact file rotation using the pinned LeRobot writer's rule.
    """
    from lerobot.datasets.utils import (
        DEFAULT_EPISODES_PATH, DEFAULT_DATA_FILE_SIZE_IN_MB,
        DEFAULT_CHUNK_SIZE, get_parquet_file_size_in_mb, update_chunk_file_indices,
    )

    pairs = sorted(set(zip(
        src_meta.episodes["meta/episodes/chunk_index"],
        src_meta.episodes["meta/episodes/file_index"],
        strict=False,
    )))
    for src_chunk, src_file in pairs:
        src_path = src_meta.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=int(src_chunk), file_index=int(src_file),
        )
        if not src_path.is_file():
            raise FileNotFoundError(f"Source Episode Parquet shard missing: {src_path}")
        df = pd.read_parquet(src_path)
        df = _update_meta_data(df, dst_meta, meta_idx, data_idx, videos_idx)
        chunk_idx, file_idx = int(meta_idx["chunk"]), int(meta_idx["file"])
        dst_path = dst_meta.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=chunk_idx, file_index=file_idx,
        )
        if dst_path.exists() and (
            get_parquet_file_size_in_mb(src_path) +
            get_parquet_file_size_in_mb(dst_path) >= DEFAULT_DATA_FILE_SIZE_IN_MB
        ):
            chunk_idx, file_idx = update_chunk_file_indices(
                chunk_idx, file_idx, DEFAULT_CHUNK_SIZE,
            )
        df["meta/episodes/chunk_index"] = int(chunk_idx)
        df["meta/episodes/file_index"] = int(file_idx)
        meta_idx = aggregate.append_or_create_parquet_file(
            df, src_path, meta_idx, DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_CHUNK_SIZE, DEFAULT_EPISODES_PATH, contains_images=False,
            aggr_root=dst_meta.root,
        )
        if (int(meta_idx["chunk"]), int(meta_idx["file"])) != (int(chunk_idx), int(file_idx)):
            raise RuntimeError("LeRobot Episode Parquet shard rotation unexpectedly changed")
    for state in videos_idx.values():
        state["latest_duration"] += state["episode_duration"]
    return meta_idx


def merge_v042_datasets(datasets, *, output_repo_id, output_dir):
    """Use LeRobot official merge with only the v0.4.2 shard-index fixes scoped."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(aggregate, "aggregate_videos", _aggregate_videos))
        stack.enter_context(patch.object(aggregate, "aggregate_data", _aggregate_data))
        stack.enter_context(patch.object(aggregate, "update_meta_data", _update_meta_data))
        stack.enter_context(patch.object(aggregate, "aggregate_metadata", _aggregate_metadata))
        return merge_datasets(
            datasets=datasets, output_repo_id=output_repo_id, output_dir=Path(output_dir),
        )
