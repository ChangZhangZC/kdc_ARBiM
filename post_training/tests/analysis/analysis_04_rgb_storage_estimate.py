from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_RL_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
for path in (REPO_ROOT, POST_RL_SRC, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from post_rl.data import data_prepare as dp


def _select_indices(total: int, mode: str, count: int, start: int, seed: int) -> np.ndarray:
    count = min(int(count), total)
    if count <= 0:
        raise ValueError("num_steps must be positive")
    if mode == "continuous":
        if start < 0 or start >= total:
            raise ValueError(f"start_index must be in [0,{total - 1}], got {start}")
        stop = min(start + count, total)
        return np.arange(start, stop, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))


def _group_selected(episodes, selected: np.ndarray):
    selected_set = set(int(value) for value in selected)
    grouped = []
    for episode_id, frame_indices in episodes.items():
        chosen = [index for index in frame_indices if index in selected_set]
        if chosen:
            grouped.append((episode_id, chosen))
    return grouped


def estimate(dataset, episodes, selected, batch_size, jpeg_quality, label):
    grouped = _group_selected(episodes, selected)
    per_camera = {key: 0 for key in dp.RGB_FEATURE_TO_BUFFER}
    sampled = 0
    progress = tqdm(total=len(selected), desc=f"JPEG estimate ({label})")
    batch_decode = dp._can_batch_decode(dataset)

    for episode_id, indices in grouped:
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            if batch_decode:
                items = dp._load_lerobot_frame_batch(
                    dataset, episode_id, batch_indices, use_depth=False
                )
            else:
                items = [
                    dp._frame_arrays(dataset[index], index, use_depth=False)
                    for index in batch_indices
                ]
            for item in items:
                for key in dp.RGB_FEATURE_TO_BUFFER:
                    per_camera[key] += len(
                        dp.encode_rgb_jpeg(item["rgb"][key], jpeg_quality)
                    )
                sampled += 1
                progress.update(1)
    progress.close()

    total_dataset = len(dataset)
    sampled_bytes = sum(per_camera.values())
    bytes_per_step = sampled_bytes / sampled
    estimated_bytes = bytes_per_step * total_dataset
    print(f"\n[{label}] sampled timesteps: {sampled}")
    print(f"[{label}] JPEG quality: {jpeg_quality}")
    for key, value in per_camera.items():
        print(
            f"[{label}] {key}: {value / sampled / 1024:.1f} KiB/image average"
        )
    print(f"[{label}] RGB total: {bytes_per_step / 1024:.1f} KiB/timestep")
    print(
        f"[{label}] estimated full RGB payload: "
        f"{estimated_bytes / 1e9:.2f} GB ({estimated_bytes / (1024 ** 3):.2f} GiB)"
    )
    return estimated_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analysis 04: sample LeRobot timesteps, encode them with the production "
            "JPEG path, and estimate full processed-NPY/Zarr RGB storage."
        )
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "post_training/configs/data/data_prepare.yaml"),
    )
    parser.add_argument(
        "--sample-mode",
        choices=("continuous", "random", "both"),
        default="both",
    )
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument("--stream-batch-size", type=int, default=None)
    args = parser.parse_args()

    config = dp.load_config(args.config)
    lerobot_root = config.get("lerobot_root")
    if not lerobot_root:
        raise ValueError("config must define lerobot_root")
    dataset = dp.load_lerobot_dataset(lerobot_root)
    episodes = dp._group_episode_indices(dataset)
    batch_size = int(
        args.stream_batch_size
        if args.stream_batch_size is not None
        else config.get("stream_batch_size", dp.ZARR_CHUNK_LEAD)
    )
    jpeg_quality = int(
        args.jpeg_quality
        if args.jpeg_quality is not None
        else config.get("jpeg_quality", dp.DEFAULT_JPEG_QUALITY)
    )

    modes = ("continuous", "random") if args.sample_mode == "both" else (args.sample_mode,)
    estimates = []
    for mode in modes:
        selected = _select_indices(
            len(dataset), mode, args.num_steps, args.start_index, args.seed
        )
        estimates.append(
            estimate(
                dataset,
                episodes,
                selected,
                batch_size,
                jpeg_quality,
                mode,
            )
        )
    if len(estimates) == 2:
        low = min(estimates)
        high = max(estimates)
        print(
            "\nCombined estimate from continuous/random samples: "
            f"{low / (1024 ** 3):.2f}--{high / (1024 ** 3):.2f} GiB RGB payload"
        )
        print("NPY/Zarr metadata and numeric transition fields add only a small overhead.")


if __name__ == "__main__":
    main()
