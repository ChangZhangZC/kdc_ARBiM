from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

import numpy as np
import torch
import zarr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_RL_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
for path in (REPO_ROOT, POST_RL_SRC, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from post_rl.data import data_prepare as dp
from post_rl.data.offline_buffer import OfflineBuffer


def _make_work_dir(path: str | None) -> pathlib.Path:
    if path:
        work_dir = pathlib.Path(path).expanduser().resolve() / "smoke_00_data_prepare"
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    return pathlib.Path(tempfile.mkdtemp(prefix="arbim_smoke_00_data_prepare_"))


def _required_zarr_keys(use_depth: bool) -> set[str]:
    required = {
        "state",
        "next_state",
        "action",
        "next_action",
        "reward",
        "return",
        "done",
        "timeout",
        "next_index",
    }
    for name in dp.RGB_FEATURE_TO_BUFFER.values():
        required.add(f"{name}_jpeg_data")
        required.add(f"{name}_jpeg_offsets")
    if use_depth:
        required.update(dp.DEPTH_FEATURE_TO_BUFFER.values())
        required.update(f"next_{name}" for name in dp.DEPTH_FEATURE_TO_BUFFER.values())
    return required


def _validate_npy(
    path: pathlib.Path,
    expected_frames: int,
    expected_episodes: int,
    use_depth: bool,
) -> None:
    info = dp._read_source_header(path, use_depth)
    assert info["streamed"], "raw_to_npy must produce the streamed NPY format"
    assert info["stream_format"] == dp.STREAM_NPY_FORMAT
    assert info["rgb_storage"] == "jpeg"
    assert info["num_frames"] == expected_frames
    assert info["num_episodes"] == expected_episodes

    counted_frames = 0
    counted_episodes = 0
    last_timeout = False
    for frame in dp._iter_processed_frames(path, use_depth):
        counted_frames += 1
        counted_episodes += int(frame["timeout"])
        last_timeout = bool(frame["timeout"])
        for key in dp.RGB_FEATURE_TO_BUFFER:
            assert isinstance(frame["rgb"][key], bytes)
            decoded = dp.decode_rgb_jpeg(frame["rgb"][key])
            assert decoded.dtype == np.uint8
            assert decoded.shape == info["rgb_shapes"][key]
        if use_depth:
            assert set(dp.DEPTH_FEATURE_TO_BUFFER).issubset(frame["depth"])

    assert counted_frames == expected_frames
    assert counted_episodes == expected_episodes
    assert last_timeout, "processed stream must terminate at an episode boundary"


def _validate_zarr(
    path: pathlib.Path,
    expected_frames: int,
    expected_episodes: int,
    use_depth: bool,
) -> None:
    root = zarr.open_group(str(path), mode="r")
    assert "data" in root and "meta" in root
    assert root.attrs.get("rgb_storage") == dp.RGB_ZARR_STORAGE
    data = root["data"]
    meta = root["meta"]

    required = _required_zarr_keys(use_depth)
    missing = sorted(required.difference(data.keys()))
    assert not missing, f"missing zarr fields: {missing}"

    for key in (
        "state",
        "next_state",
        "action",
        "next_action",
        "reward",
        "return",
        "done",
        "timeout",
        "next_index",
    ):
        assert data[key].shape[0] == expected_frames, (key, data[key].shape)
    for name in dp.RGB_FEATURE_TO_BUFFER.values():
        assert data[f"{name}_jpeg_offsets"].shape[0] == expected_frames + 1
        offsets = np.asarray(data[f"{name}_jpeg_offsets"][:], dtype=np.int64)
        assert offsets[0] == 0
        assert np.all(np.diff(offsets) > 0)
        assert offsets[-1] == data[f"{name}_jpeg_data"].shape[0]

    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    assert len(episode_ends) == expected_episodes
    assert episode_ends[-1] == expected_frames
    assert np.all(np.diff(episode_ends) > 0)

    done = np.asarray(data["done"][:], dtype=bool).reshape(-1)
    timeout = np.asarray(data["timeout"][:], dtype=bool).reshape(-1)
    np.testing.assert_array_equal(done, timeout)
    expected_done = np.zeros(expected_frames, dtype=bool)
    expected_done[episode_ends - 1] = True
    np.testing.assert_array_equal(done, expected_done)

    next_index = np.asarray(data["next_index"][:], dtype=np.int64)
    expected_next = np.arange(expected_frames, dtype=np.int64) + 1
    expected_next[expected_done] = np.arange(expected_frames, dtype=np.int64)[expected_done]
    np.testing.assert_array_equal(next_index, expected_next)

    reward = np.asarray(data["reward"][:], dtype=np.float32)
    stored_return = np.asarray(data["return"][:], dtype=np.float32)
    not_done = (~done).astype(np.float32).reshape(-1, 1)
    expected_return = dp.compute_return(reward, not_done, gamma=dp.RETURN_GAMMA)
    np.testing.assert_allclose(stored_return, expected_return, rtol=1e-6, atol=1e-6)

    buffer = OfflineBuffer(device=torch.device("cpu"), gamma=dp.RETURN_GAMMA, use_depth=use_depth)
    buffer.load_zarr(str(path))
    for name in dp.RGB_FEATURE_TO_BUFFER.values():
        current = buffer[name][0]
        nxt = buffer[f"next_{name}"][0]
        assert current.dtype == np.uint8
        assert nxt.dtype == np.uint8
        assert current.shape == tuple(root.attrs["rgb_shapes"][name])
        assert nxt.shape == current.shape


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke 00: raw_to_npy -> build_db compressed RGB contract"
    )
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--use-depth", action="store_true")
    parser.add_argument("--max-episode-len", type=int, default=2000)
    parser.add_argument("--lambda-penalty", type=float, default=0.05)
    parser.add_argument("--smooth-penalty", type=float, default=0.01)
    parser.add_argument("--stream-batch-size", type=int, default=50)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    lerobot_root = str(pathlib.Path(args.lerobot_root).expanduser().resolve())
    dataset = dp.load_lerobot_dataset(lerobot_root)
    expected_frames = len(dataset)
    expected_episodes = int(dataset.meta.total_episodes)
    if expected_frames <= 0 or expected_episodes <= 0:
        raise AssertionError("LeRobot dataset must contain frames and episodes")

    work_dir = _make_work_dir(args.work_dir)
    npy_path = work_dir / "processed.npy"
    zarr_path = work_dir / "offline_rl.zarr"
    common = {
        "use_depth": bool(args.use_depth),
        "max_episode_len": int(args.max_episode_len),
        "lambda_penalty": float(args.lambda_penalty),
        "smooth_penalty": float(args.smooth_penalty),
        "stream_batch_size": int(args.stream_batch_size),
        "jpeg_quality": int(args.jpeg_quality),
        "overwrite": True,
    }

    print("\n=== raw_to_npy ===")
    result_npy = dp.process_raw_teleop_to_npy(
        {
            **common,
            "lerobot_root": lerobot_root,
            "processed_npy_output": str(npy_path),
        }
    )
    assert result_npy["num_frames"] == expected_frames
    assert result_npy["num_episodes"] == expected_episodes
    _validate_npy(npy_path, expected_frames, expected_episodes, bool(args.use_depth))
    print("[PASS] raw_to_npy JPEG-stream contract")

    print("\n=== build_db ===")
    result_zarr = dp.run_build_db(
        {
            **common,
            "zarr_output_path": str(zarr_path),
            "teleop_sources": [{"name": "smoke", "path": str(npy_path)}],
        }
    )
    assert result_zarr["num_frames"] == expected_frames
    assert result_zarr["num_episodes"] == expected_episodes
    _validate_zarr(zarr_path, expected_frames, expected_episodes, bool(args.use_depth))
    print("[PASS] build_db compressed Offline RL Zarr contract")

    print(f"\nGenerated NPY: {npy_path}")
    print(f"Generated Zarr: {zarr_path}")
    print("SMOKE 00 PASSED")


if __name__ == "__main__":
    main()
