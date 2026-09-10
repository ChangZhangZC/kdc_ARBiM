from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

import numpy as np
import zarr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_RL_SRC = REPO_ROOT / "post_training" / "src"
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"
for path in (REPO_ROOT, POST_RL_SRC, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from post_rl.data import data_prepare as dp
from post_rl.data.streaming_data_prepare import build_zarr_from_lerobot


def _make_work_dir(path: str | None) -> pathlib.Path:
    if path:
        work_dir = pathlib.Path(path).expanduser().resolve() / "smoke_00_data_prepare"
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    return pathlib.Path(tempfile.mkdtemp(prefix="arbim_smoke_00_data_prepare_"))


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_pass(message: str) -> None:
    print(f"[PASS] {message}")


def _as_numpy(value, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.size == 0:
        raise AssertionError(f"{name} is empty")
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        raise AssertionError(f"{name} is not numeric: dtype={array.dtype}")
    if not np.isfinite(array.astype(np.float64, copy=False)).all():
        raise AssertionError(f"{name} contains NaN or Inf")
    return array


def _assert_array_equal(actual, expected, name: str) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name} shape mismatch: {actual.shape} != {expected.shape}"
        )
    if np.issubdtype(actual.dtype, np.floating) or np.issubdtype(
        expected.dtype, np.floating
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, err_msg=name)
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=name)


def _probe_lerobot_input(root: str, use_depth: bool) -> tuple[int, int]:
    dataset = dp.load_lerobot_dataset(root)
    if len(dataset) == 0:
        raise AssertionError("LeRobot dataset contains no frames")

    probe_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    episode_ids = set()
    required = [
        "episode_index",
        "observation.state",
        "action",
        *dp.RGB_FEATURE_TO_BUFFER.keys(),
    ]
    if use_depth:
        required.extend(dp.DEPTH_FEATURE_TO_BUFFER.keys())

    for index in probe_indices:
        frame = dataset[index]
        missing = [key for key in required if key not in frame]
        if missing:
            raise AssertionError(
                f"LeRobot probe frame {index} is missing required keys: {missing}"
            )
        episode_id = frame["episode_index"]
        if hasattr(episode_id, "item"):
            episode_id = episode_id.item()
        episode_ids.add(episode_id)
        _as_numpy(frame["observation.state"], f"frame[{index}].observation.state")
        _as_numpy(frame["action"], f"frame[{index}].action")
        for key in dp.RGB_FEATURE_TO_BUFFER:
            _as_numpy(frame[key], f"frame[{index}].{key}")
        if use_depth:
            for key in dp.DEPTH_FEATURE_TO_BUFFER:
                _as_numpy(frame[key], f"frame[{index}].{key}")

    return len(dataset), len(episode_ids)


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
        *dp.RGB_FEATURE_TO_BUFFER.values(),
        *(f"next_{name}" for name in dp.RGB_FEATURE_TO_BUFFER.values()),
    }
    if use_depth:
        required.update(dp.DEPTH_FEATURE_TO_BUFFER.values())
        required.update(f"next_{name}" for name in dp.DEPTH_FEATURE_TO_BUFFER.values())
    return required


def _transition_probe_indices(n: int, episode_ends: np.ndarray) -> list[int]:
    indices = {0, n // 2, n - 1}
    for end in episode_ends:
        terminal = int(end) - 1
        indices.add(terminal)
        if terminal > 0:
            indices.add(terminal - 1)
    return sorted(index for index in indices if 0 <= index < n)[:64]


def _validate_reward_and_return(data, episode_ends: np.ndarray, config: dict) -> None:
    actions = np.asarray(data["action"][:], dtype=np.float32)
    rewards = np.asarray(data["reward"][:], dtype=np.float32)
    done = np.asarray(data["done"][:], dtype=bool).reshape(-1)
    timeout = np.asarray(data["timeout"][:], dtype=bool).reshape(-1)
    stored_return = np.asarray(data["return"][:], dtype=np.float32)

    n = len(actions)
    expected_done = np.zeros(n, dtype=bool)
    expected_done[episode_ends - 1] = True
    np.testing.assert_array_equal(done, expected_done)
    np.testing.assert_array_equal(timeout, expected_done)

    start = 0
    for episode_idx, end in enumerate(episode_ends):
        end = int(end)
        episode_len = end - start
        if episode_len <= 0:
            raise AssertionError(f"episode {episode_idx} has invalid length {episode_len}")
        for local_t, index in enumerate(range(start, end)):
            expected_reward = float(local_t == episode_len - 1)
            if local_t == episode_len - 1:
                expected_reward -= (
                    float(config["lambda_penalty"])
                    * episode_len
                    / float(config["max_episode_len"])
                )
            if local_t > 0:
                expected_reward -= float(config["smooth_penalty"]) * np.linalg.norm(
                    actions[index] - actions[index - 1]
                )
            if not np.isclose(
                rewards[index, 0], expected_reward, rtol=1e-6, atol=1e-6
            ):
                raise AssertionError(
                    f"reward mismatch at episode={episode_idx}, t={local_t}: "
                    f"actual={rewards[index, 0]:.8f}, expected={expected_reward:.8f}"
                )
        start = end

    not_done = (~expected_done).astype(np.float32).reshape(-1, 1)
    expected_return = dp.compute_return(rewards, not_done, gamma=dp.RETURN_GAMMA)
    np.testing.assert_allclose(stored_return, expected_return, rtol=1e-6, atol=1e-6)


def _validate_source_alignment(
    lerobot_root: str,
    data,
    source_indices: np.ndarray,
    episode_ends: np.ndarray,
    use_depth: bool,
) -> None:
    dataset = dp.load_lerobot_dataset(lerobot_root)
    n = len(source_indices)
    terminal_indices = set((episode_ends - 1).tolist())

    for output_index in _transition_probe_indices(n, episode_ends):
        source_index = int(source_indices[output_index])
        next_output_index = output_index if output_index in terminal_indices else output_index + 1
        next_source_index = int(source_indices[next_output_index])
        frame = dataset[source_index]
        next_frame = frame if next_source_index == source_index else dataset[next_source_index]

        _assert_array_equal(
            data["state"][output_index],
            _as_numpy(frame["observation.state"], f"source.state[{source_index}]").astype(np.float32),
            f"state[{output_index}]",
        )
        _assert_array_equal(
            data["next_state"][output_index],
            _as_numpy(next_frame["observation.state"], f"source.next_state[{next_source_index}]").astype(np.float32),
            f"next_state[{output_index}]",
        )
        _assert_array_equal(
            data["action"][output_index],
            _as_numpy(frame["action"], f"source.action[{source_index}]").astype(np.float32),
            f"action[{output_index}]",
        )
        _assert_array_equal(
            data["next_action"][output_index],
            _as_numpy(next_frame["action"], f"source.next_action[{next_source_index}]").astype(np.float32),
            f"next_action[{output_index}]",
        )

        for feature_name, buffer_name in dp.RGB_FEATURE_TO_BUFFER.items():
            _assert_array_equal(
                data[buffer_name][output_index],
                _as_numpy(frame[feature_name], f"source.{feature_name}[{source_index}]"),
                f"{buffer_name}[{output_index}]",
            )
            _assert_array_equal(
                data[f"next_{buffer_name}"][output_index],
                _as_numpy(next_frame[feature_name], f"source.next_{feature_name}[{next_source_index}]"),
                f"next_{buffer_name}[{output_index}]",
            )

        if use_depth:
            for feature_name, buffer_name in dp.DEPTH_FEATURE_TO_BUFFER.items():
                _assert_array_equal(
                    data[buffer_name][output_index],
                    _as_numpy(frame[feature_name], f"source.{feature_name}[{source_index}]"),
                    f"{buffer_name}[{output_index}]",
                )
                _assert_array_equal(
                    data[f"next_{buffer_name}"][output_index],
                    _as_numpy(next_frame[feature_name], f"source.next_{feature_name}[{next_source_index}]"),
                    f"next_{buffer_name}[{output_index}]",
                )


def _validate_zarr(
    zarr_path: str,
    result: dict,
    lerobot_root: str,
    config: dict,
    use_depth: bool,
) -> None:
    root = zarr.open_group(zarr_path, mode="r")
    if "data" not in root or "meta" not in root:
        raise AssertionError("Zarr must contain data/ and meta/ groups")
    data = root["data"]
    meta = root["meta"]
    required = _required_zarr_keys(use_depth)
    missing = sorted(required.difference(data.keys()))
    if missing:
        raise AssertionError(f"Zarr data group is missing fields: {missing}")
    if "episode_ends" not in meta:
        raise AssertionError("Zarr meta group is missing episode_ends")

    n = int(result["num_frames"])
    bad_lengths = {
        key: data[key].shape[0]
        for key in required
        if data[key].shape[0] != n
    }
    if bad_lengths:
        raise AssertionError(f"Zarr fields do not all have length {n}: {bad_lengths}")

    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    np.testing.assert_array_equal(episode_ends, result["episode_ends"])
    if len(episode_ends) == 0 or episode_ends[-1] != n:
        raise AssertionError("Zarr episode_ends does not close the final transition")
    if np.any(np.diff(episode_ends) <= 0):
        raise AssertionError("Zarr episode_ends must be strictly increasing")

    for key in ("state", "next_state", "action", "next_action", "reward", "return"):
        _as_numpy(data[key][:], f"zarr.data.{key}")

    _validate_reward_and_return(data, episode_ends, config)
    _validate_source_alignment(
        lerobot_root,
        data,
        np.asarray(result["source_indices"], dtype=np.int64),
        episode_ends,
        use_depth,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke 00: memory-safe LeRobot -> Offline RL Zarr contract"
    )
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--use-depth", action="store_true")
    parser.add_argument("--max-episode-len", type=int, default=2000)
    parser.add_argument("--lambda-penalty", type=float, default=0.05)
    parser.add_argument("--smooth-penalty", type=float, default=0.01)
    args = parser.parse_args()

    if args.max_episode_len <= 0:
        raise ValueError("--max-episode-len must be positive")

    lerobot_root = str(pathlib.Path(args.lerobot_root).expanduser().resolve())
    work_dir = _make_work_dir(args.work_dir)
    zarr_path = work_dir / "offline_rl.zarr"
    config = {
        "lerobot_root": lerobot_root,
        "zarr_output_path": str(zarr_path),
        "use_depth": bool(args.use_depth),
        "max_episode_len": int(args.max_episode_len),
        "lambda_penalty": float(args.lambda_penalty),
        "smooth_penalty": float(args.smooth_penalty),
        "overwrite": True,
    }

    _print_section("LeRobot input contract")
    frame_count, probed_episode_count = _probe_lerobot_input(
        lerobot_root, bool(args.use_depth)
    )
    _print_pass(
        f"LeRobot input probes are valid: frames={frame_count}, "
        f"probe_episode_ids={probed_episode_count}"
    )

    _print_section("LeRobot -> streaming Offline RL Zarr")
    result = build_zarr_from_lerobot(config)
    _print_pass(
        f"streaming conversion completed: frames={result['num_frames']}, "
        f"episodes={result['num_episodes']}"
    )

    _print_section("Zarr contract and source alignment")
    _validate_zarr(
        str(zarr_path),
        result,
        lerobot_root,
        config,
        bool(args.use_depth),
    )
    _print_pass("reward, return, terminal self-loop, RGB and transition alignment are valid")

    print(f"\nGenerated Zarr: {zarr_path}")
    print("SMOKE 00 PASSED")


if __name__ == "__main__":
    main()
