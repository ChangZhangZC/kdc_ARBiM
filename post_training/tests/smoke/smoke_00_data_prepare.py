from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import zarr

from _common import make_work_dir, print_pass, print_section
from post_rl.data import data_prepare as dp
from post_rl.data.offline_buffer import OfflineBuffer


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


def _episode_ends_from_timeout(timeout) -> np.ndarray:
    timeout = np.asarray(timeout, dtype=bool)
    return np.flatnonzero(timeout) + 1


def _validate_processed_reward(data: dict, config: dict) -> None:
    actions = data["action"]
    rewards = np.asarray(data["reward"], dtype=np.float64)
    episode_ends = _episode_ends_from_timeout(data["timeout"])
    if len(episode_ends) == 0:
        raise AssertionError("Processed data contains no completed episode")

    start = 0
    for episode_idx, end in enumerate(episode_ends):
        episode_len = int(end - start)
        if episode_len <= 0:
            raise AssertionError(f"Episode {episode_idx} has invalid length {episode_len}")

        for local_t, index in enumerate(range(start, end)):
            expected = float(local_t == episode_len - 1)
            if expected == 1.0:
                expected -= (
                    float(config["lambda_penalty"])
                    * episode_len
                    / float(config["max_episode_len"])
                )
            if local_t > 0:
                current_action = np.asarray(actions[index])
                previous_action = np.asarray(actions[index - 1])
                expected -= float(config["smooth_penalty"]) * np.linalg.norm(
                    current_action - previous_action
                )

            if not np.isclose(rewards[index], expected, rtol=1e-6, atol=1e-6):
                raise AssertionError(
                    f"Reward mismatch at episode={episode_idx}, t={local_t}: "
                    f"actual={rewards[index]:.8f}, expected={expected:.8f}"
                )
        start = int(end)

    if start != len(rewards):
        raise AssertionError(
            f"Processed episodes cover {start} frames but dataset has {len(rewards)}"
        )


def _validate_processed_npy(data: dict, use_depth: bool, config: dict) -> np.ndarray:
    dp._validate_processed_data(data, use_depth, "smoke processed npy")
    required = ["agent_pos", "action", "rgb", "reward", "done", "timeout"]
    if use_depth:
        required.append("depth")

    lengths = {key: len(data[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"Processed field lengths differ: {lengths}")
    if next(iter(lengths.values())) == 0:
        raise AssertionError("Processed NPY contains zero frames")

    for index, value in enumerate(data["agent_pos"]):
        _as_numpy(value, f"processed.agent_pos[{index}]")
    for index, value in enumerate(data["action"]):
        _as_numpy(value, f"processed.action[{index}]")

    episode_ends = _episode_ends_from_timeout(data["timeout"])
    done_ends = np.flatnonzero(np.asarray(data["done"], dtype=bool)) + 1
    np.testing.assert_array_equal(episode_ends, done_ends)
    if episode_ends[-1] != len(data["timeout"]):
        raise AssertionError("Final processed episode is not closed at the last frame")

    _validate_processed_reward(data, config)
    return episode_ends.astype(np.int64)


def _transition_probe_indices(n: int, episode_ends: np.ndarray) -> list[int]:
    indices = {0, n // 2, n - 1}
    for end in episode_ends:
        terminal = int(end) - 1
        indices.add(terminal)
        if terminal > 0:
            indices.add(terminal - 1)
    return sorted(index for index in indices if 0 <= index < n)[:64]


def _validate_transition_buffers(
    data: dict,
    buffers: dict,
    episode_ends: np.ndarray,
    use_depth: bool,
) -> None:
    n = len(data["action"])
    if int(buffers["_total_count"]) != n:
        raise AssertionError(
            f"Transition count mismatch: {buffers['_total_count']} != {n}"
        )
    np.testing.assert_array_equal(
        np.asarray(buffers["episode_ends"], dtype=np.int64), episode_ends
    )

    timeout = np.asarray(data["timeout"], dtype=bool)
    next_indices = np.arange(n, dtype=np.int64) + 1
    next_indices[-1] = n - 1
    terminal_indices = np.flatnonzero(timeout)
    next_indices[terminal_indices] = terminal_indices

    state = np.stack(data["agent_pos"], axis=0)
    action = np.stack(data["action"], axis=0)
    _assert_array_equal(np.stack(buffers["state"], axis=0), state, "buffer.state")
    _assert_array_equal(
        np.stack(buffers["next_state"], axis=0),
        state[next_indices],
        "buffer.next_state",
    )
    _assert_array_equal(np.stack(buffers["action"], axis=0), action, "buffer.action")
    _assert_array_equal(
        np.stack(buffers["next_action"], axis=0),
        action[next_indices],
        "buffer.next_action",
    )
    _assert_array_equal(
        np.asarray(buffers["reward"], dtype=np.float32),
        np.asarray(data["reward"], dtype=np.float32),
        "buffer.reward",
    )
    np.testing.assert_array_equal(
        np.asarray(buffers["done"], dtype=bool), np.asarray(data["done"], dtype=bool)
    )
    np.testing.assert_array_equal(
        np.asarray(buffers["timeout"], dtype=bool), timeout
    )

    probe_indices = _transition_probe_indices(n, episode_ends)
    for index in probe_indices:
        next_index = int(next_indices[index])
        for feature_name, buffer_name in dp.RGB_FEATURE_TO_BUFFER.items():
            _assert_array_equal(
                buffers[buffer_name][index],
                data["rgb"][index][feature_name],
                f"{buffer_name}[{index}]",
            )
            _assert_array_equal(
                buffers[f"next_{buffer_name}"][index],
                data["rgb"][next_index][feature_name],
                f"next_{buffer_name}[{index}]",
            )
        if use_depth:
            for feature_name, buffer_name in dp.DEPTH_FEATURE_TO_BUFFER.items():
                _assert_array_equal(
                    buffers[buffer_name][index],
                    data["depth"][index][feature_name],
                    f"{buffer_name}[{index}]",
                )
                _assert_array_equal(
                    buffers[f"next_{buffer_name}"][index],
                    data["depth"][next_index][feature_name],
                    f"next_{buffer_name}[{index}]",
                )


def _validate_zarr(
    zarr_path: str,
    buffers: dict,
    episode_ends: np.ndarray,
    use_depth: bool,
) -> None:
    root = zarr.open_group(zarr_path, mode="r")
    if "data" not in root or "meta" not in root:
        raise AssertionError("Zarr must contain data/ and meta/ groups")
    data = root["data"]
    meta = root["meta"]

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

    missing = sorted(required.difference(data.keys()))
    if missing:
        raise AssertionError(f"Zarr data group is missing fields: {missing}")
    if "episode_ends" not in meta:
        raise AssertionError("Zarr meta group is missing episode_ends")

    n = len(buffers["state"])
    bad_lengths = {key: data[key].shape[0] for key in required if data[key].shape[0] != n}
    if bad_lengths:
        raise AssertionError(f"Zarr fields do not all have length {n}: {bad_lengths}")

    stored_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    np.testing.assert_array_equal(stored_ends, episode_ends)
    if len(stored_ends) == 0 or stored_ends[-1] != n:
        raise AssertionError("Zarr episode_ends does not close the final transition")
    if np.any(np.diff(stored_ends) <= 0):
        raise AssertionError("Zarr episode_ends must be strictly increasing")

    for key in ("state", "next_state", "action", "next_action", "reward", "return"):
        _as_numpy(data[key][:], f"zarr.data.{key}")

    probe_indices = _transition_probe_indices(n, episode_ends)
    image_keys = list(dp.RGB_FEATURE_TO_BUFFER.values())
    if use_depth:
        image_keys.extend(dp.DEPTH_FEATURE_TO_BUFFER.values())
    for key in image_keys:
        for index in probe_indices:
            _assert_array_equal(data[key][index], buffers[key][index], f"zarr.{key}[{index}]")
            _assert_array_equal(
                data[f"next_{key}"][index],
                buffers[f"next_{key}"][index],
                f"zarr.next_{key}[{index}]",
            )

    offline = OfflineBuffer(
        device=torch.device("cpu"),
        gamma=dp.RETURN_GAMMA,
        use_depth=use_depth,
    )
    offline.load_zarr(zarr_path)
    if len(offline) != n:
        raise AssertionError(f"OfflineBuffer length mismatch: {len(offline)} != {n}")
    np.testing.assert_array_equal(offline.episode_ends, episode_ends)
    offline.compute_return()
    np.testing.assert_allclose(
        offline["return"],
        np.asarray(data["return"][:], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke 00: LeRobot -> processed NPY -> Offline RL Zarr contract"
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
    work_dir = pathlib.Path(make_work_dir(args, "smoke_00_data_prepare"))
    processed_path = work_dir / "processed.npy"
    zarr_path = work_dir / "offline_rl.zarr"

    config = {
        "lerobot_root": lerobot_root,
        "processed_npy_output": str(processed_path),
        "use_depth": bool(args.use_depth),
        "max_episode_len": int(args.max_episode_len),
        "lambda_penalty": float(args.lambda_penalty),
        "smooth_penalty": float(args.smooth_penalty),
        "overwrite": True,
    }

    print_section("LeRobot input contract")
    frame_count, probed_episode_count = _probe_lerobot_input(
        lerobot_root, bool(args.use_depth)
    )
    print_pass(
        f"LeRobot input probes are valid: frames={frame_count}, "
        f"probe_episode_ids={probed_episode_count}"
    )

    print_section("LeRobot -> processed NPY")
    dp.process_raw_teleop_to_npy(config)
    processed = dp.load_processed_npy(str(processed_path))
    episode_ends = _validate_processed_npy(processed, bool(args.use_depth), config)
    print_pass(
        f"processed NPY is valid: frames={len(processed['action'])}, "
        f"episodes={len(episode_ends)}, reward formula matches"
    )

    print_section("processed NPY -> transition buffers")
    buffers = dp.make_buffers(use_depth=bool(args.use_depth))
    dp.append_processed_transitions(processed, buffers, "smoke_teleop", config)
    dp.record_source(
        buffers,
        "teleop_npy",
        {"name": "smoke_teleop", "path": str(processed_path)},
    )
    _validate_transition_buffers(
        processed,
        buffers,
        episode_ends,
        bool(args.use_depth),
    )
    print_pass("single-step and terminal self-loop transition alignment is correct")

    print_section("transition buffers -> Zarr -> OfflineBuffer")
    dp.write_zarr(buffers, str(zarr_path), overwrite=True)
    _validate_zarr(
        str(zarr_path),
        buffers,
        episode_ends,
        bool(args.use_depth),
    )
    print_pass("Zarr contract and OfflineBuffer reload are valid")

    print(f"\nProcessed NPY: {processed_path}")
    print(f"Generated Zarr: {zarr_path}")
    print("SMOKE 00 PASSED")


if __name__ == "__main__":
    main()
