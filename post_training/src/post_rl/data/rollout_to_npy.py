"""Convert successful Data Wheel staging rollouts to ARBiM Processed NPY V2.

Standalone converter: no ROS, LeRobot, or Zarr imports. Inputs are the
arbim_rollout_staging_v2 files written by rollout_recorder.py. The output
matches data_prepare.py's streamed JPEG NPY reader and run_build_db contract.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


STAGING_FORMAT = "arbim_rollout_staging_v2"
PROCESSED_FORMAT = "arbim_processed_npy_stream_v2"
REWARD_MODE = "last_gripper_release_sparse"
CAMERA_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
)


def _discover_sources(sources):
    paths = []
    for source in sources:
        source = Path(source).expanduser().resolve()
        if source.is_dir():
            paths.extend(sorted(source.glob("*_staging.npy")))
        elif source.is_file():
            paths.append(source)
        else:
            raise FileNotFoundError(f"Staging source not found: {source}")
    unique = sorted(set(paths))
    if not unique:
        raise ValueError("No staging NPY files found")
    return unique


def _load_staging(path):
    with path.open("rb") as handle:
        payload = np.load(handle, allow_pickle=True).item()
        if handle.read(1):
            raise ValueError(f"{path}: unexpected trailing data")
    if not isinstance(payload, dict) or payload.get("__format__") != STAGING_FORMAT:
        raise ValueError(f"{path}: expected {STAGING_FORMAT}")
    return payload


def _validate_staging(path, data):
    """Return input dimensions if success and Reward V2 are valid."""
    if not data.get("success") or data.get("reward_status") != "valid":
        return None
    if data.get("reward_mode") != REWARD_MODE or data.get("rgb_storage") != "jpeg":
        raise ValueError(f"{path}: unexpected reward_mode or rgb_storage")

    actions = np.asarray(data["actions"], dtype=np.float32)
    reward = np.asarray(data["reward"], dtype=np.float32)
    observations = data["observations"]
    steps = int(data["num_steps"])
    if actions.shape != (steps, 16) or steps < 2:
        raise ValueError(f"{path}: actions must be [T,16] with T>=2")
    if reward.shape != (steps,) or len(observations) != steps + 1:
        raise ValueError(f"{path}: reward/observation lengths do not match actions")
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(reward)):
        raise ValueError(f"{path}: nonfinite actions or rewards")
    ones = np.flatnonzero(reward == 1.0)
    if len(ones) != 1 or np.count_nonzero(reward) != 1 or ones[0] >= steps - 1:
        raise ValueError(f"{path}: expected a single nonterminal +1 reward")
    event = data.get("reward_event")
    if not isinstance(event, dict) or event.get("frame_offset") != int(ones[0]):
        raise ValueError(f"{path}: reward_event and reward index mismatch")

    rgb_shapes = {}
    for index, obs in enumerate(observations):
        if not isinstance(obs, dict):
            raise ValueError(f"{path}: observation {index} is not a dict")
        state = np.asarray(obs["agent_pos"], dtype=np.float32)
        if state.shape != (16,) or not np.all(np.isfinite(state)):
            raise ValueError(f"{path}: invalid state at observation {index}")
        if set(obs["rgb"]) != set(CAMERA_KEYS):
            raise ValueError(f"{path}: camera keys mismatch at observation {index}")
        for key in CAMERA_KEYS:
            jpeg = obs["rgb"][key]
            if not isinstance(jpeg, bytes) or len(jpeg) < 4 or not jpeg.startswith(b"\xff\xd8"):
                raise ValueError(f"{path}: invalid JPEG bytes for {key} at {index}")
            if index == 0:
                bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
                    raise ValueError(f"{path}: cannot decode {key}")
                rgb_shapes[key] = (3, int(bgr.shape[0]), int(bgr.shape[1]))
    return {"steps": steps, "state_shape": (16,), "action_shape": (16,),
            "rgb_shapes": rgb_shapes, "reward_event": event}


def _new_chunk():
    return {
        "agent_pos": [], "action": [], "rgb": [],
        "reward": [], "done": [], "timeout": [],
    }


def _write_chunk(handle, chunk):
    if chunk["action"]:
        np.save(handle, chunk, allow_pickle=True)


def convert_staging_to_npy(sources, output, *, batch_size=50, overwrite=False):
    """Combine valid successful staging episodes into one streamed Processed NPY.

    Invalid-reward episodes are skipped and counted. Corrupted valid records,
    incompatible camera dimensions or action shapes abort conversion.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    paths = _discover_sources(sources)
    output = Path(output).expanduser().resolve()
    if output in paths:
        raise ValueError("Output path must differ from source paths")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    accepted = []
    skipped = []
    expected_shapes = None
    for path in paths:
        data = _load_staging(path)
        info = _validate_staging(path, data)
        if info is None:
            skipped.append(str(path))
            continue
        shape_spec = (
            info["state_shape"], info["action_shape"], info["rgb_shapes"]
        )
        if expected_shapes is None:
            expected_shapes = shape_spec
        elif shape_spec != expected_shapes:
            raise ValueError(f"{path}: mismatched camera/state/action dimensions")
        accepted.append((path, info))

    if not accepted:
        raise ValueError("No successful staging episodes with valid Reward V2")
    frame_count = sum(info["steps"] for _, info in accepted)
    first_info = accepted[0][1]
    manifest = {
        "__format__": PROCESSED_FORMAT,
        "num_frames": frame_count,
        "num_episodes": len(accepted),
        "use_depth": False,
        "state_shape": first_info["state_shape"],
        "action_shape": first_info["action_shape"],
        "rgb_storage": "jpeg",
        "jpeg_quality": 95,
        "rgb_shapes": first_info["rgb_shapes"],
        "rgb_dtypes": {key: "uint8" for key in CAMERA_KEYS},
        "source_root": str(paths[0].parent),
        "source_paths": [str(path) for path, _ in accepted],
        "reward_mode": REWARD_MODE,
        "reward_event": {
            "episodes": [
                {"source": str(path), "event": info["reward_event"]}
                for path, info in accepted
            ]
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    if temp.exists():
        raise FileExistsError(f"Temporary output already exists: {temp}")
    chunks = 0
    try:
        with temp.open("wb") as handle:
            np.save(handle, manifest, allow_pickle=True)
            chunk = _new_chunk()
            for path, info in accepted:
                data = _load_staging(path)
                # T transitions use obs[0:T], actions[0:T], reward[0:T].
                # The last obs[T] is preserved in staging, but run_build_db
                # currently repeats terminal state as next_state by design.
                actions, rewards, observations = (
                    data["actions"], data["reward"], data["observations"]
                )
                for step in range(info["steps"]):
                    obs = observations[step]
                    terminal = step == info["steps"] - 1
                    chunk["agent_pos"].append(obs["agent_pos"])
                    chunk["action"].append(actions[step])
                    chunk["rgb"].append(obs["rgb"])
                    chunk["reward"].append(float(rewards[step]))
                    chunk["done"].append(terminal)
                    chunk["timeout"].append(terminal)
                    if len(chunk["action"]) >= batch_size:
                        _write_chunk(handle, chunk)
                        chunks += 1
                        chunk = _new_chunk()
            if chunk["action"]:
                _write_chunk(handle, chunk)
                chunks += 1
        os.replace(temp, output)
    finally:
        if temp.exists():
            temp.unlink()
    return {
        "npy_path": str(output),
        "num_frames": frame_count,
        "num_episodes": len(accepted),
        "num_chunks": chunks,
        "skipped_invalid": len(skipped),
        "skipped_paths": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="Staging files or directories")
    parser.add_argument("--output", required=True, help="Processed NPY V2 output")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = convert_staging_to_npy(
        args.sources, args.output, batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(result)


if __name__ == "__main__":
    main()
