"""Convert successful ARBiM Data Wheel staging episodes into a LeRobot dataset.

Uses the actual Teleop LeRobot dataset as a schema/FPS template.
Keeps Reward V2 annotations in meta/arbim_staging_provenance.json (not as
training feature columns); invalid-reward successes are eligible for IL.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
for folder in (REPO_ROOT, REPO_ROOT / "third_party" / "lerobot" / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import lerobot_patches.custom_patches  # noqa: F401  (same patch as Teleop converter)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


RGB_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
)
DEPTH_KEYS = ("observation.depth_h", "observation.depth_l", "observation.depth_r")
STAGING_FORMAT = "arbim_rollout_staging_v2"
REWARD_MODE = "last_gripper_release_sparse"
PROVENANCE_NAME = "arbim_staging_provenance.json"


def discover_staging(sources):
    paths = []
    for arg in sources:
        path = Path(arg).expanduser().resolve()
        if path.is_dir():
            paths.extend(path.glob("*_staging.npy"))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Staging source does not exist: {path}")
    paths = sorted(set(paths))
    if not paths:
        raise ValueError("No staging NPY files found")
    return paths


def load_staging(path):
    with open(path, "rb") as fp:
        payload = np.load(fp, allow_pickle=True).item()
        if fp.read(1):
            raise ValueError(f"{path}: unexpected trailing bytes")
    if not isinstance(payload, dict) or payload.get("__format__") != STAGING_FORMAT:
        raise ValueError(f"{path}: not an ARBiM staging V2 file")
    if not payload.get("success"):
        raise ValueError(f"{path}: not a successful rollout")
    actions = np.asarray(payload["actions"], dtype=np.float32)
    length = int(payload["num_steps"])
    observations = payload["observations"]
    if actions.shape != (length, 16) or len(observations) != length + 1:
        raise ValueError(f"{path}: expected actions [T,16] and T+1 observations")
    if not np.isfinite(actions).all():
        raise ValueError(f"{path}: non-finite actions")
    if payload.get("rgb_storage") != "jpeg":
        raise ValueError(f"{path}: unsupported RGB storage")
    status = payload.get("reward_status")
    if status not in ("valid", "invalid"):
        raise ValueError(f"{path}: unknown reward_status {status!r}")
    if payload.get("reward_mode") != REWARD_MODE:
        raise ValueError(f"{path}: incompatible reward mode")
    if status == "valid":
        reward = np.asarray(payload["reward"], dtype=np.float32)
        indices = np.flatnonzero(reward)
        if (
            reward.shape != (length,) or indices.size != 1
            or reward[indices[0]] != 1 or indices[0] >= length - 1
        ):
            raise ValueError(f"{path}: invalid stored Reward V2")
        event = payload.get("reward_event")
        if not isinstance(event, dict) or event.get("frame_offset") != int(indices[0]):
            raise ValueError(f"{path}: invalid Reward V2 event metadata")
    return payload


def _decode_rgb(payload, shape, name):
    if not isinstance(payload, bytes) or not payload.startswith(b"\xff\xd8"):
        raise ValueError(f"{name}: JPEG payload missing")
    bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"{name}: failed to decode JPEG")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if tuple(rgb.shape) != (shape[1], shape[2], shape[0]):
        raise ValueError(f"{name}: JPEG shape mismatch {rgb.shape} vs {shape}")
    return np.ascontiguousarray(rgb)


def _decode_depth(payload, shape, name):
    if not isinstance(payload, bytes) or not payload.startswith(b"\x89PNG"):
        raise ValueError(f"{name}: Depth PNG missing")
    gray = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if gray is None or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(f"{name}: expected single-channel uint8 PNG")
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    if tuple(rgb.shape) != (shape[1], shape[2], shape[0]):
        raise ValueError(f"{name}: Depth dimension mismatch")
    return np.ascontiguousarray(rgb)


def _read_template(template_root, use_depth):
    template_root = Path(template_root).expanduser().resolve()
    if not (template_root / "meta").is_dir():
        raise FileNotFoundError(f"Teleop LeRobot template not found: {template_root}")
    dataset = LeRobotDataset(repo_id=template_root.name, root=template_root)
    feature_keys = ("observation.state", "action", *RGB_KEYS)
    if use_depth:
        feature_keys += DEPTH_KEYS
    features = {}
    for key in feature_keys:
        if key not in dataset.meta.features:
            raise ValueError(f"Template missing feature {key!r}")
        features[key] = copy.deepcopy(dataset.meta.features[key])
    if tuple(features["observation.state"]["shape"]) != (16,):
        raise ValueError("Template observation.state must be 16D")
    if tuple(features["action"]["shape"]) != (16,):
        raise ValueError("Template action must be 16D")
    for key in (*RGB_KEYS, *(DEPTH_KEYS if use_depth else ())):
        if features[key]["dtype"] not in ("video", "image"):
            raise ValueError(f"Template camera {key} must have image/video dtype")
        if tuple(features[key]["shape"])[0] != 3:
            raise ValueError(f"Template camera {key} must have 3 channels")
    info = dataset.meta.info
    fps = float(info["fps"])
    robot_type = info.get("robot_type", "kuavo4pro")
    return features, fps, robot_type


def convert_staging_to_lerobot(
    staging_sources, template_root, output_root, repo_id, task,
    *, use_depth=False, require_valid_reward=False,
):
    paths = discover_staging(staging_sources)
    accepted = []
    skipped = []
    for path in paths:
        payload = load_staging(path)
        if require_valid_reward and payload["reward_status"] != "valid":
            skipped.append(str(path))
            continue
        if use_depth and not payload.get("use_depth"):
            raise ValueError(f"{path}: Depth requested but missing")
        accepted.append((path, payload))
    if not accepted:
        raise ValueError("No eligible successful staging episodes")

    output_root = Path(output_root).expanduser().resolve()
    template_root = Path(template_root).expanduser().resolve()
    if output_root.exists() or output_root == template_root:
        raise FileExistsError(f"LeRobot output must be new: {output_root}")
    for path, _ in accepted:
        if path == output_root or output_root in path.parents:
            raise ValueError("Output may not overwrite an input staging directory")

    features, fps, robot_type = _read_template(template_root, use_depth)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    # Use the same creation/saving API as kuavo_data/CvtRosbag2Lerobot.py.
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(output_root),
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=True,
    )
    episodes = []
    total_frames = 0
    for ordinal, (path, sample) in enumerate(accepted):
        actions = np.asarray(sample["actions"], dtype=np.float32)
        for index, action in enumerate(actions):
            obs = sample["observations"][index]  # T actions consume obs[0:T].
            state = np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1)
            if state.shape != (16,) or not np.isfinite(state).all():
                raise ValueError(f"{path}: invalid state at frame {index}")
            frame = {
                "observation.state": torch.from_numpy(state.copy()),
                "action": torch.from_numpy(action.copy()),
            }
            for key in RGB_KEYS:
                if key not in obs["rgb"]:
                    raise ValueError(f"{path}: missing RGB camera {key}")
                shape = tuple(features[key]["shape"])
                frame[key] = _decode_rgb(obs["rgb"][key], shape, key)
            if use_depth:
                if set(obs.get("depth", {})) != set(DEPTH_KEYS):
                    raise ValueError(f"{path}: missing Depth cameras at frame {index}")
                for key in DEPTH_KEYS:
                    shape = tuple(features[key]["shape"])
                    frame[key] = _decode_depth(obs["depth"][key], shape, key)
            dataset.add_frame(frame, task=task)
        dataset.save_episode()
        total_frames += len(actions)
        # Keep the last T+1 observation in Staging, do not fabricate a dummy action.
        reward_event = sample.get("reward_event") if sample["reward_status"] == "valid" else None
        episodes.append({
            "lerobot_episode_index": ordinal,
            "source": str(path),
            "num_frames": len(actions),
            "success": True,
            "reward_mode": sample["reward_mode"],
            "reward_status": sample["reward_status"],
            "reward_event": reward_event,
            "reward_index": int(reward_event["frame_offset"]) if reward_event else None,
            "terminal_observation_in_staging_only": True,
        })
        # Teleop converter also refreshes the HF view between episodes.
        dataset.hf_dataset = dataset.create_hf_dataset()

    provenance = {
        "format": "arbim_lerobot_staging_provenance_v1",
        "fps": fps, "task": task, "use_depth": bool(use_depth),
        "source_count": len(accepted), "skipped_invalid": skipped,
        "episodes": episodes,
    }
    metadata_path = output_root / "meta" / PROVENANCE_NAME
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return {
        "lerobot_root": str(output_root), "num_episodes": len(accepted),
        "num_frames": total_frames, "skipped_invalid": len(skipped),
        "provenance": str(metadata_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staging", nargs="+", help="Staging files or directories")
    parser.add_argument("--template-lerobot-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", default="ARBiM Data Wheel")
    parser.add_argument("--use-depth", action="store_true")
    parser.add_argument("--require-valid-reward", action="store_true")
    args = parser.parse_args()
    print(convert_staging_to_lerobot(
        args.staging, args.template_lerobot_root, args.output_root,
        args.repo_id, args.task,
        use_depth=args.use_depth,
        require_valid_reward=args.require_valid_reward,
    ))


if __name__ == "__main__":
    main()
