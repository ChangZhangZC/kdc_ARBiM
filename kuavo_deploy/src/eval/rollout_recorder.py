"""Capture successful MuJoCo episodes for later Processed NPY conversion.

This is an intermediate staging format, NOT arbim_processed_npy_stream_v2.
Reward V2 is annotated here; training-ready NPY export belongs to Batch 04.
"""

import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np


CAMERA_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
)
STAGING_FORMAT = "arbim_rollout_staging_v2"
REWARD_MODE = "last_gripper_release_sparse"


def _numpy_copy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def _snapshot(observation, jpeg_quality):
    """Freeze an unnormalized observation before policy preprocessing."""
    state = _numpy_copy(observation["observation.state"]).astype(np.float32)
    if state.ndim == 2 and state.shape[0] == 1:
        state = state[0]
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError(f"Invalid state shape/values: {state.shape}")

    rgb = {}
    for key in CAMERA_KEYS:
        image = _numpy_copy(observation[key])
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"{key} must be CHW RGB, got {image.shape}")
        if np.issubdtype(image.dtype, np.floating):
            if not np.all(np.isfinite(image)):
                raise ValueError(f"{key} contains nonfinite pixels")
            image = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            raise ValueError(f"{key} must be uint8 or float RGB, got {image.dtype}")
        rgb_hwc = np.ascontiguousarray(image.transpose(1, 2, 0))
        bgr = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if not ok:
            raise RuntimeError(f"JPEG encoding failed for {key}")
        rgb[key] = encoded.tobytes()
    return {"agent_pos": state, "rgb": rgb}



def _infer_gripper_thresholds(values, *, open_side="auto", start_frames=5):
    """Match data_prepare.py Reward V2 thresholds for one complete episode."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Gripper action contains no finite values")

    q10, q90 = np.quantile(finite, [0.10, 0.90]).tolist()
    span = q90 - q10
    if span <= 1e-8:
        raise ValueError(
            f"Gripper action has almost no dynamic range: q10={q10}, q90={q90}"
        )
    low = q10 + 0.35 * span
    high = q10 + 0.65 * span

    start_median = float(np.median(values[: min(start_frames, len(values))]))
    if open_side == "auto":
        open_side = (
            "low" if abs(start_median - q10) <= abs(start_median - q90) else "high"
        )
    if open_side not in {"low", "high"}:
        raise ValueError(f"Invalid gripper open_side: {open_side!r}")
    return {
        "low": float(low),
        "high": float(high),
        "open_side": open_side,
        "q10": float(q10),
        "q90": float(q90),
        "start_median": start_median,
    }


def _classify_gripper(value, thresholds):
    value = float(value)
    if value <= thresholds["low"]:
        side = "low"
    elif value >= thresholds["high"]:
        side = "high"
    else:
        return None
    return "open" if side == thresholds["open_side"] else "closed"


def _debounced_gripper_transitions(values, thresholds, min_dwell=3):
    """Copy the Reward V2 state machine, including transition start offsets."""
    state = None
    candidate = None
    candidate_start = None
    candidate_count = 0
    transitions = []

    for index, value in enumerate(np.asarray(values, dtype=np.float64)):
        classified = _classify_gripper(value, thresholds)
        if classified is None:
            continue

        if state is None:
            if candidate == classified:
                candidate_count += 1
            else:
                candidate = classified
                candidate_start = index
                candidate_count = 1
            if candidate_count >= min_dwell:
                state = classified
                candidate = None
                candidate_start = None
                candidate_count = 0
            continue

        if classified == state:
            candidate = None
            candidate_start = None
            candidate_count = 0
            continue

        if candidate == classified:
            candidate_count += 1
        else:
            candidate = classified
            candidate_start = index
            candidate_count = 1

        if candidate_count >= min_dwell:
            transitions.append({
                "frame_offset": int(candidate_start),
                "from_state": state,
                "to_state": classified,
            })
            state = classified
            candidate = None
            candidate_start = None
            candidate_count = 0

    return transitions


def _annotate_reward_v2(actions):
    """Use existing Reward V2 semantics on a successful rollout's actions.

    Requires exactly one closed-to-open event per gripper. +1 is placed at the
    later release event, *not* at the MuJoCo success or episode terminal step.
    Per-episode threshold estimation can differ from the LeRobot dataset-wide
    calibration; no reward is fabricated if detection is invalid.
    """
    action = np.asarray(actions, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] <= 15:
        raise ValueError(f"Expected action [T,D] with D>=16, got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("Action contains nonfinite values")

    left_thresholds = _infer_gripper_thresholds(
        action[:, 7], open_side="auto", start_frames=5
    )
    right_thresholds = _infer_gripper_thresholds(
        action[:, 15], open_side="auto", start_frames=5
    )
    left_transitions = _debounced_gripper_transitions(
        action[:, 7], left_thresholds, min_dwell=3
    )
    right_transitions = _debounced_gripper_transitions(
        action[:, 15], right_thresholds, min_dwell=3
    )
    left_releases = [
        item["frame_offset"] for item in left_transitions
        if item["from_state"] == "closed" and item["to_state"] == "open"
    ]
    right_releases = [
        item["frame_offset"] for item in right_transitions
        if item["from_state"] == "closed" and item["to_state"] == "open"
    ]
    if len(left_releases) != 1 or len(right_releases) != 1:
        raise ValueError(
            "Reward V2 requires one release per gripper: "
            f"left_release={left_releases}, right_release={right_releases}"
        )

    release_index = max(left_releases[-1], right_releases[-1])
    if release_index >= len(action) - 1:
        raise ValueError(
            f"Reward release index={release_index} must precede final "
            f"recorded action index={len(action) - 1}"
        )
    reward = np.zeros(len(action), dtype=np.float32)
    reward[release_index] = 1.0
    metadata = {
        "frame_offset": int(release_index),
        "left_release": int(left_releases[-1]),
        "right_release": int(right_releases[-1]),
        "left_thresholds": left_thresholds,
        "right_thresholds": right_thresholds,
        "min_dwell": 3,
        "expected_cycles": 1,
        "open_infer_frames": 5,
    }
    return reward, metadata



class RolloutRecorder:
    """Capture (obs_t, executed_action_t, obs_t+1); persist successes only."""

    def __init__(self, output_dir, episode, jpeg_quality=95):
        self.output_dir = Path(output_dir)
        self.episode = int(episode)
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self._observations = []
        self._actions = []
        self._active = False

    def start_episode(self, observation):
        if self._active:
            raise RuntimeError("Recorder episode already active")
        self._observations = [_snapshot(observation, self.jpeg_quality)]
        self._actions = []
        self._active = True

    def append_step(self, action, next_observation):
        if not self._active:
            raise RuntimeError("Recorder episode not active")
        action = _numpy_copy(action).astype(np.float32).reshape(-1)
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains nonfinite values")
        if self._actions and action.shape != self._actions[0].shape:
            raise ValueError("Action dimension changed within episode")
        following = _snapshot(next_observation, self.jpeg_quality)
        self._actions.append(action)
        self._observations.append(following)

    def finish_episode(self, success):
        if not self._active:
            raise RuntimeError("Recorder episode not active")
        try:
            if not success or not self._actions:
                return None
            self.output_dir.mkdir(parents=True, exist_ok=True)
            dest = self.output_dir / (
                f"rollout_{self.episode:05d}_{time.time_ns()}_staging.npy"
            )
            tmp = dest.with_suffix(".tmp")
            actions = np.stack(self._actions).astype(np.float32)
            try:
                reward, reward_event = _annotate_reward_v2(actions)
                reward_status = "valid"
                reward_error = None
            except ValueError as exc:
                # Preserve successful behavior data; Batch 04 must reject a
                # trajectory without valid Reward V2 rather than invent +1.
                reward = None
                reward_event = None
                reward_status = "invalid"
                reward_error = str(exc)
                warnings.warn(
                    f"Data Wheel episode {self.episode}: Reward V2 failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            payload = {
                "__format__": STAGING_FORMAT,
                "episode": self.episode,
                "success": True,
                "rgb_storage": "jpeg",
                "num_steps": len(self._actions),
                "observations": self._observations,
                "actions": actions,
                "reward_mode": REWARD_MODE,
                "reward": reward,
                "reward_event": reward_event,
                "reward_status": reward_status,
                "reward_error": reward_error,
            }
            try:
                with tmp.open("wb") as handle:
                    np.save(handle, payload, allow_pickle=True)
                os.replace(tmp, dest)
            finally:
                if tmp.exists():
                    tmp.unlink()
            return dest
        finally:
            self.abort()

    def abort(self):
        self._observations = []
        self._actions = []
        self._active = False
