"""Capture successful MuJoCo episodes for later Processed NPY conversion.

This is an intermediate staging format, NOT arbim_processed_npy_stream_v2.
Reward annotation and training-ready NPY export belong to later batches.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np


CAMERA_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
)
STAGING_FORMAT = "arbim_rollout_staging_v1"


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
            payload = {
                "__format__": STAGING_FORMAT,
                "episode": self.episode,
                "success": True,
                "rgb_storage": "jpeg",
                "num_steps": len(self._actions),
                "observations": self._observations,
                "actions": np.stack(self._actions).astype(np.float32),
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
