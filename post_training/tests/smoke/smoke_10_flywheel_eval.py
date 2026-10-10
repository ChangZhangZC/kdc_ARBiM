"""Batch 06: exercise the existing MuJoCo eval episode with lightweight mocks.

No ROS/MuJoCo needed. Extract the actual run_single_episode function from
sim_auto_test.py by AST to avoid importing hardware and LeRobot modules.
This checks Data Wheel wiring, normal-mode isolation and abnormal cleanup.
"""
import ast
import gc
import os
import pathlib
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kuavo_deploy.src.eval.rollout_recorder import CAMERA_KEYS

SCRIPT = ROOT / "kuavo_deploy" / "src" / "eval" / "sim_auto_test.py"


def observation(value):
    obs = {"observation.state": torch.full((1, 16), float(value))}
    for key in CAMERA_KEYS:
        obs[key] = torch.full((1, 3, 12, 12), 0.4)
    return obs


def actions():
    values = np.zeros((14, 16), dtype=np.float32)
    values[3:6, 7] = 1.0
    values[4:8, 15] = 1.0
    return values


class StubEnv:
    def __init__(self, success, failure_step=None):
        self.success = success
        self.failure_step = failure_step
        self.current = 0
        self.closed = False
        self.unwrapped = self
        self.ros_rate = 10
        self.average_sleep_time = 0.0

    def reset(self, seed):
        self.current = 0
        return observation(0), {}

    def check_action(self, action):
        return np.clip(action, 0, 1)

    def step(self, action):
        if self.failure_step is not None and self.current == self.failure_step:
            raise RuntimeError("simulated step exception")
        self.current += 1
        terminal = self.current == len(actions())
        if terminal and self.success:
            SUCCESS_EVENT.set()
        return observation(self.current), 0, terminal, False, {}

    def close(self):
        self.closed = True


class StubROSManager:
    instances = []

    def __init__(self):
        self.closed = False
        self.registered = []
        self.instances.append(self)

    def register_subscriber(self, topic, msg_type, callback):
        self.registered.append(topic)

    def close(self):
        self.closed = True


class StubImageIO:
    @staticmethod
    def imwrite(path, image):
        pathlib.Path(path).write_bytes(b"stub")

    @staticmethod
    def imread(path):
        return np.zeros((12, 12, 3), dtype=np.uint8)

    @staticmethod
    def mimsave(path, frames, fps):
        pathlib.Path(path).write_bytes(b"video")


class StubLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class StubPolicy:
    def __init__(self):
        self.index = 0

    def reset(self):
        self.index = 0

    def select_action(self, obs):
        value = actions()[self.index]
        self.index += 1
        return torch.from_numpy(value).unsqueeze(0)


SUCCESS_EVENT = threading.Event()


def compile_episode(env, allow_steps=True):
    text = SCRIPT.read_text()
    tree = ast.parse(text, filename=str(SCRIPT))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_single_episode")
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    ros = StubROSManager
    ns = {
        "gym": types.SimpleNamespace(make=lambda *args, **kwargs: env),
        "ROSManager": ros,
        "rospy": types.SimpleNamespace(ServiceProxy=lambda *args: lambda req: None),
        "Trigger": type("Trigger", (), {}),
        "TriggerRequest": type("TriggerRequest", (), {}),
        "Bool": type("Bool", (), {}),
        "env_success_callback": lambda msg: None,
        "success_evt": SUCCESS_EVENT,
        "torch": torch,
        "np": np,
        "time": time,
        "gc": gc,
        "os": os,
        "imageio": StubImageIO,
        "log_model": StubLogger(),
        "log_robot": StubLogger(),
        "check_control_signals": lambda: allow_steps,
    }
    exec(compile(module, str(SCRIPT), "exec"), ns)
    return ns["run_single_episode"]


class TestFlywheelEpisodeIntegration(unittest.TestCase):
    def setUp(self):
        SUCCESS_EVENT.clear()
        StubROSManager.instances.clear()
        self.cfg = types.SimpleNamespace(
            inference=types.SimpleNamespace(
                seed=0, task="test", policy_type="client", max_episode_steps=20
            ),
            env=types.SimpleNamespace(env_name="StubEnv"),
        )

    def run_episode(self, folder, *, enabled, success, failure_step=None, allow_steps=True):
        env = StubEnv(success=success, failure_step=failure_step)
        func = compile_episode(env, allow_steps=allow_steps)
        policy = StubPolicy()
        with patch.dict(os.environ, {"ARBIM_DATA_WHEEL": "1" if enabled else "0"}):
            outcome = func(self.cfg, policy, lambda x: x, lambda x: x, 0, folder)
        return outcome, env

    def assert_clean(self, env):
        self.assertTrue(env.closed)
        self.assertEqual(len(StubROSManager.instances), 1)
        self.assertTrue(StubROSManager.instances[0].closed)
        self.assertEqual(StubROSManager.instances[0].registered, ["/simulator/success"])

    def test_success_collects_staging_without_auto_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            result, env = self.run_episode(directory, enabled=True, success=True)
            self.assertEqual(result, 1)
            self.assert_clean(env)
            files = list((directory / "data_wheel" / "staging").glob("*_staging.npy"))
            self.assertEqual(len(files), 1)
            saved = np.load(files[0], allow_pickle=True).item()
            self.assertEqual(saved["num_steps"], 14)
            self.assertEqual(len(saved["observations"]), 15)
            self.assertEqual(saved["reward_status"], "valid")
            self.assertEqual(np.flatnonzero(saved["reward"]).tolist(), [8])
            self.assertEqual(saved["actions"].shape, (14, 16))
            self.assertFalse(list(directory.glob("**/*processed*.npy")))
            self.assertFalse(list(directory.glob("**/*.zarr")))

    def test_failure_never_writes_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            result, env = self.run_episode(directory, enabled=True, success=False)
            self.assertEqual(result, 0)
            self.assert_clean(env)
            self.assertFalse(list(directory.glob("**/*_staging.npy")))

    def test_stop_closes_resources_and_discards(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            result, env = self.run_episode(
                directory, enabled=True, success=False, allow_steps=False
            )
            self.assertEqual(result, 0)
            self.assert_clean(env)
            self.assertFalse(list(directory.glob("**/*_staging.npy")))

    def test_exception_closes_resources_and_discards(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            env = StubEnv(success=False, failure_step=2)
            func = compile_episode(env)
            with patch.dict(os.environ, {"ARBIM_DATA_WHEEL": "1"}):
                with self.assertRaisesRegex(RuntimeError, "simulated step exception"):
                    func(self.cfg, StubPolicy(), lambda x: x, lambda x: x, 0, directory)
            self.assert_clean(env)
            self.assertFalse(list(directory.glob("**/*_staging.npy")))

    def test_normal_eval_never_writes_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            result, env = self.run_episode(directory, enabled=False, success=True)
            self.assertEqual(result, 1)
            self.assert_clean(env)
            self.assertFalse(list(directory.glob("**/*_staging.npy")))
            self.assertEqual(len(list(directory.glob("*.mp4"))), len(CAMERA_KEYS))


if __name__ == "__main__":
    unittest.main()
