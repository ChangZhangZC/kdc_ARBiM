"""Verify Data Wheel success-target loop without ROS/MuJoCo.

Extract the actual kuavo_eval_autotest() function via AST, supplying fake
services and a deterministic sequence of success/failure episode results.
"""
import ast
import datetime
import gc
import itertools
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "kuavo_deploy" / "src" / "eval" / "sim_auto_test.py"


class AlwaysReady:
    def is_set(self):
        return True

    def clear(self):
        pass


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class DummyPolicy:
    def reset(self):
        pass


class DummyService:
    def shutdown(self):
        pass


def create_runner(results):
    node = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    fn = next(item for item in node.body
              if isinstance(item, ast.FunctionDef) and item.name == "kuavo_eval_autotest")
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    attempts = []
    iterator = iter(results)

    def run_single_episode(config, policy, preprocessor, postprocessor, episode, output_directory):
        attempts.append(episode)
        return next(iterator)

    ns = {
        "os": os, "itertools": itertools, "datetime": datetime, "Path": Path,
        "time": __import__("time"), "gc": gc, "torch": torch,
        "log_model": DummyLogger(), "log_robot": DummyLogger(),
        "rospy": types.SimpleNamespace(
            ServiceProxy=lambda *args: lambda *args: None,
            Service=lambda *args: DummyService(),
        ),
        "Trigger": object, "init_evt": AlwaysReady(), "success_evt": AlwaysReady(),
        "env_init_service": lambda req: None,
        "pause_sub": types.SimpleNamespace(unregister=lambda: None),
        "stop_sub": types.SimpleNamespace(unregister=lambda: None),
        "check_control_signals": lambda: True,
        "set_seed": lambda seed: None,
        "resolve_pretrained_path": lambda cfg: Path("/unused/model"),
        "setup_policy": lambda *args, **kwargs: DummyPolicy(),
        "build_pre_post_processors": lambda *args: (lambda obs: obs, lambda act: act),
        "safe_reset_service": lambda service: None,
        "run_single_episode": run_single_episode,
        "KuavoConfig": object,
    }
    exec(compile(module, str(SCRIPT), "exec"), ns)
    return ns["kuavo_eval_autotest"], attempts


class TestTargetSuccessCount(unittest.TestCase):
    def setUp(self):
        self.cfg = types.SimpleNamespace(inference=types.SimpleNamespace(
            task="fake_sim", method="fake_model", timestamp="test_run",
            epoch=0, eval_episodes=2, seed=0, policy_type="client", device="cpu"
        ))

    def run_mock(self, *, target=None):
        fn, attempts = create_runner([0, 1, 0, 1, 1])
        environment = {"ARBIM_DATA_WHEEL": "1" if target is not None else "0"}
        if target is not None:
            environment["ARBIM_FLYWHEEL_SUCCESS_TARGET"] = str(target)
        with tempfile.TemporaryDirectory() as tmp:
            before = os.getcwd()
            try:
                os.chdir(tmp)
                with patch.dict(os.environ, environment):
                    fn(self.cfg)
                eval_base = Path("outputs/eval/fake_sim/fake_model/test_run")
                output_name = "epoch0_flywheel" if target is not None else "epoch0"
                self.assertTrue((eval_base / output_name).is_dir())
                self.assertFalse((eval_base / ("epoch0" if target is not None else "epoch0_flywheel")).exists())
                log = (eval_base / output_name / "evaluation_autotest.log").read_text()
            finally:
                os.chdir(before)
        return attempts, log

    def test_flywheel_waits_for_three_successes_in_five_attempts(self):
        attempts, log = self.run_mock(target=3)
        self.assertEqual(attempts, [0, 1, 2, 3, 4])
        self.assertIn("Target successful rollouts: 3", log)
        self.assertIn("Success Count: 3 / Already eval episodes: 5", log)

    def test_normal_eval_is_still_limited_by_eval_episodes(self):
        attempts, log = self.run_mock(target=None)
        self.assertEqual(attempts, [0, 1])
        self.assertIn("Total Episodes: 2", log)
        self.assertIn("Success Count: 1 / Already eval episodes: 2", log)

    def test_target_must_be_positive(self):
        runner, attempts = create_runner([1])
        with patch.dict(os.environ, {
            "ARBIM_DATA_WHEEL": "1",
            "ARBIM_FLYWHEEL_SUCCESS_TARGET": "0",
        }):
            with self.assertRaisesRegex(ValueError, "target must be >= 1"):
                runner(self.cfg)
        self.assertEqual(attempts, [])


if __name__ == "__main__":
    unittest.main()
