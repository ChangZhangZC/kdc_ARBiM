import ast
from pathlib import Path
from types import SimpleNamespace

import torch

import lerobot_patches.custom_patches  # noqa: F401
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from kuavo_train.wrapper.dataset.LeRobotDatasetWrapper import (
    CustomLeRobotDataset,
    filter_depth_policy_features,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _policy_features():
    return {
        "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 8, 8)),
        "observation.depth.front": PolicyFeature(FeatureType.DEPTH, (1, 8, 8)),
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        "action": PolicyFeature(FeatureType.ACTION, (2,)),
    }


def test_disabled_depth_is_removed_from_policy_features():
    features = _policy_features()

    active_features, excluded_keys = filter_depth_policy_features(features, use_depth=False)

    assert set(active_features) == {
        "observation.images.front",
        "observation.state",
        "action",
    }
    assert excluded_keys == {"observation.depth.front"}
    assert set(features) == set(_policy_features())


def test_enabled_depth_keeps_policy_features():
    features = _policy_features()

    active_features, excluded_keys = filter_depth_policy_features(features, use_depth=True)

    assert active_features == features
    assert excluded_keys == set()


def test_disabled_depth_is_not_in_video_timestamp_queries():
    dataset = CustomLeRobotDataset.__new__(CustomLeRobotDataset)
    dataset.excluded_keys = frozenset({"observation.depth.front"})
    dataset.meta = SimpleNamespace(
        video_keys=["observation.images.front", "observation.depth.front"]
    )

    timestamps = dataset._get_query_timestamps(current_ts=1.5)

    assert timestamps == {"observation.images.front": [1.5]}


def test_disabled_depth_video_is_not_decoded(monkeypatch):
    dataset = CustomLeRobotDataset.__new__(CustomLeRobotDataset)
    dataset.excluded_keys = frozenset({"observation.depth.front"})
    dataset.root = Path("/tmp/unused-depth-test")
    dataset.tolerance_s = 1e-4
    dataset.video_backend = "pyav"
    dataset.meta = SimpleNamespace(
        video_keys=["observation.images.front", "observation.depth.front"],
        episodes=[
            {
                "videos/observation.images.front/from_timestamp": 0.0,
                "videos/observation.depth.front/from_timestamp": 0.0,
            }
        ],
    )
    dataset.meta.get_video_file_path = lambda _ep_idx, key: Path(f"{key}.mp4")
    decoded_keys = []

    def fake_decode(path, timestamps, tolerance_s, video_backend):
        decoded_keys.append(path.stem)
        return torch.zeros(1, 1, 2, 2)

    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.decode_video_frames", fake_decode)

    dataset._query_videos(
        {
            "observation.images.front": [1.5],
            "observation.depth.front": [1.5],
        },
        ep_idx=0,
    )

    assert decoded_keys == ["observation.images.front"]


def test_disabled_depth_column_is_removed_from_hf_dataset(monkeypatch):
    dataset = CustomLeRobotDataset.__new__(CustomLeRobotDataset)
    dataset.excluded_keys = frozenset({"observation.depth.front"})

    class FakeDataset:
        column_names = ["observation.images.front", "observation.depth.front", "action"]

        def __init__(self):
            self.removed_columns = []

        def remove_columns(self, columns):
            self.removed_columns.append(list(columns))
            return self

    fake_dataset = FakeDataset()
    monkeypatch.setattr(LeRobotDataset, "load_hf_dataset", lambda _self: fake_dataset)

    loaded_dataset = dataset.load_hf_dataset()

    assert loaded_dataset is fake_dataset
    assert fake_dataset.removed_columns == [["observation.depth.front"]]


def test_episode_file_paths_are_strings_deduplicated_and_depth_free():
    dataset = CustomLeRobotDataset.__new__(CustomLeRobotDataset)
    dataset.excluded_keys = frozenset({"observation.depth.front"})
    dataset.episodes = [0, 1]
    dataset.meta = SimpleNamespace(
        video_keys=["observation.images.front", "observation.depth.front"],
        get_data_file_path=lambda episode: Path(f"data/{episode}.parquet"),
        get_video_file_path=lambda _episode, key: Path(f"videos/{key}.mp4"),
    )

    paths = dataset.get_episodes_file_paths()

    assert all(isinstance(path, str) for path in paths)
    assert set(paths) == {
        "data/0.parquet",
        "data/1.parquet",
        "videos/observation.images.front.mp4",
    }
    assert len(paths) == 3


def test_entrypoints_use_shared_filter_and_custom_dataset():
    for filename in (
        "kuavo_train/train_policy.py",
        "kuavo_train/train_policy_with_accelerate.py",
    ):
        source = (REPO_ROOT / filename).read_text()
        tree = ast.parse(source, filename=filename)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "CustomLeRobotDataset"
        ]

        assert calls, f"{filename} must instantiate CustomLeRobotDataset"
        assert "filter_depth_policy_features" in source
        assert "excluded_keys" in source


def test_delta_timestamps_only_include_active_policy_keys():
    metadata = SimpleNamespace(
        info={
            "features": {
                "observation.images.front": {},
                "observation.depth.front": {},
                "observation.state": {},
                "action": {},
            }
        },
        fps=10,
    )
    policy_cfg = SimpleNamespace(
        observation_delta_indices=[-1, 0],
        action_delta_indices=[0, 1],
    )
    active_input_features, _ = filter_depth_policy_features(
        {
            "observation.images.front": _policy_features()["observation.images.front"],
            "observation.depth.front": _policy_features()["observation.depth.front"],
            "observation.state": _policy_features()["observation.state"],
        },
        use_depth=False,
    )
    output_features = {"action": _policy_features()["action"]}

    for filename in (
        "kuavo_train/train_policy.py",
        "kuavo_train/train_policy_with_accelerate.py",
    ):
        source = (REPO_ROOT / filename).read_text()
        namespace = {}
        exec(compile(ast.Module(body=[node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "build_delta_timestamps"], type_ignores=[]), filename, "exec"), namespace)
        delta_timestamps = namespace["build_delta_timestamps"](
            metadata,
            policy_cfg,
            input_features=active_input_features,
            output_features=output_features,
        )
        assert set(delta_timestamps) == {
            "observation.images.front",
            "observation.state",
            "action",
        }
