from collections.abc import Callable, Collection, Mapping
from pathlib import Path

from pandas import notna
import torch
import torchvision
import torchvision.transforms.functional

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _is_depth_policy_feature(key: str, feature) -> bool:
    """Return whether a policy feature represents a depth observation."""
    feature_type = getattr(feature, "type", None)
    depth_type = getattr(FeatureType, "DEPTH", None)
    feature_type_value = getattr(feature_type, "value", feature_type)
    return (
        (depth_type is not None and feature_type == depth_type)
        or feature_type_value == "DEPTH"
        or "depth" in key.lower()
    )


def filter_depth_policy_features(
    policy_features: Mapping[str, object], use_depth: bool
) -> tuple[dict[str, object], set[str]]:
    """Filter disabled depth features and return the keys excluded from the loader."""
    if use_depth:
        return dict(policy_features), set()

    excluded_keys = {
        key for key, feature in policy_features.items() if _is_depth_policy_feature(key, feature)
    }
    active_features = {
        key: feature for key, feature in policy_features.items() if key not in excluded_keys
    }
    return active_features, excluded_keys


def policy_uses_depth(policy_cfg) -> bool:
    """Read the optional policy.custom.use_depth flag with a safe false default."""
    if isinstance(policy_cfg, Mapping):
        custom = policy_cfg.get("custom", {})
    else:
        custom = getattr(policy_cfg, "custom", {})

    if isinstance(custom, Mapping):
        return bool(custom.get("use_depth", False))
    return bool(getattr(custom, "use_depth", False))


class CustomLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        excluded_keys: Collection[str] | None = None,
    ):
        # This must be set before the parent constructor calls load_hf_dataset().
        self.excluded_keys = frozenset(excluded_keys or ())
        super().__init__(repo_id,
                         root,
                         episodes,
                         image_transforms,
                         delta_timestamps,
                         tolerance_s,
                         revision,
                         force_cache_sync,
                         download_videos,
                         video_backend,
                         batch_encoding_size,
                         )

    @property
    def features(self) -> dict[str, dict]:
        """Expose only the metadata features that the loader is allowed to read."""
        return {
            key: feature
            for key, feature in self.meta.features.items()
            if key not in self.excluded_keys
        }

    @property
    def active_video_keys(self) -> list[str]:
        """Video keys that remain active after applying the exclusion filter."""
        return [key for key in self.meta.video_keys if key not in self.excluded_keys]

    @property
    def active_camera_keys(self) -> list[str]:
        """Camera keys that remain active after applying the exclusion filter."""
        return [key for key in self.meta.camera_keys if key not in self.excluded_keys]

    def load_hf_dataset(self):
        """Load only active parquet columns and remove excluded columns defensively."""
        hf_dataset = super().load_hf_dataset()
        column_names = getattr(hf_dataset, "column_names", ())
        columns_to_remove = [key for key in column_names if key in self.excluded_keys]
        if columns_to_remove:
            hf_dataset = hf_dataset.remove_columns(columns_to_remove)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check cached data without requiring disabled video files."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }
        requested_episodes = (
            set(range(self.meta.total_episodes)) if self.episodes is None else set(self.episodes)
        )
        if not requested_episodes.issubset(available_episodes):
            return False

        for ep_idx in requested_episodes:
            for video_key in self.active_video_keys:
                video_path = self.root / self.meta.get_video_file_path(ep_idx, video_key)
                if not video_path.exists():
                    return False
        return True

    def get_episodes_file_paths(self) -> list[str]:
        """Return data/video paths without downloading excluded videos."""
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        file_paths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        video_paths = [
            str(self.meta.get_video_file_path(ep_idx, video_key))
            for video_key in self.active_video_keys
            for ep_idx in episodes
        ]
        return list({*file_paths, *video_paths})

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        """Build timestamp queries only for active video keys."""
        query_timestamps = {}
        for key in self.active_video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Query active non-video columns only."""
        active_query_indices = {
            key: values
            for key, values in query_indices.items()
            if key not in self.excluded_keys and key not in self.meta.video_keys
        }
        return super()._query_hf_dataset(active_query_indices)

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int):
        """Decode only active video keys, even if a caller passes extra queries."""
        active_query_timestamps = {
            key: values for key, values in query_timestamps.items() if key in self.active_video_keys
        }
        return super()._query_videos(active_query_timestamps, ep_idx)
    

    def __getitem__(self, idx) -> dict:
        self._ensure_hf_dataset_loaded()
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        # print("before", item.keys())

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            # print("padding",item.keys(),padding.keys(),query_indices.keys())
            
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
            

        if len(self.active_video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            # print("video",video_frames.keys(),item.keys())
            item = {**video_frames, **item}
        # print("after", item.keys())
        # raise ValueError()
        # img = item["observation.images.wrist_cam_l"]
        # depth = item["observation.depth_l"]
        # print(img.shape,img.dtype,img.max())
        # print(depth.shape,depth.dtype,depth.max())
        # import cv2
        # import numpy as np
        # cv2.imwrite("outputs/images/img.png",cv2.cvtColor((img[0].permute(1,2,0).numpy()*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
        # np.save("outputs/images/depth.npy",depth.numpy()[0,0,...].astype(np.uint16))
        # raise ValueError()

        if self.image_transforms is not None:
            image_keys = self.active_camera_keys
            depth_keys = [
                key
                for key in item.keys()
                if key in image_keys and "depth" in key and "is_pad" not in key
            ]

            if len(depth_keys)==0:
                for cam in image_keys:
                    item[cam], _no, __no_use = self.image_transforms(item[cam])
            else:
                for rgb_cam, depth_cam in zip(image_keys, depth_keys):
                    item[rgb_cam], crop_position, resize_shape = self.image_transforms(item[rgb_cam])

                    # Crop depth
                    if crop_position is not None:
                        if isinstance(crop_position, (list, tuple)) and len(crop_position) == 4:
                            item[depth_cam] = torchvision.transforms.functional.crop(item[depth_cam], *crop_position)
                        else:
                            # If crop_position is an int or single value
                            item[depth_cam] = torchvision.transforms.functional.center_crop(item[depth_cam], crop_position)

                    # resize depth
                    if resize_shape is not None:
                        item[depth_cam] = torchvision.transforms.functional.resize(item[depth_cam], resize_shape,torchvision.transforms.InterpolationMode.NEAREST)



        # Add task as a string
        task_idx = int(item["task_index"].item())
        tasks = self.meta.tasks
        if hasattr(tasks, "columns") and "task_index" in tasks.columns:
            matches = tasks.index[tasks["task_index"] == task_idx]
            if len(matches) != 1:
                raise KeyError(
                    f"Expected exactly one task for task_index={task_idx}, got {len(matches)}"
                )
            item["task"] = str(matches[0])
        else:
            item["task"] = tasks[task_idx]

        return item
