from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import zarr

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POST_RL_SRC = REPO_ROOT / "post_training" / "src"
for path in (REPO_ROOT, POST_RL_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from post_rl.data.reward_zarr_migration import (
    clone_latent_cache_for_reward_clone,
    compute_discounted_return,
    migrate_reward_zarr,
)


def _episode_action(length: int, left_closed: tuple[int, int], right_closed: tuple[int, int]):
    action = np.zeros((length, 16), dtype=np.float32)
    action[left_closed[0]:left_closed[1], 7] = 1.0
    action[right_closed[0]:right_closed[1], 15] = 1.0
    return action


def _create_source_zarr(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    ep0 = _episode_action(14, (3, 7), (4, 9))
    ep1 = _episode_action(13, (3, 6), (4, 8))
    action = np.concatenate([ep0, ep1], axis=0)
    episode_ends = np.asarray([len(ep0), len(ep0) + len(ep1)], dtype=np.int64)
    n = len(action)

    done = np.zeros(n, dtype=bool)
    done[episode_ends - 1] = True
    reward = done.astype(np.float32)
    returns = compute_discounted_return(reward, done, 0.99)

    root = zarr.group(str(path))
    data = root.create_group("data")
    meta = root.create_group("meta")
    for name, array in {
        "action": action,
        "reward": reward[:, None],
        "return": returns[:, None],
        "done": done[:, None],
        "timeout": done[:, None],
        "state": np.arange(n * 2, dtype=np.float32).reshape(n, 2),
    }.items():
        z = data.create_dataset(
            name,
            shape=array.shape,
            dtype=array.dtype,
            chunks=(min(8, n), *array.shape[1:]),
            overwrite=True,
        )
        z[:] = array
    ep = meta.create_dataset(
        "episode_ends",
        shape=episode_ends.shape,
        dtype=episode_ends.dtype,
        chunks=episode_ends.shape,
        overwrite=True,
    )
    ep[:] = episode_ends
    root.attrs["marker"] = "source_unchanged"
    return action, episode_ends


def _create_source_cache(
    path: pathlib.Path,
    source_zarr: pathlib.Path,
    episode_ends: np.ndarray,
) -> np.ndarray:
    path.mkdir(parents=True)
    n = int(episode_ends[-1])
    latent = np.arange(n * 6, dtype=np.float32).reshape(n, 2, 3)
    next_indices = np.arange(n, dtype=np.int64) + 1
    next_indices[episode_ends - 1] = episode_ends - 1
    np.save(path / "obs_latent.npy", latent)
    np.save(path / "next_indices.npy", next_indices)

    metadata = {
        "version": 1,
        "source_dataset": os.path.realpath(source_zarr),
        "encoder_sha256": "encoder-test",
        "normalizer_sha256": "normalizer-test",
        "use_depth": False,
        "feature_dim": 3,
        "size": n,
        "episode_ends_sha256": hashlib.sha256(
            np.asarray(episode_ends, dtype=np.int64).tobytes()
        ).hexdigest(),
        "transition_alignment": "next_obs=obs[next_index]; terminal=self",
        "latent_shape": [2, 3],
        "dtype": "float32",
    }
    with (path / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)
    return latent


def main() -> None:
    work = pathlib.Path(tempfile.mkdtemp(prefix="arbim_reward_v2_migration_"))
    source = work / "source.zarr"
    target = work / "target_reward_v2.zarr"
    action, episode_ends = _create_source_zarr(source)

    source_reward_before = np.asarray(
        zarr.open_group(str(source), mode="r")["data"]["reward"][:]
    ).copy()

    summary = migrate_reward_zarr(
        source,
        target,
        gamma=0.99,
        horizon=8,
        left_gripper_index=7,
        right_gripper_index=15,
        open_infer_frames=3,
        min_dwell=3,
        expected_cycles=1,
        copy_mode="copy",
    )

    assert summary["frames"] == len(action)
    assert summary["episodes"] == 2
    assert summary["reward_count"] == 2
    assert summary["reward_sum"] == 2.0
    assert summary["final_h_chunk_contains_reward_fraction"] == 1.0

    source_root = zarr.open_group(str(source), mode="r")
    target_root = zarr.open_group(str(target), mode="r")
    np.testing.assert_array_equal(
        np.asarray(source_root["data"]["reward"][:]),
        source_reward_before,
    )
    assert source_root.attrs["marker"] == "source_unchanged"

    target_reward = np.asarray(target_root["data"]["reward"][:]).reshape(-1)
    expected_reward = np.zeros(len(action), dtype=np.float32)
    expected_reward[9] = 1.0
    expected_reward[14 + 8] = 1.0
    np.testing.assert_array_equal(target_reward, expected_reward)

    source_done = np.asarray(source_root["data"]["done"][:])
    target_done = np.asarray(target_root["data"]["done"][:])
    source_timeout = np.asarray(source_root["data"]["timeout"][:])
    target_timeout = np.asarray(target_root["data"]["timeout"][:])
    np.testing.assert_array_equal(target_done, source_done)
    np.testing.assert_array_equal(target_timeout, source_timeout)
    np.testing.assert_array_equal(
        np.asarray(target_root["meta"]["episode_ends"][:]),
        episode_ends,
    )

    expected_return = compute_discounted_return(
        expected_reward,
        target_done.reshape(-1),
        0.99,
    )
    np.testing.assert_allclose(
        np.asarray(target_root["data"]["return"][:]).reshape(-1),
        expected_return,
        rtol=1e-6,
        atol=1e-6,
    )
    assert target_reward[episode_ends[0] - 1] == 0.0
    assert target_reward[episode_ends[1] - 1] == 0.0
    assert target_root.attrs["reward_mode"] == "last_gripper_release_sparse"

    source_cache = work / "source_cache" / "encoder_normalizer"
    latent = _create_source_cache(source_cache, source, episode_ends)
    cache_summary = clone_latent_cache_for_reward_clone(
        source_cache,
        source,
        target,
        link_mode="hardlink",
    )
    target_cache = pathlib.Path(cache_summary["target_cache"])
    target_latent = np.load(target_cache / "obs_latent.npy", mmap_mode="r")
    np.testing.assert_array_equal(target_latent, latent)

    with (target_cache / "metadata.json").open("r") as file:
        target_metadata = json.load(file)
    assert target_metadata["source_dataset"] == os.path.realpath(target)
    assert target_metadata["reward_only_dataset_clone"] is True
    assert target_metadata["derived_from_cache"] == str(source_cache.resolve())
    assert os.stat(source_cache / "obs_latent.npy").st_ino == os.stat(
        target_cache / "obs_latent.npy"
    ).st_ino

    print("SMOKE reward-v2 Zarr migration + latent-cache reuse PASSED")
    print(f"Work dir: {work}")


if __name__ == "__main__":
    main()
