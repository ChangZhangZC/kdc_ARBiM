import argparse
import gc
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
import zarr
from termcolor import cprint
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[4]
LEROBOT_SRC = REPO_ROOT / "third_party" / "lerobot" / "src"

if not LEROBOT_SRC.is_dir():
    raise RuntimeError(
        "LeRobot submodule is not initialized. "
        "Run `git submodule update --init --recursive`."
    )

for path in (REPO_ROOT, LEROBOT_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import lerobot_patches.custom_patches
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_CONFIG_PATH = str(
    REPO_ROOT / "post_training" / "configs" / "data" / "data_prepare.yaml"
)

RGB_FEATURE_TO_BUFFER = {
    'observation.images.head_cam_h': 'head_rgb',
    'observation.images.wrist_cam_l': 'wrist_left_rgb',
    'observation.images.wrist_cam_r': 'wrist_right_rgb',
}

DEPTH_FEATURE_TO_BUFFER = {
    'observation.depth_h': 'head_depth',
    'observation.depth_l': 'wrist_left_depth',
    'observation.depth_r': 'wrist_right_depth',
}

RETURN_GAMMA = 0.99
ZARR_CHUNK_LEAD = 50


def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault('lambda_penalty', 0.05)
    cfg.setdefault('smooth_penalty', 0.01)
    cfg.setdefault('max_episode_len', 2000)
    cfg.setdefault("use_depth", False)
    cfg.setdefault('overwrite', True)
    cfg.setdefault('teleop_sources', [])

    return cfg


def load_lerobot_dataset(root):
    root = Path(root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"LeRobot dataset root does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"LeRobot dataset root is not a directory: {root}")

    if not (root / "meta").exists():
        raise FileNotFoundError(f"LeRobot dataset metadata not found: {root / 'meta'}")

    repo_id = root.name

    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
    )


def extract_rgb(frame):
    return {
        key: frame[key]
        for key in RGB_FEATURE_TO_BUFFER
        if key in frame
    }


def extract_depth(frame):
    return {
        key: frame[key]
        for key in DEPTH_FEATURE_TO_BUFFER
        if key in frame
    }


def _validate_processed_data(data, use_depth, context):
    """Validate the frame-aligned processed LeRobot data contract."""
    required_keys = (
        'agent_pos',
        'action',
        'rgb',
        'reward',
        'done',
        'timeout',
    )
    if use_depth:
        required_keys = required_keys + ('depth',)

    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f'{context} is missing required key(s): {missing}')

    lengths = {key: len(data[key]) for key in required_keys}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f'{context} fields must have equal lengths, got {lengths}'
        )

    done = list(data['done'])
    timeout = list(data['timeout'])
    for index, (done_value, timeout_value) in enumerate(zip(done, timeout)):
        if bool(done_value) != bool(timeout_value):
            raise ValueError(
                f'{context} frame {index} has done != timeout '
                f'({done_value!r} != {timeout_value!r})'
            )

    if timeout and not bool(timeout[-1]):
        raise ValueError(
            f'{context} must end with timeout=True so the final episode is closed'
        )

    for index, rgb in enumerate(data['rgb']):
        if not isinstance(rgb, dict):
            raise TypeError(f'{context} frame {index} rgb must be a dict')
        missing_rgb = [
            key for key in RGB_FEATURE_TO_BUFFER if key not in rgb
        ]
        if missing_rgb:
            raise KeyError(
                f'{context} frame {index} is missing RGB feature(s): {missing_rgb}'
            )

    if use_depth:
        for index, depth in enumerate(data['depth']):
            if not isinstance(depth, dict):
                raise TypeError(f'{context} frame {index} depth must be a dict')
            missing_depth = [
                key for key in DEPTH_FEATURE_TO_BUFFER if key not in depth
            ]
            if missing_depth:
                raise KeyError(
                    f'{context} frame {index} is missing depth feature(s): '
                    f'{missing_depth}'
                )


def process_raw_teleop_to_npy(config):
    lerobot_root = config.get("lerobot_root")
    output_path = config.get("processed_npy_output")

    if not lerobot_root or not output_path:
        raise ValueError(
            "raw_to_npy requires lerobot_root and processed_npy_output"
        )

    dataset = load_lerobot_dataset(lerobot_root)

    use_depth = config["use_depth"]
    max_episode_len = config["max_episode_len"]
    lambda_penalty = config["lambda_penalty"]
    smooth_penalty = config["smooth_penalty"]
    if max_episode_len <= 0:
        raise ValueError("max_episode_len must be positive")

    data = {
        "action": [],
        "agent_pos": [],
        "rgb": [],
        "reward": [],
        "done": [],
        "timeout": [],
    }
    if use_depth:
        data["depth"] = []

    def as_numpy(value, feature_name, frame_index):
        if hasattr(value, "numpy"):
            try:
                value = value.numpy()
            except (RuntimeError, TypeError) as exc:
                raise ValueError(
                    f"frame {frame_index} has a non-CPU {feature_name!r} value; "
                    "data preparation expects NumPy arrays or CPU tensors"
                ) from exc
        array = np.asarray(value)
        if array.size == 0 or array.dtype == object or not np.issubdtype(array.dtype, np.number):
            raise ValueError(
                f"frame {frame_index} has an invalid {feature_name!r} value "
                f"(shape={array.shape}, dtype={array.dtype})"
            )
        return array

    episodes = {}
    for frame_index in tqdm(range(len(dataset)), desc="Grouping frames by episode"):
        frame = dataset[frame_index]
        if "episode_index" not in frame:
            raise KeyError(f"frame {frame_index} is missing required key 'episode_index'")
        episode_id = frame["episode_index"]
        if hasattr(episode_id, "item"):
            episode_id = episode_id.item()
        episodes.setdefault(episode_id, []).append((frame_index, frame))

    if not episodes:
        raise RuntimeError("LeRobot dataset contains no frames")

    print(f"Processing {len(episodes)} episodes")

    for episode_frames in tqdm(episodes.values(), total=len(episodes), desc="Processing episodes"):
        episode_length = len(episode_frames)
        previous_action = None

        for t, (frame_index, frame) in enumerate(episode_frames):
            if "observation.state" not in frame or "action" not in frame:
                missing = [
                    key for key in ("observation.state", "action") if key not in frame
                ]
                raise KeyError(f"frame {frame_index} is missing required key(s): {missing}")

            state = as_numpy(frame["observation.state"], "observation.state", frame_index)
            action = as_numpy(frame["action"], "action", frame_index)
            rgb = {
                key: as_numpy(value, key, frame_index)
                for key, value in extract_rgb(frame).items()
            }
            missing_rgb = [
                key for key in RGB_FEATURE_TO_BUFFER if key not in rgb
            ]
            if missing_rgb:
                raise KeyError(
                    f"frame {frame_index} is missing RGB feature(s): {missing_rgb}"
                )
            depth = None
            if use_depth:
                depth = {
                    key: as_numpy(value, key, frame_index)
                    for key, value in extract_depth(frame).items()
                }
                missing_depth = [
                    key for key in DEPTH_FEATURE_TO_BUFFER if key not in depth
                ]
                if missing_depth:
                    raise KeyError(
                        f"frame {frame_index} is missing depth feature(s): "
                        f"{missing_depth}"
                    )
            reward = float(t == episode_length - 1)

            if reward == 1.0:
                reward -= lambda_penalty * episode_length / max_episode_len

            if previous_action is not None:
                reward -= smooth_penalty * np.linalg.norm(action - previous_action)

            previous_action = action
            done = t == episode_length - 1
            timeout = done

            data["agent_pos"].append(state)
            data["action"].append(action)
            data["rgb"].append(rgb)
            if use_depth:
                data["depth"].append(depth)
            data["reward"].append(reward)
            data["done"].append(done)
            data["timeout"].append(timeout)

    _validate_processed_data(data, use_depth, 'processed LeRobot output')

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(output_path, data)
    cprint(f"Saved processed LeRobot npy to {output_path}", "green")


def make_buffers(use_depth=False):
    """Create the in-memory single-step RL transition buffers."""
    buffers = {
        'state': [],
        'next_state': [],
        'action': [],
        'next_action': [],
        'reward': [],
        'done': [],
        'timeout': [],
        'episode_ends': [],
        'source_manifest': [],
        '_total_count': 0,
    }

    for buffer_name in RGB_FEATURE_TO_BUFFER.values():
        buffers[buffer_name] = []
        buffers[f'next_{buffer_name}'] = []

    if use_depth:
        for buffer_name in DEPTH_FEATURE_TO_BUFFER.values():
            buffers[buffer_name] = []
            buffers[f'next_{buffer_name}'] = []

    return buffers


def load_processed_npy(path):
    return np.load(path, allow_pickle=True).item()


def _push_episode_end(buffers, total_count_sub):
    buffers['_total_count'] += total_count_sub
    buffers['episode_ends'].append(buffers['_total_count'])


def append_processed_transitions(data, buffers, source_name, config):
    """Append processed LeRobot frames as single-step RL transitions."""
    config = config or {}
    use_depth = bool(config.get('use_depth', False))
    _validate_processed_data(data, use_depth, f'[teleop:{source_name}]')

    expected_buffer_names = [
        'state',
        'next_state',
        'action',
        'next_action',
        'reward',
        'done',
        'timeout',
    ]
    expected_buffer_names.extend(RGB_FEATURE_TO_BUFFER.values())
    expected_buffer_names.extend(
        f'next_{name}' for name in RGB_FEATURE_TO_BUFFER.values()
    )
    if use_depth:
        expected_buffer_names.extend(DEPTH_FEATURE_TO_BUFFER.values())
        expected_buffer_names.extend(
            f'next_{name}' for name in DEPTH_FEATURE_TO_BUFFER.values()
        )
    missing_buffers = [
        name for name in expected_buffer_names if name not in buffers
    ]
    if missing_buffers:
        raise KeyError(
            f'[teleop:{source_name}] buffers missing required field(s): '
            f'{missing_buffers}; create them with make_buffers(use_depth={use_depth})'
        )

    state = data['agent_pos']
    action = data['action']
    rgb_frames = data['rgb']
    depth_frames = data.get('depth')
    rewards = np.asarray(data['reward'], dtype=np.float32)
    dones = list(data['done'])
    timeouts = list(data['timeout'])
    n = len(timeouts)
    if n == 0:
        print(f'[teleop:{source_name}] empty source')
        return

    appended_episodes = 0
    appended_transitions = 0
    total_count_sub = 0
    total_reward = 0.0
    for i in range(n):
        is_episode_end = bool(timeouts[i]) or i == n - 1
        next_index = i if is_episode_end else i + 1

        buffers['state'].append(state[i])
        buffers['action'].append(action[i])
        buffers['reward'].append(float(rewards[i]))
        buffers['done'].append(bool(dones[i]))
        buffers['timeout'].append(bool(timeouts[i]))
        buffers['next_state'].append(state[next_index])
        buffers['next_action'].append(action[next_index])

        for feature_name, buffer_name in RGB_FEATURE_TO_BUFFER.items():
            buffers[buffer_name].append(rgb_frames[i][feature_name])
            buffers[f'next_{buffer_name}'].append(
                rgb_frames[next_index][feature_name]
            )

        if use_depth:
            for feature_name, buffer_name in DEPTH_FEATURE_TO_BUFFER.items():
                buffers[buffer_name].append(depth_frames[i][feature_name])
                buffers[f'next_{buffer_name}'].append(
                    depth_frames[next_index][feature_name]
                )

        total_count_sub += 1
        total_reward += float(rewards[i])
        if bool(timeouts[i]):
            _push_episode_end(buffers, total_count_sub)
            appended_episodes += 1
            appended_transitions += total_count_sub
            print(
                f'[teleop:{source_name}] episode {appended_episodes}, '
                f'length: {total_count_sub}, return: {total_reward:.2f}'
            )
            total_count_sub = 0
            total_reward = 0.0

    if total_count_sub:
        raise ValueError(
            f'[teleop:{source_name}] ended with an unterminated episode'
        )

    print(
        f'[teleop:{source_name}] episodes: {appended_episodes}, '
        f'transitions: {appended_transitions}'
    )


def safe_prepare_output_dir(path, overwrite):
    if os.path.exists(path):
        if not overwrite:
            cprint(f'output {path} already exists and overwrite=false', 'red')
            raise SystemExit(1)

        cprint(f'overwriting {path}', 'yellow')
        shutil.rmtree(path)

    parent = os.path.dirname(path)

    if parent:
        os.makedirs(parent, exist_ok=True)


def compute_return(reward, not_done, gamma=RETURN_GAMMA):
    size_ = len(reward)
    return_ = np.zeros((size_, 1), dtype=np.float32)
    pre_return = 0.0

    for i in tqdm(reversed(range(size_)), total=size_, desc='Computing returns'):
        return_[i] = (
            reward[i]
            + gamma
            * pre_return
            * not_done[i]
        )

        pre_return = return_[i]

    return return_


def write_zarr(buffers, output_path, overwrite):
    """Write multimodal RL transition buffers to a Zarr dataset."""
    if not output_path:
        raise ValueError('zarr_output_path is required')

    if len(buffers['state']) == 0:
        raise RuntimeError('no transitions collected, refusing to write empty zarr')

    depth_buffer_names = list(DEPTH_FEATURE_TO_BUFFER.values())
    depth_presence = [name in buffers for name in depth_buffer_names]

    if any(depth_presence) and not all(depth_presence):
        missing = [name for name in depth_buffer_names if name not in buffers]
        raise KeyError(f'incomplete depth buffers, missing: {missing}')

    use_depth = all(depth_presence)

    safe_prepare_output_dir(output_path, overwrite)
    os.makedirs(output_path, exist_ok=True)

    root = zarr.group(output_path)
    data = root.create_group('data')
    meta = root.create_group('meta')
    root.attrs['source_manifest'] = buffers.get('source_manifest', [])

    try:
        from numcodecs import Blosc
        compressor = Blosc(cname='zstd', clevel=3, shuffle=1)
    except Exception:
        compressor = None

    def create_array(group, name, array, chunks=None, dtype=None):
        """Create Zarr arrays while keeping compatibility with original RL-100 v2/v3 handling."""
        if hasattr(group, 'create_dataset'):
            kwargs = {'data': array, 'overwrite': True}
            if dtype is not None:
                kwargs['dtype'] = dtype
            if chunks is not None:
                kwargs['chunks'] = chunks
            if compressor is not None:
                kwargs['compressor'] = compressor
            return group.create_dataset(name, **kwargs)

        kwargs = {'data': array, 'overwrite': True}
        if chunks is not None:
            kwargs['chunks'] = chunks
        return group.create_array(name, **kwargs)

    for buffer_name in RGB_FEATURE_TO_BUFFER.values():
        next_buffer_name = f'next_{buffer_name}'
        if buffer_name not in buffers:
            raise KeyError(f'missing RGB buffer: {buffer_name}')
        if next_buffer_name not in buffers:
            raise KeyError(f'missing RGB buffer: {next_buffer_name}')

        rgb = np.stack(buffers[buffer_name], axis=0)
        next_rgb = np.stack(buffers[next_buffer_name], axis=0)

        rgb_chunks = (ZARR_CHUNK_LEAD, *rgb.shape[1:])
        next_rgb_chunks = (ZARR_CHUNK_LEAD, *next_rgb.shape[1:])

        create_array(data, buffer_name, rgb, chunks=rgb_chunks, dtype=rgb.dtype)
        create_array(data, next_buffer_name, next_rgb, chunks=next_rgb_chunks, dtype=next_rgb.dtype)

        cprint(
            f'{buffer_name} shape: {rgb.shape}, dtype: {rgb.dtype}, range: [{np.min(rgb)}, {np.max(rgb)}]',
            'green',
        )
        cprint(
            f'{next_buffer_name} shape: {next_rgb.shape}, dtype: {next_rgb.dtype}, range: [{np.min(next_rgb)}, {np.max(next_rgb)}]',
            'green',
        )

        del rgb
        del next_rgb
        gc.collect()

    if use_depth:
        for buffer_name in DEPTH_FEATURE_TO_BUFFER.values():
            next_buffer_name = f'next_{buffer_name}'
            if next_buffer_name not in buffers:
                raise KeyError(f'missing depth buffer: {next_buffer_name}')

            depth = np.stack(buffers[buffer_name], axis=0)
            next_depth = np.stack(buffers[next_buffer_name], axis=0)

            depth_chunks = (ZARR_CHUNK_LEAD, *depth.shape[1:])
            next_depth_chunks = (ZARR_CHUNK_LEAD, *next_depth.shape[1:])

            create_array(data, buffer_name, depth, chunks=depth_chunks, dtype=depth.dtype)
            create_array(data, next_buffer_name, next_depth, chunks=next_depth_chunks, dtype=next_depth.dtype)

            cprint(
                f'{buffer_name} shape: {depth.shape}, dtype: {depth.dtype}, range: [{np.min(depth)}, {np.max(depth)}]',
                'green',
            )
            cprint(
                f'{next_buffer_name} shape: {next_depth.shape}, dtype: {next_depth.dtype}, range: [{np.min(next_depth)}, {np.max(next_depth)}]',
                'green',
            )

            del depth
            del next_depth
            gc.collect()

    state = np.stack(buffers['state'], axis=0).astype(np.float32)
    next_state = np.stack(buffers['next_state'], axis=0).astype(np.float32)
    action = np.stack(buffers['action'], axis=0).astype(np.float32)
    next_action = np.stack(buffers['next_action'], axis=0).astype(np.float32)
    reward = np.asarray(buffers['reward'], dtype=np.float32).reshape(-1, 1)
    done = np.asarray(buffers['done'], dtype=bool).reshape(-1, 1)
    timeout = np.asarray(buffers['timeout'], dtype=bool).reshape(-1, 1)
    episode_ends = np.asarray(buffers['episode_ends'], dtype=np.int64)
    not_done = 1.0 - (done | timeout).astype(np.float32)
    return_ = compute_return(reward, not_done, gamma=RETURN_GAMMA).astype(np.float32)

    create_array(data, 'state', state, chunks=(ZARR_CHUNK_LEAD, *state.shape[1:]), dtype='float32')
    create_array(data, 'next_state', next_state, chunks=(ZARR_CHUNK_LEAD, *next_state.shape[1:]), dtype='float32')
    create_array(data, 'action', action, chunks=(ZARR_CHUNK_LEAD, *action.shape[1:]), dtype='float32')
    create_array(data, 'next_action', next_action, chunks=(ZARR_CHUNK_LEAD, *next_action.shape[1:]), dtype='float32')
    create_array(data, 'reward', reward, chunks=(ZARR_CHUNK_LEAD, reward.shape[1]), dtype='float32')
    create_array(data, 'return', return_, chunks=(ZARR_CHUNK_LEAD, return_.shape[1]), dtype='float32')
    create_array(data, 'done', done, chunks=(ZARR_CHUNK_LEAD, done.shape[1]), dtype='bool')
    create_array(data, 'timeout', timeout, chunks=(ZARR_CHUNK_LEAD, timeout.shape[1]), dtype='bool')
    create_array(meta, 'episode_ends', episode_ends, dtype='int64')

    cprint(f'state shape: {state.shape}, range: [{np.min(state)}, {np.max(state)}]', 'green')
    cprint(f'next_state shape: {next_state.shape}, range: [{np.min(next_state)}, {np.max(next_state)}]', 'green')
    cprint(f'action shape: {action.shape}, range: [{np.min(action)}, {np.max(action)}]', 'green')
    cprint(f'next_action shape: {next_action.shape}, range: [{np.min(next_action)}, {np.max(next_action)}]', 'green')
    cprint(f'reward shape: {reward.shape}, range: [{np.min(reward)}, {np.max(reward)}]', 'green')
    cprint(f'return shape: {return_.shape}, range: [{np.min(return_)}, {np.max(return_)}]', 'green')
    cprint(f'done shape: {done.shape}, range: [{np.min(done)}, {np.max(done)}]', 'green')
    cprint(f'timeout shape: {timeout.shape}, range: [{np.min(timeout)}, {np.max(timeout)}]', 'green')
    cprint(f'episode_ends shape: {episode_ends.shape}, episodes: {len(episode_ends)}', 'green')
    cprint(f'Saved zarr file to {output_path}', 'green')


def source_id(source):
    return source.get('name') or source['path']


def record_source(buffers, kind, source):
    entry = {'kind': kind, 'name': source_id(source), 'path': source.get('path')}

    if entry not in buffers['source_manifest']:
        buffers['source_manifest'].append(entry)


def run_build_zarr(config):
    """
    V1 scope:
        - supports processed teleoperation sources only
        - does NOT support policy rollout sources
    """
    output_path = config.get('zarr_output_path')
    if not output_path:
        raise ValueError('build_zarr requires zarr_output_path')

    teleop_sources = config.get('teleop_sources', [])
    if not teleop_sources:
        raise ValueError('build_zarr requires at least one teleop_source')

    use_depth = bool(config.get('use_depth', False))
    overwrite = bool(config.get('overwrite', True))
    buffers = make_buffers(use_depth=use_depth)

    cprint(f'building Zarr from {len(teleop_sources)} teleop source(s)', 'cyan')
    for src in teleop_sources:
        if 'path' not in src:
            raise ValueError(f'teleop source missing path: {src}')
        source_path = src['path']
        if not os.path.exists(source_path):
            raise FileNotFoundError(f'teleop source does not exist: {source_path}')

        name = source_id(src)
        cprint(f'[teleop:{name}] loading {source_path}', 'cyan')

        processed_data = load_processed_npy(source_path)
        append_processed_transitions(processed_data, buffers, name, config)
        record_source(buffers, 'teleop_npy', src)

        del processed_data
        gc.collect()

    write_zarr(buffers, output_path, overwrite)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    mode = config.get('mode')

    if mode == 'raw_to_npy':
        process_raw_teleop_to_npy(config)
    elif mode == 'build_zarr':
        run_build_zarr(config)
    else:
        raise ValueError(
            f"unknown mode: {mode!r}. expected 'raw_to_npy' or 'build_zarr'."
        )


if __name__ == "__main__":
    main()
