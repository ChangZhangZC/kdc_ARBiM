"""Prepare dataset.

Modes (selected via YAML config):
  raw_to_npy   - raw teleop demo_*.npy -> processed .npy
  build_zarr   - teleop_sources (.npy) + rollout_sources (.h5 dirs) -> new zarr
  extend_zarr  - base zarr + rollout_sources -> new zarr (base zarr is read-only)
"""

import argparse
import gc
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import zarr
from termcolor import cprint
from tqdm import tqdm


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'configs',
    'data_prepare.yaml',
)

# Camera intrinsics for the realsense depth used in raw teleop frames.
RAW_CAMERA_INTRINSICS = (228.45703125, 228.734375, 151.828125, 125.9453125)

RETURN_GAMMA = 0.99
ZARR_CHUNK_LEAD = 100


def load_config(path):
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'PyYAML is required to read config files. Install it with: '
            'python -m pip install PyYAML'
        ) from exc

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault('overwrite', True)
    cfg.setdefault('num_points', 1024)
    cfg.setdefault('max_episode_len', 2000)
    cfg.setdefault('lambda_penalty', 0.05)
    cfg.setdefault('smooth_penalty', 0.01)
    cfg.setdefault('only_success', False)
    cfg.setdefault('min_episode_len', 30)
    cfg.setdefault('include_rollouts', False)
    cfg.setdefault('teleop_sources', [])
    cfg.setdefault('rollout_sources', [])
    cfg.setdefault('base_zarr_path', None)
    return cfg


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
        return_[i] = reward[i] + gamma * pre_return * not_done[i]
        pre_return = return_[i]
    return return_


def make_buffers():
    return {
        'point_cloud': [],
        'next_point_cloud': [],
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


def source_id(source):
    return source.get('name') or source['path']


def record_source(buffers, kind, source):
    entry = {
        'kind': kind,
        'name': source_id(source),
        'path': source.get('path'),
    }
    if source.get('type'):
        entry['type'] = source['type']
    if source.get('round'):
        entry['round'] = source['round']
    if entry not in buffers['source_manifest']:
        buffers['source_manifest'].append(entry)


def selected_rollout_sources(config, require_rollouts=False):
    rounds = set(config.get('_selected_rollout_rounds') or [])
    names = set(config.get('_selected_rollout_names') or [])
    include_rollouts = bool(
        config.get('include_rollouts') or rounds or names or config.get('mode') == 'extend_zarr'
    )
    if not include_rollouts:
        return []

    selected = []
    for source in config.get('rollout_sources', []):
        if not source.get('enabled', True):
            continue
        if rounds and source.get('round') not in rounds:
            continue
        if names and source_id(source) not in names:
            continue
        selected.append(source)

    if rounds:
        matched_rounds = {source.get('round') for source in selected}
        missing = rounds - matched_rounds
        if missing:
            raise ValueError(f'no rollout_sources matched rounds: {sorted(missing)}')
    if names:
        matched_names = {source_id(source) for source in selected}
        missing = names - matched_names
        if missing:
            raise ValueError(f'no rollout_sources matched names: {sorted(missing)}')
    if require_rollouts and not selected:
        raise ValueError('extend_zarr requires at least one selected rollout source')

    return selected


def existing_source_names(buffers):
    return {entry.get('name') for entry in buffers['source_manifest'] if entry.get('name')}


def ensure_sources_not_already_used(buffers, sources):
    existing = existing_source_names(buffers)
    duplicates = sorted(source_id(source) for source in sources if source_id(source) in existing)
    if duplicates:
        raise ValueError(
            'selected rollout source(s) already exist in base zarr source_manifest: '
            f'{duplicates}'
        )


def process_raw_teleop_to_npy(config):
    # Lazy import: realsense pulls in cv2/pyrealsense2/fpsample, only needed for raw mode.
    from realsense import X_root_camera, depth2pc, point_cloud_downsample

    raw_dir = config.get('raw_teleop_dir')
    output_path = config.get('processed_npy_output')
    if not raw_dir or not output_path:
        raise ValueError('raw_to_npy requires raw_teleop_dir and processed_npy_output')

    num_points = config['num_points']
    max_episode_len = config['max_episode_len']
    lp = config['lambda_penalty']
    sp = config['smooth_penalty']

    data = {
        'action': [],
        'reward': [],
        'is_success': [],
        'done': [],
        'agent_pos': [],
        'point_cloud': [],
        'timeout': [],
        'ee_pose': [],
    }

    folder = Path(raw_dir)
    file_names = sorted(
        f.name for f in folder.iterdir() if f.is_file() and f.name.startswith('demo_')
    )
    for file_name in file_names:
        demo_path = folder / file_name
        print('demo id:', file_name)
        demo = np.load(str(demo_path), allow_pickle=True)
        demo_length = len(demo)
        print(demo_length)
        for i in range(demo_length):
            if i % 100 == 0:
                print('Frame:', i)
            frame = demo[i]
            point_cloud = depth2pc(
                frame['depth'] * frame['depth_scale'],
                RAW_CAMERA_INTRINSICS,
                X_root_camera,
            )
            reward = float(i == demo_length - 1)
            if reward == 1.0:
                reward -= lp * demo_length / max_episode_len
            if i > 0:
                reward -= sp * np.linalg.norm(
                    frame['ee_euler_action'] - demo[i - 1]['ee_euler_action']
                )
            data['action'].append(frame['ee_euler_action'])
            data['reward'].append(reward)
            data['done'].append(i == demo_length - 1)
            data['timeout'].append(i == demo_length - 1)
            data['agent_pos'].append(frame['qpos'])
            data['point_cloud'].append(point_cloud_downsample(point_cloud, num_points))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(output_path, data)
    cprint(f'saved processed teleop npy to {output_path}', 'green')


def load_processed_npy(path):
    return np.load(path, allow_pickle=True).item()


def _push_episode_end(buffers, total_count_sub):
    buffers['_total_count'] += total_count_sub
    buffers['episode_ends'].append(buffers['_total_count'])


def append_processed_transitions(data, buffers, source_name, config):
    pc = data['point_cloud']
    state = data['agent_pos']
    action = data['action']
    rewards = np.asarray(data['reward'], dtype=np.float32)
    timeouts = list(data['timeout'])
    n = len(timeouts)
    if n == 0:
        cprint(f'[teleop:{source_name}] empty source', 'yellow')
        return

    appended_episodes = 0
    appended_transitions = 0
    total_count_sub = 0
    total_reward = 0.0
    for i in range(n):
        buffers['point_cloud'].append(pc[i])
        buffers['state'].append(state[i])
        buffers['action'].append(action[i])
        buffers['reward'].append(float(rewards[i]))
        # Teleop sources use timeout as both episode boundary and done flag.
        buffers['done'].append(bool(timeouts[i]))
        buffers['timeout'].append(bool(timeouts[i]))
        if i == n - 1:
            buffers['next_point_cloud'].append(pc[i])
            buffers['next_state'].append(state[i])
            buffers['next_action'].append(action[i])
        else:
            buffers['next_point_cloud'].append(pc[i + 1])
            buffers['next_state'].append(state[i + 1])
            buffers['next_action'].append(action[i + 1])
        total_count_sub += 1
        total_reward += float(rewards[i])
        if timeouts[i]:
            _push_episode_end(buffers, total_count_sub)
            appended_episodes += 1
            appended_transitions += total_count_sub
            print(
                f'[teleop:{source_name}] episode {appended_episodes}, '
                f'length: {total_count_sub}, return: {total_reward:.2f}'
            )
            total_count_sub = 0
            total_reward = 0.0
    cprint(
        f'[teleop:{source_name}] episodes: {appended_episodes}, '
        f'transitions: {appended_transitions}',
        'green',
    )


def append_rollout_source(source, buffers, config):
    name = source.get('name') or source['path']
    src_type = source.get('type', 'h5_rollout_dir')
    if src_type != 'h5_rollout_dir':
        raise ValueError(f'unknown rollout source type: {src_type}')
    traj_path = source['path']
    if not os.path.isdir(traj_path):
        cprint(f'[rollout:{name}] not a directory, skipping: {traj_path}', 'yellow')
        return

    min_len = config['min_episode_len']
    max_len = config['max_episode_len']
    lp = config['lambda_penalty']
    sp = config['smooth_penalty']
    only_success = config['only_success']

    total_trajectories = 0
    successful_trajectories = 0
    skipped_short = 0
    appended_episodes = 0
    appended_transitions = 0

    for subfolder in sorted(os.listdir(traj_path)):
        subfolder_path = os.path.join(traj_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for filename in sorted(os.listdir(subfolder_path)):
            if not filename.endswith('.h5'):
                continue
            file_path = os.path.join(subfolder_path, filename)
            with h5py.File(file_path, 'r') as f:
                data = {key: f[key][()] for key in f.keys()}

            demo_length = len(data['done'])
            if demo_length < min_len:
                skipped_short += 1
                continue

            is_success = bool(np.asarray(data['is_success']).any())
            total_trajectories += 1
            if is_success:
                successful_trajectories += 1
            if only_success and not is_success:
                continue
            if is_success:
                data['reward'][-1] = 1

            print(f'[rollout:{name}] loading {file_path}')
            total_reward = 0.0
            total_count_sub = 0
            ep_in_file = 0
            for total_id in range(demo_length):
                reward = float(data['reward'][total_id])
                if total_id == demo_length - 1 and reward == 1:
                    reward -= lp * demo_length / max_len
                if total_id > 0:
                    reward -= sp * np.linalg.norm(
                        np.asarray(data['action'][total_id]).squeeze()
                        - np.asarray(data['action'][total_id - 1]).squeeze()
                    )
                total_reward += reward
                buffers['point_cloud'].append(data['point_cloud'][total_id])
                buffers['state'].append(data['state'][total_id])
                buffers['action'].append(np.asarray(data['action'][total_id]).squeeze())
                buffers['next_action'].append(
                    np.asarray(data['next_action'][total_id]).squeeze()
                )
                buffers['reward'].append(reward)
                buffers['done'].append(bool(data['done'][total_id]))
                buffers['timeout'].append(bool(data['timeout'][total_id]))
                if total_id == demo_length - 1:
                    buffers['next_point_cloud'].append(data['point_cloud'][total_id])
                    buffers['next_state'].append(data['state'][total_id])
                else:
                    buffers['next_point_cloud'].append(data['point_cloud'][total_id + 1])
                    buffers['next_state'].append(data['state'][total_id + 1])
                total_count_sub += 1
                # Match original boundary: enumerate iterated over data['done'].
                if bool(data['done'][total_id]):
                    _push_episode_end(buffers, total_count_sub)
                    ep_in_file += 1
                    appended_episodes += 1
                    appended_transitions += total_count_sub
                    print(
                        f'[rollout:{name}] episode {appended_episodes}, '
                        f'length: {total_count_sub}, return: {total_reward:.2f}, '
                        f'final_reward: {reward:.2f}'
                    )
                    total_count_sub = 0
                    total_reward = 0.0

    if total_trajectories > 0:
        success_rate = successful_trajectories / total_trajectories * 100
        cprint(
            f'[rollout:{name}] trajectories: {total_trajectories}, '
            f'successful: {successful_trajectories} ({success_rate:.2f}%), '
            f'short_skipped: {skipped_short}, '
            f'appended_episodes: {appended_episodes}, '
            f'appended_transitions: {appended_transitions}',
            'green',
        )
    else:
        cprint(
            f'[rollout:{name}] no valid trajectories (short_skipped={skipped_short})',
            'yellow',
        )


def load_base_zarr(path, buffers):
    if not os.path.exists(path):
        raise FileNotFoundError(f'base_zarr_path does not exist: {path}')
    root = zarr.open(path, mode='r')
    data = root['data']
    meta = root['meta']
    n = data['point_cloud'].shape[0]
    if n == 0:
        cprint(f'[base_zarr:{path}] empty', 'yellow')
        return

    offset_before = buffers['_total_count']
    pc = data['point_cloud'][:]
    npc = data['next_point_cloud'][:]
    st = data['state'][:]
    nst = data['next_state'][:]
    act = data['action'][:]
    nact = data['next_action'][:]
    rew = np.asarray(data['reward'][:]).reshape(-1)
    dn = np.asarray(data['done'][:]).reshape(-1)
    to = np.asarray(data['timeout'][:]).reshape(-1)
    ep_ends = np.asarray(meta['episode_ends'][:])

    for i in range(n):
        buffers['point_cloud'].append(pc[i])
        buffers['next_point_cloud'].append(npc[i])
        buffers['state'].append(st[i])
        buffers['next_state'].append(nst[i])
        buffers['action'].append(act[i])
        buffers['next_action'].append(nact[i])
        buffers['reward'].append(float(rew[i]))
        buffers['done'].append(bool(dn[i]))
        buffers['timeout'].append(bool(to[i]))
    buffers['_total_count'] += n
    for end in ep_ends:
        buffers['episode_ends'].append(int(end) + offset_before)
    buffers['source_manifest'].extend(list(root.attrs.get('source_manifest', [])))

    cprint(
        f'[base_zarr:{path}] transitions: {n}, episodes: {len(ep_ends)}',
        'green',
    )


def write_zarr(buffers, output_path, overwrite):
    if not output_path:
        raise ValueError('zarr_output_path is required')
    if len(buffers['point_cloud']) == 0:
        raise RuntimeError('no transitions collected, refusing to write empty zarr')

    safe_prepare_output_dir(output_path, overwrite)
    os.makedirs(output_path, exist_ok=True)

    root = zarr.group(output_path)
    data = root.create_group('data')
    meta = root.create_group('meta')
    root.attrs['source_manifest'] = buffers['source_manifest']

    try:
        from numcodecs import Blosc
        compressor = Blosc(cname='zstd', clevel=3, shuffle=1)
    except Exception:
        compressor = None

    def create_array(group, name, array, chunks=None, dtype=None):
        """Create zarr arrays on both zarr v2 and v3 APIs."""
        if hasattr(group, 'create_dataset'):
            kwargs = {
                'data': array,
                'dtype': dtype,
                'overwrite': True,
            }
            if chunks is not None:
                kwargs['chunks'] = chunks
            if compressor is not None:
                kwargs['compressor'] = compressor
            return group.create_dataset(name, **kwargs)

        kwargs = {
            'data': array,
            'overwrite': True,
        }
        if chunks is not None:
            kwargs['chunks'] = chunks
        # zarr v3 uses a different codec API. Rely on its default compressors
        # rather than passing a v2 numcodecs compressor.
        return group.create_array(name, **kwargs)

    point_cloud = np.stack(buffers['point_cloud'], axis=0).astype(np.float32)
    pc_chunks = (ZARR_CHUNK_LEAD, point_cloud.shape[1], point_cloud.shape[2])
    create_array(data, 'point_cloud', point_cloud, chunks=pc_chunks, dtype='float32')
    cprint(
        f'point_cloud shape: {point_cloud.shape}, '
        f'range: [{np.min(point_cloud)}, {np.max(point_cloud)}]',
        'green',
    )
    del point_cloud
    gc.collect()

    next_point_cloud = np.stack(buffers['next_point_cloud'], axis=0).astype(np.float32)
    npc_chunks = (ZARR_CHUNK_LEAD, next_point_cloud.shape[1], next_point_cloud.shape[2])
    create_array(
        data, 'next_point_cloud', next_point_cloud, chunks=npc_chunks, dtype='float32'
    )
    cprint(
        f'next_point_cloud shape: {next_point_cloud.shape}, '
        f'range: [{np.min(next_point_cloud)}, {np.max(next_point_cloud)}]',
        'green',
    )
    del next_point_cloud
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

    create_array(
        data, 'state', state, chunks=(ZARR_CHUNK_LEAD, state.shape[1]),
        dtype='float32',
    )
    create_array(
        data, 'next_state', next_state, chunks=(ZARR_CHUNK_LEAD, next_state.shape[1]),
        dtype='float32',
    )
    create_array(
        data, 'action', action, chunks=(ZARR_CHUNK_LEAD, action.shape[1]),
        dtype='float32',
    )
    create_array(
        data, 'next_action', next_action,
        chunks=(ZARR_CHUNK_LEAD, next_action.shape[1]), dtype='float32',
    )
    create_array(
        data, 'reward', reward, chunks=(ZARR_CHUNK_LEAD, reward.shape[1]),
        dtype='float32',
    )
    create_array(
        data, 'return', return_, chunks=(ZARR_CHUNK_LEAD, return_.shape[1]),
        dtype='float32',
    )
    create_array(
        data, 'done', done, chunks=(ZARR_CHUNK_LEAD, done.shape[1]),
        dtype='bool',
    )
    create_array(
        data, 'timeout', timeout, chunks=(ZARR_CHUNK_LEAD, timeout.shape[1]),
        dtype='bool',
    )
    create_array(meta, 'episode_ends', episode_ends, dtype='int64')

    cprint(f'state shape: {state.shape}, range: [{np.min(state)}, {np.max(state)}]', 'green')
    cprint(f'action shape: {action.shape}, range: [{np.min(action)}, {np.max(action)}]', 'green')
    cprint(f'next_state shape: {next_state.shape}, range: [{np.min(next_state)}, {np.max(next_state)}]', 'green')
    cprint(f'next_action shape: {next_action.shape}, range: [{np.min(next_action)}, {np.max(next_action)}]', 'green')
    cprint(f'reward shape: {reward.shape}, range: [{np.min(reward)}, {np.max(reward)}]', 'green')
    cprint(f'done shape: {done.shape}, range: [{np.min(done)}, {np.max(done)}]', 'green')
    cprint(f'timeout shape: {timeout.shape}, range: [{np.min(timeout)}, {np.max(timeout)}]', 'green')
    cprint(f'return shape: {return_.shape}, range: [{np.min(return_)}, {np.max(return_)}]', 'green')
    cprint(f'episode_ends shape: {episode_ends.shape}, episodes: {len(episode_ends)}', 'green')
    cprint(f'Saved zarr file to {output_path}', 'green')


def run_build_zarr(config):
    buffers = make_buffers()
    for src in config['teleop_sources']:
        if 'path' not in src:
            raise ValueError(f'teleop source missing path: {src}')
        name = src.get('name') or src['path']
        data = load_processed_npy(src['path'])
        append_processed_transitions(data, buffers, name, config)
        record_source(buffers, 'teleop_npy', src)
        del data
        gc.collect()
    rollout_sources = selected_rollout_sources(config)
    if rollout_sources:
        cprint(f'using {len(rollout_sources)} rollout source(s)', 'cyan')
    else:
        cprint('not using rollout sources for this build', 'yellow')
    for src in rollout_sources:
        if 'path' not in src:
            raise ValueError(f'rollout source missing path: {src}')
        append_rollout_source(src, buffers, config)
        record_source(buffers, 'rollout_h5_dir', src)
    write_zarr(buffers, config['zarr_output_path'], config['overwrite'])


def run_extend_zarr(config):
    base = config.get('base_zarr_path')
    if not base:
        raise ValueError('extend_zarr requires base_zarr_path')
    output_path = config['zarr_output_path']
    if os.path.abspath(base) == os.path.abspath(output_path):
        raise ValueError('extend_zarr refuses to write back into base_zarr_path')
    buffers = make_buffers()
    load_base_zarr(base, buffers)
    rollout_sources = selected_rollout_sources(config, require_rollouts=True)
    ensure_sources_not_already_used(buffers, rollout_sources)
    cprint(f'extending with {len(rollout_sources)} rollout source(s)', 'cyan')
    for src in rollout_sources:
        if 'path' not in src:
            raise ValueError(f'rollout source missing path: {src}')
        append_rollout_source(src, buffers, config)
        record_source(buffers, 'rollout_h5_dir', src)
    write_zarr(buffers, output_path, config['overwrite'])


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'Prepare dataset. Modes: raw_to_npy, build_zarr, extend_zarr. '
            'All inputs, outputs, sources, and tuning values are read from the YAML config.'
        ),
    )
    p.add_argument(
        '--config',
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f'Path to YAML config (default: {DEFAULT_CONFIG_PATH})',
    )
    p.add_argument(
        '--mode',
        choices=('raw_to_npy', 'build_zarr', 'extend_zarr'),
        help='Override mode from config.',
    )
    rollout_group = p.add_mutually_exclusive_group()
    rollout_group.add_argument(
        '--include-rollouts',
        dest='include_rollouts',
        action='store_true',
        default=None,
        help='Include configured rollout_sources for this run.',
    )
    rollout_group.add_argument(
        '--no-rollouts',
        dest='include_rollouts',
        action='store_false',
        help='Ignore configured rollout_sources for this run.',
    )
    p.add_argument(
        '--rollout-round',
        action='append',
        default=[],
        help='Use only rollout_sources with this round. Can be passed multiple times.',
    )
    p.add_argument(
        '--rollout-source',
        action='append',
        default=[],
        help='Use only rollout_sources with this name. Can be passed multiple times.',
    )
    p.add_argument('--base-zarr-path', help='Override base_zarr_path from config.')
    p.add_argument('--zarr-output-path', help='Override zarr_output_path from config.')
    args = p.parse_args()
    if args.include_rollouts is False and (args.rollout_round or args.rollout_source):
        p.error('--no-rollouts cannot be combined with rollout filters')
    return args


def apply_cli_overrides(config, args):
    if args.mode:
        config['mode'] = args.mode
    if args.include_rollouts is not None:
        config['include_rollouts'] = args.include_rollouts
    if args.rollout_round:
        config['_selected_rollout_rounds'] = args.rollout_round
    if args.rollout_source:
        config['_selected_rollout_names'] = args.rollout_source
    if args.base_zarr_path:
        config['base_zarr_path'] = args.base_zarr_path
    if args.zarr_output_path:
        config['zarr_output_path'] = args.zarr_output_path
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    mode = config.get('mode')
    if mode == 'raw_to_npy':
        process_raw_teleop_to_npy(config)
    elif mode == 'build_zarr':
        run_build_zarr(config)
    elif mode == 'extend_zarr':
        run_extend_zarr(config)
    else:
        raise ValueError(
            f'unknown mode: {mode!r}. expected raw_to_npy, build_zarr, or extend_zarr.'
        )


if __name__ == '__main__':
    main()
