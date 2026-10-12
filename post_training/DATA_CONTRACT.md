# ARBiM data-format contract — LeRobot + Offline RL Zarr

## 1. Storage roles

- **LeRobot** is the long-term, reproducible trajectory dataset for ACT imitation learning and for reconstructing Offline RL datasets.
- **Zarr** is the current Offline RL/OfflineBuffer training format. Its schema is preserved, including JPEG byte storage, next-indexing and Reward/Return.
- **Staging NPY** (`arbim_rollout_staging_v2`) remains a temporary Data Wheel acquisition format with T actions and T+1 observations.
- **Processed NPY** is **deprecated** as a required intermediate. Existing `raw_to_npy`, `build_db`, and `rollout_to_npy` remain usable during migration and for regression tests.

No existing LeRobot, Zarr or Staging datasets are deleted or overwritten by the new workflows.

## 2. New and existing workflows

```text
Rosbag → CvtRosbag2Lerobot → Teleop LeRobot ──────┐
                                                  ├── merge_lerobot → ACT (IL)
Data Wheel → Staging NPY → staging_to_lerobot ─────┘
                              │
                              ▼
                     Rollout LeRobot
                              │
                     lerobot_to_zarr
                              │
                              ▼
                         Round Zarr ──┐
                                     ├── merge_zarr → Offline PPO
             Teleop LeRobot ──────┐  │
                                 ▼  │
                          lerobot_to_zarr
                                 │   │
                          Teleop Zarr ┘
```

Always distinguish **success** (used to collect IL demonstrations) from a **valid Reward V2** (required for RL). Successful Staging episodes with invalid Reward are retained for IL unless `--require-valid-reward` is set.

## 3. LeRobot → Zarr

The default `post_training/configs/data/data_prepare.yaml` mode is `lerobot_to_zarr`. Use a new Zarr output directory:

```yaml
mode: lerobot_to_zarr
lerobot_root: /absolute/path/to/teleop_or_rollout_lerobot
zarr_output_path: /absolute/path/to/new_reward_v2.zarr
overwrite: false
use_depth: false
stream_batch_size: 50
jpeg_quality: 95
reward_mode: last_gripper_release_sparse
reward_source: auto
```

```bash
python post_training/src/post_rl/data/data_prepare.py --config post_training/configs/data/data_prepare.yaml
```

Reward handling:
- `reward_source: auto`: respect Data Wheel recorded Reward V2 provenance when present; compute Reward V2 for other Teleop episodes, calibrated only on the Teleop subset.
- `reward_source: recorded`: require a valid recorded Reward V2 for **every** episode.
- `reward_source: computed`: intentionally ignore provenance and recompute.
- A successful but Reward-invalid Staging Episode is **not accepted into a Reward V2 Zarr** by default.
- A video or RGB/Depth source is streamed in batches. No Processed NPY is written.
- Zarr uses the existing terminal convention: the terminal transition repeats the last saved LeRobot observation/action because LeRobot stores T action-aligned observations, not the final extra observation.
- `gamma=0.99` is used for returns, as in the previous `run_build_db`.

## 4. Data Wheel Staging → LeRobot

Use the original Teleop LeRobot directory as a **schema template** to preserve ACT feature names, camera dimensions, FPS, and robot type.

```bash
python post_training/src/post_rl/data/staging_to_lerobot.py \
  outputs/eval/<task>/<method>/<run>/epochbest_flywheel/data_wheel/staging \
  --template-lerobot-root /absolute/path/to/original_teleop_lerobot \
  --output-root /absolute/path/to/new_rollout_lerobot \
  --repo-id lerobot/arbim_rollout_round_01 \
  --task "ARBiM Data Wheel" \
  --require-valid-reward
```

Omit `--require-valid-reward` to also export Reward-invalid successful episodes for **IL**. Add `--use-depth` only when the template and every selected source include all three Depth cameras. Metadata is saved under `meta/arbim_staging_provenance.json`; the final extra Staging observation is retained only in the original Staging NPY.

**FPS limitation:** Staging V2 does not store per-frame timestamps/FPS. The exporter uses the template's FPS and assumes the Data Wheel sampled at that rate. Verify the data acquisition `ros_rate` matches the template's training FPS before treating the resulting LeRobot timestamps as physical time.

## 5. Multi-Zarr merge

```yaml
mode: merge_zarr
overwrite: false
zarr_output_path: /absolute/path/to/new_merged.zarr
return_gamma: 0.99
zarr_sources:
  - name: teleop_reward_v2
    path: /absolute/path/to/teleop_reward_v2.zarr
  - name: flywheel_round_01
    path: /absolute/path/to/rollout_reward_v2.zarr
```

```bash
python post_training/src/post_rl/data/data_prepare.py --config /path/to/merge_zarr.yaml
```

The merger copies JPEG bytes without RGB decode/re-encode; offsets, `next_index` and `meta/episode_ends` are relocated per source. Rewards are preserved and Returns recomputed with the configured discount. It rejects incompatible RGB/Depth shapes, Zarr arrays, different Reward modes, invalid V2 Reward, and overwrites. Inputs are read-only.

## 6. Multi-LeRobot merge (for ACT)

```bash
python post_training/src/post_rl/data/lerobot_merge.py \
  /absolute/path/to/teleop_lerobot \
  /absolute/path/to/rollout_lerobot \
  --output-root /absolute/path/to/new_combined_lerobot \
  --repo-id lerobot/arbim_combined_round_01 \
  --fallback-task "ARBiM"
```

Add `--use-depth` when both sources share three Depth features. The converter verifies FPS, 16D state/action, camera keys, dtypes, names, image/video mode, and robot type. It writes a new dataset via the installed LeRobot writer; this **re-decodes and re-encodes videos** and may take substantial time. Merge provenance is written to `meta/arbim_merge_provenance.json`. If merged LeRobot later becomes an RL source, the direct Zarr conversion reads and preserves recorded Data Wheel Reward V2 while recalculating Teleop rewards.

Only common ACT training features (16D state/action, 3 RGB, optional 3 Depth) are currently exported; extra ROS-specific features are intentionally omitted.

## 7. Consolidated tests (after all feature batches)

```bash
git fetch origin
git switch feature/flywheel-data-contract
git pull --ff-only origin feature/flywheel-data-contract
bash post_training/tests/run_data_contract.sh
```

Validate the **original** full Teleop LeRobot dataset versus the migrated Reward V2 Zarr:

```bash
python post_training/tests/integration/check_lerobot_to_zarr_real.py \
  --source-npy data/sim_task1.npy \
  --output data/sim_task1_direct_contract_test.zarr \
  --reference-zarr data/sim_task1.zarr \
  --expected-episodes 600 --expected-frames 90144
```

Then exercise all Data Wheel and ACT/PPO conversion paths:

```bash
python post_training/tests/integration/check_data_contract_real.py \
  --teleop-lerobot /absolute/path/to/original_teleop_lerobot \
  --teleop-zarr data/sim_task1.zarr \
  --staging outputs/eval/sim_toy_pick/act_postrl_single_step/run_20261008_054349/epochbest_flywheel/data_wheel/staging \
  --output-dir outputs/data_contract_real_test \
  --expected-teleop-episodes 600 --expected-teleop-frames 90144
```

Optionally add `--verify-combined-lerobot-to-zarr` to also re-encode merged LeRobot and compare its Reward/episode boundaries with the directly merged Zarr. **Use fresh output directories** and check available disk space first. These full data-contract tests must pass before PR to `feature/flywheel`.
