# ARBiM Data Contract V1 — LeRobot / Zarr / Data Wheel

This is the integration contract for `feature/flywheel-data-contract`.
It is **not considered validated** until the final smoke and real-data
integration tests pass on the ARBiM environment.

## Canonical formats

| Format | Owner | Purpose |
| --- | --- | --- |
| LeRobot | long-lived data source | ACT IL training; reconstructed Teleop and successful Data Wheel episodes |
| ARBiM Offline RL Zarr | derived training cache | PPO OfflineBuffer / OfflineDataset; reward/return/transition indices |
| `arbim_rollout_staging_v2` NPY | Data Wheel staging | temporary T-action / T+1-observation snapshots, reward status and event |
| `arbim_processed_npy_stream_v2` | **deprecated** | historical bridge, not required in the new data flow |

This migration changes the *data preparation contract*, **not** ACT/PPO
trainer interfaces or the existing Zarr schema (`jpeg_bytes_v1`).

## Pipeline

```text
Teleop Rosbag  -> CvtRosbag2Lerobot.py -> Teleop LeRobot -------------+
                                                                       |
Data Wheel eval -> Staging NPY -> staging_to_lerobot.py -> Rollout LeRobot
                                   |                          |       |
                    successful Reward-invalid data           |       |
                    still usable for IL by default           |       |
                                                              |       |
  Teleop LeRobot + Rollout LeRobot ---> lerobot_merge.py ---> ACT IL
          |                         (re-encodes LeRobot video)
          +--> lerobot_to_zarr -----------------------------+
                                                              |
  Rollout LeRobot --> lerobot_to_zarr (recorded Reward V2) ----+
                                                              |
                      merge_zarr (no JPEG re-encode) ----------+
                                  |
                                  +--> merged PPO Offline RL Zarr
```

### Batch 01: LeRobot -> Zarr

```bash
python post_training/src/post_rl/data/data_prepare.py \
  --config post_training/configs/data/lerobot_to_zarr.yaml
```

Set `lerobot_root`, a **new** `zarr_output_path`, and `use_depth`.
`reward_mode: last_gripper_release_sparse` is Reward V2.
`reward_source: auto` (default) checks for
`meta/arbim_staging_provenance.json` and uses **recorded** reward indices
for Data Wheel LeRobot. A successful rollout with invalid Reward is still a
valid IL demonstration, but its RL export raises rather than fabricates +1.
For Teleop LeRobot without provenance, Reward V2 is computed from Actions.
`reward_source: recorded` requires provenance;
`reward_source: computed` forces detection.

Zarr contains `data/{state,next_state,action,next_action,reward,return,done,timeout,next_index}`,
three JPEG byte streams and offsets, optional dense Depth arrays, and
`meta/episode_ends`. The `data/return` discount is 0.99.

### Batch 02: Multiple Zarr -> One Zarr

```bash
python post_training/src/post_rl/data/data_prepare.py \
  --config post_training/configs/data/merge_zarr.yaml
```

All inputs must have the same RGB/Depth shape/dtype, Reward semantics and
Zarr storage contract. Reject unknown/old/mixed Reward metadata; V2 sources
must contain exactly one **nonterminal** +1 per episode. The merge:
- copies dense arrays and JPEG payloads in bounded blocks;
- recomputes JPEG byte offsets, episode ends and local `next_index`;
- **preserves** Reward and recomputes Return with `return_gamma: 0.99`;
- saves `source_manifest` and `merge_inputs` provenance;
- writes a new directory atomically, refusing overwrite.

Batch 02 does not open, decode or re-encode RGB images.

### Batch 03: Data Wheel Staging -> LeRobot

Use an **existing Teleop LeRobot dataset as template** so FPS,
`observation.state` / `action` names and camera schema remain consistent.
The installed PyAV video encoder expects an integer FPS (such as `10`, not
`10.0`); conversions normalize integral metadata to Python `int` and reject
non-integral FPS rather than silently rounding timestamps.
If `data/sim_task1.npy` already exists, its manifest `source_root` identifies
the original LeRobot root.

```bash
python post_training/src/post_rl/data/staging_to_lerobot.py \
  /path/to/epochbest_flywheel/data_wheel/staging \
  --template-lerobot-root /path/to/teleop_lerobot \
  --output-root /path/to/new/flywheel_lerobot \
  --repo-id lerobot/flywheel_round_01 \
  --task "ARBiM Data Wheel"
```

Default **IL export**: every successful staging episode, even
`reward_status=invalid`. Use `--require-valid-reward` to export a
**Reward-V2-valid RL subset**, and `--use-depth` only when every selected
Staging episode and the LeRobot template have all three Depth cameras.

Staging has `T` actions and `T+1` observations. LeRobot stores
`(obs_t, action_t)` for `t=0..T-1`: do **not** insert a fake final Action.
The final observation remains in the original Staging file. Keep Staging
files as archive for that information and image evidence.

For provenance, write `meta/arbim_staging_provenance.json` with episode
order, origin, Reward V2 status/event/index. The subsequent direct RL converter
reads and validates this; it will not silently replace verified labels.

### Batch 04: Multiple LeRobot -> One LeRobot

```bash
python post_training/src/post_rl/data/lerobot_merge.py \
  /path/to/teleop_lerobot /path/to/flywheel_lerobot \
  --output-root /path/to/new/combined_lerobot \
  --repo-id lerobot/arbim_combined \
  --fallback-task "ARBiM"
```

Optional: `--use-depth` and `--batch-size 25`. Inputs must match features
(including state/action names), FPS, robot type, RGB (and selected Depth).
The writer streams complete Episodes and records
`meta/arbim_merge_provenance.json`. LeRobot videos are decoded and
re-encoded; unlike Zarr Merge this is **not byte-for-byte video copying**.
A combined IL LeRobot is not the default RL input: convert each clean source
to its own Zarr and merge Zarr to preserve recorded Reward semantics.

## Safe migration and testing

Prior NPY utilities are **still available as deprecated compatibility code**
until the consolidated comparison test passes. No part of the **new** modes
requires a Processed NPY. Do not delete legacy artifacts solely because this
branch exists.

Run synthetic tests from the project root in the `arbim` environment:

```bash
bash post_training/tests/run_data_contract.sh
```

The complete real-data integration runner (new **work root** required):
```bash
python post_training/tests/integration/check_data_contract_real.py \
  --source-npy data/sim_task1.npy \
  --staging-dir /path/to/epochbest_flywheel/data_wheel/staging \
  --reference-zarr data/sim_task1.zarr \
  --work-root outputs/data_contract_e2e_01 \
  --expected-teleop-episodes 600 \
  --expected-teleop-frames 90144
```

Requires substantial temporary disk space for a Teleop Zarr, Rollout Zarr,
merged Zarr and re-encoded combined LeRobot. Work on a new directory; it
does **not overwrite** Teleop source, historic NPY or the previous PPO Zarr.
Always verify Rollout Reward indices, Episode boundaries, and source provenance.

## Git delivery protocol

```text
feature/flywheel
   -> feature/flywheel-data-contract
         <- squash Batch 01: gpt/lerobot-to-zarr
         <- squash Batch 02: gpt/merge-zarr
         <- squash Batch 03: gpt/staging-to-lerobot
         <- squash Batch 04: gpt/merge-lerobot
         <- squash Batch 05: gpt/data-contract-cleanup
   -> consolidated smoke/real-data tests and fixes
   -> final PR to feature/flywheel
```
