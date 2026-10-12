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

**Depth policy (independent at each stage):** Data Wheel must capture all
three Depth cameras; Staging NPY, the converted Rollout LeRobot, and the
merged LeRobot archive must preserve them. Zarr is a derived PPO cache:
`use_depth: false` is the **default**. Set `use_depth: true` explicitly
in `data_prepare.yaml` (or `--zarr-use-depth` in real E2E) to add
Depth to a new Zarr. Both ACT/IL and PPO training use RGB-only by default
unless the *model/dataset* configuration is explicitly Depth-enabled.
A training setting never controls whether Depth was archived.

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
          |                         (official file-level copy/remux)
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

Set `lerobot_root`, a **new** `zarr_output_path`, and choose `use_depth`:
`false` (default) builds an RGB-only PPO Zarr **without removing any Depth
from the LeRobot source**; `true` opts into Depth arrays in the Zarr.
This covers Staging NPY → LeRobot → Zarr as well as Teleop LeRobot → Zarr.
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
**Reward-V2-valid RL subset**. All three Depth cameras are **required and
retained by default**. Data Wheel checks for them at capture time and the
converter rejects missing Depth; for historic RGB-only Staging use
`--rgb-only-legacy` deliberately, but never treat such data as a full archive.
Depth is stored in Staging as normalized quantized PNG uint8, **not raw
millimeter depths**.

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

Archival merge **requires all three Depth features by default**, and official
`lerobot.datasets.dataset_tools.merge_datasets()` additionally requires
the **complete** feature schema, FPS and robot type to match across sources.
No features are silently removed. LeRobot copies existing video streams,
remuxing only when needed (no framewise AV1 re-encode), then rewrites
Parquet indices/tasks/metadata. ARBiM writes
`meta/arbim_merge_provenance.json` to preserve Reward V2 source events,
including on repeated merges. `--allow-rgb-only` exists for intentional
legacy RGB-only archives; `--batch-size` is retained only for API
compatibility and does not control the official file-level operation.
A combined IL LeRobot is not the default RL input: convert each clean source
to RGB-only Zarr (unless explicitly opting into Depth), then merge Zarr to
preserve recorded Reward semantics.

## Safe migration and testing

Prior NPY utilities are **still available as deprecated compatibility code**
until the consolidated comparison test passes. No part of the **new** modes
requires a Processed NPY. Do not delete legacy artifacts solely because this
branch exists.

Run synthetic tests from the project root in the `arbim` environment:

```bash
bash post_training/tests/run_data_contract.sh
```

The complete real-data integration runner (new **work root** required).
Its Staging sources must have three Depth cameras; older RGB-only staging
from an earlier evaluation cannot be used to test this archive contract.
The following builds RGB-only Zarr by default; append `--zarr-use-depth`
to explicitly test Depth-enabled Zarr:
```bash
python post_training/tests/integration/check_data_contract_real.py \
  --source-npy data/sim_task1.npy \
  --staging-dir /path/to/epochbest_flywheel/data_wheel/staging \
  --reference-zarr data/sim_task1.zarr \
  --work-root outputs/data_contract_e2e_01 \
  --expected-teleop-episodes 600 \
  --expected-teleop-frames 90144
```

For a fast rerun after the 90,144-frame Teleop Zarr has already passed
the historical reference comparison, reuse that **unchanged** RGB-only Zarr:

```bash
python post_training/tests/integration/check_data_contract_real.py \
  --source-npy data/sim_task1.npy \
  --staging-dir /path/to/new/depth_enabled_flywheel/staging \
  --reference-zarr data/sim_task1.zarr \
  --reuse-teleop-zarr outputs/data_contract_e2e_03/teleop_reward_v2.zarr \
  --work-root outputs/data_contract_e2e_depth_01 \
  --expected-teleop-episodes 600 \
  --expected-teleop-frames 90144
```

The resumed test still creates and validates the new **Depth-enabled**
Data Wheel LeRobot, RGB-only Rollout Zarr, merged PPO Zarr and the new
full-Depth official LeRobot archive. Only Teleop Zarr conversion is skipped.
Use a new work root, never overwrite the old artifact. It also decodes real
Depth video frames at the boundary between Teleop and Rollout Episodes.

Requires substantial temporary disk space for a Teleop Zarr (unless reused),
Rollout Zarr, merged Zarr and a file-level merged LeRobot. Work on a new directory; it
does **not overwrite** Teleop source, historic NPY or the previous PPO Zarr.
Always verify Rollout Reward indices, Episode boundaries, and source provenance.

### Resuming after a failure at Step 5 (LeRobot IL merge)

The full E2E is expensive: converting 90,144 Teleop frames to Zarr can take
more than an hour. If Steps 1–4 already completed and Step 5 failed, **do not**
rerun them or reuse the incomplete `merged_il_lerobot/` directory. Instead:

```bash
python post_training/tests/integration/resume_data_contract_merge_lerobot.py \
  --source-npy data/sim_task1.npy \
  --work-root outputs/data_contract_e2e_03 \
  --reference-zarr data/sim_task1.zarr \
  --output-name merged_il_lerobot_resume_01 \
  --expected-teleop-episodes 600 \
  --expected-teleop-frames 90144 \
  --expected-rollout-episodes 4 \
  --expected-rollout-frames 600
```

This validates the already-built Teleop/Rollout/Merged Zarr episode and Reward
boundaries, reads existing Data Wheel LeRobot provenance and checks that source
LeRobot task metadata can be resolved. Only Step 5 creates new data, in the
explicitly named fresh output subdirectory. Step 6 reads the new combined
LeRobot data back and checks frame/Episode counts.

LeRobot v0.4.x stores `meta.tasks` as a pandas DataFrame indexed by task
description with a `task_index` column. The merger maps each Episode's
`task_index` to its source task text; it does **not** treat the DataFrame
as a boolean or silently replace readable original task strings.

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
