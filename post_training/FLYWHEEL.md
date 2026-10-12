# ARBiM Data Wheel V1

Data Wheel is an opt-in MuJoCo evaluation mode. It **only records successful
rollouts during evaluation**; all post-evaluation data conversion and Offline RL
dataset builds are **manual**. No dedicated Data Wheel YAML is used.

## 1. Evaluate and record

Use `kuavo_deploy/eval_kuavo.py` and open the task menu. Choose
`9. data_wheel`. After selecting a task, the menu prompts for a YAML
config path (Enter can reuse a previously selected path). Data Wheel then
prompts for a **positive integer number of successful rollouts to collect**.
Ordinary `8. auto_test` requests only the YAML path and remains capped by
`inference.eval_episodes`.

In Data Wheel mode `inference.eval_episodes` is **not** the attempt limit:
the evaluator retries failed episodes until the requested success count is
reached or the user stops the task. Success means the existing MuJoCo
`/simulator/success` event, not Reward V2 validity. Successes with an invalid
Reward are still counted as collected staging files, but the manual converter
will skip them.

Normal `8. auto_test` writes to `epoch<epoch>/`. Data Wheel `9`
writes **all evaluation outputs** (logs, rollout videos and staging) under
`epoch<epoch>_flywheel/` instead, so the two modes do not share filenames.
The policy checkpoint lookup is unchanged.

Only menu `9. data_wheel` enables the post-success rollout tail: by default
the policy continues for **5 additional environment steps** after
`/simulator/success`, allowing the existing Reward V2 release detector
to observe the full closing-to-opening transition. The tail stops early on
`terminated` or `truncated`. Override its length for menu 9 using
`ARBIM_DW_POST_SUCCESS_STEPS=N` (allowed: 0–20; 0 disables it).
Menu `8. auto_test` explicitly clears all Data Wheel environment flags and
**always exits at success without the tail**; other menu tasks are unchanged.

The existing evaluation writes videos and logs as usual, and additionally
stores successful rollout records in:

```text
outputs/eval/<task>/<method>/<timestamp>/epoch<epoch>_flywheel/data_wheel/staging/
  rollout_<episode>_<time_ns>_staging.npy
```

Each staging file uses `arbim_rollout_staging_v2` and contains T actions,
T+1 observations (16D state and three JPEG RGB cameras), and Reward V2
annotations. If the configured environment exposes all three normalized
Depth observations (`observation.depth_h/l/r`), the Recorder also stores
them as lossless **single-channel PNG uint8** in staging. This mirrors the
existing Rosbag→LeRobot per-frame depth quantization (`[0,1]*255` cast to
uint8) while avoiding redundant three-channel storage. It does not store
raw millimeter Depth. No missing Depth frames are synthesized; partial
Depth availability or changes within an episode raise an error.
The success signal is `/simulator/success`. Failed episodes and
interrupted episodes do not produce a staging file.

The existing Reward V2 assigns a sparse +1 at the later of two detected
gripper-release actions. A successful episode whose reward cannot be validated
is still saved, but marked `reward_status="invalid"` and `reward=None`.

## 2. Export successful rollouts to LeRobot (new canonical path)

The **long-term format is now LeRobot**, not Processed NPY. For the new
data-format contract and merge options, see [DATA_CONTRACT.md](DATA_CONTRACT.md).

```bash
python post_training/src/post_rl/data/staging_to_lerobot.py \
  outputs/eval/<task>/<method>/<timestamp>/epoch<epoch>_flywheel/data_wheel/staging \
  --template-lerobot-root /path/to/teleop_lerobot \
  --output-root /path/to/new_rollout_lerobot \
  --repo-id lerobot/arbim_rollout_round_01 \
  --task "ARBiM Data Wheel" \
  --require-valid-reward
```

Omit `--require-valid-reward` if collecting successful demonstrations for **IL**,
including those that have an invalid Reward V2. For RL, only episodes with
validated Reward V2 are eligible. LeRobot stores T action-aligned frames; the
original Staging preserves the final extra observation T+1.

## 3. Create and combine Offline RL Zarr datasets

First use `mode: lerobot_to_zarr` with `reward_source: auto` to convert
each original Teleop / Data Wheel LeRobot dataset directly to a Zarr. Then use
`mode: merge_zarr` to combine the resulting compatible Zarrs.
This avoids the former Staging → Processed NPY → Zarr intermediate path.

```bash
python post_training/src/post_rl/data/data_prepare.py \
  --config post_training/configs/data/lerobot_to_zarr.yaml

python post_training/src/post_rl/data/data_prepare.py \
  --config post_training/configs/data/merge_zarr.yaml
```

Edit the example paths before running. Both conversions refuse overwrite by
default and leave previous RL datasets untouched. For ACT, use
`post_training/src/post_rl/data/lerobot_merge.py` to combine multiple LeRobot
sources.

The old `rollout_to_npy.py` / `raw_to_npy` / `build_db` adapters remain
temporarily for compatibility and regression testing, but are **deprecated**.
Do not use them for new Data Wheel rounds. Without sampler padding, trajectories
under the configured action horizon (normally 50) provide no training windows.

## 4. Smoke tests

In the configured ARBiM Python environment, from the repository root:

```bash
python post_training/tests/smoke/smoke_07_flywheel_recorder.py
python post_training/tests/smoke/smoke_08_flywheel_npy.py
python post_training/tests/smoke/smoke_09_flywheel_dataset.py
python post_training/tests/smoke/smoke_10_flywheel_eval.py
python post_training/tests/smoke/smoke_11_flywheel_target.py
```

These legacy tests cover the Recorder/Reward, NPY converter,
Zarr/OfflineBuffer/50-step sampling, optional LeRobot-format Depth,
and mocked evaluation success/failure, Stop, exception, and normal-mode paths.
The quota test verifies success-target
behavior separately. The Eval test uses fake ROS/MuJoCo objects; it is **not** a
substitute for a live MuJoCo + ROS evaluation.

For the new contract run `bash post_training/tests/run_data_contract.sh`
and both real-data integration scripts documented in DATA_CONTRACT.md.

Before declaring real-world integration complete, use a short real MuJoCo
evaluation to confirm `/simulator/success`, correct 16D command/state
alignment, recorded JPEG images, and reasonable recorder overhead.
