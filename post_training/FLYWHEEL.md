# ARBiM Data Wheel V1

Data Wheel is an opt-in MuJoCo evaluation mode. It **only records successful
rollouts during evaluation**; all post-evaluation data conversion and Offline RL
dataset builds are **manual**. No dedicated Data Wheel YAML is used.

## 1. Evaluate and record

Use `kuavo_deploy/eval_kuavo.py` with your regular evaluation config; choose
`9. data_wheel` in the task menu (ordinary `8. auto_test` is unchanged).

The existing evaluation writes videos and logs as usual, and additionally
stores successful rollout records in:

```text
outputs/eval/<task>/<method>/<timestamp>/epoch<epoch>/data_wheel/staging/
  rollout_<episode>_<time_ns>_staging.npy
```

Each staging file uses `arbim_rollout_staging_v2` and contains T actions,
T+1 observations (16D state and three JPEG RGB cameras), and Reward V2
annotations. The success signal is `/simulator/success`. Failed episodes and
interrupted episodes do not produce a staging file.

The existing Reward V2 assigns a sparse +1 at the later of two detected
gripper-release actions. A successful episode whose reward cannot be validated
is still saved, but marked `reward_status="invalid"` and `reward=None`.

## 2. Manually convert staging to Processed NPY

Run from the repository root:

```bash
python post_training/src/post_rl/data/rollout_to_npy.py \
  outputs/eval/<task>/<method>/<timestamp>/epoch<epoch>/data_wheel/staging \
  --output /path/to/processed/rollout_round_01.npy
```

Only successful staging episodes with valid Reward V2 are included. The
converter reports counts and skipped-invalid paths. The output is one
`arbim_processed_npy_stream_v2` file containing multiple episodes; it
preserves the existing JPEG bytes. It refuses to overwrite by default
(`--overwrite` explicitly replaces a prior output).

If no episodes have a valid reward, conversion fails rather than generating
unlabeled training data. The final T+1 observation remains in staging; the
current `run_build_db` follows the existing terminal transition convention
and repeats the final processed state at the terminal index.

## 3. Manually build a new Offline RL Zarr

Edit **only** `post_training/configs/data/data_prepare.yaml`; set:

```yaml
mode: build_db
use_depth: false
overwrite: false
zarr_output_path: /path/to/output/offline_rl_flywheel_round_01.zarr

teleop_sources:
  - name: original_teleop
    kind: teleop_npy
    path: /path/to/processed/original_teleop.npy
  - name: flywheel_round_01
    kind: rollout_npy
    path: /path/to/processed/rollout_round_01.npy
```

Then run:

```bash
python post_training/src/post_rl/data/data_prepare.py \
  --config post_training/configs/data/data_prepare.yaml
```

`teleop_sources` is the existing interface name and supports both source
types; `kind` is optional provenance metadata. The two NPY sources must have
compatible 16D state/action, camera dimensions and depth settings. Source
rewards are preserved; `run_build_db` recomputes discounted returns. Use a
**new** `zarr_output_path` on every round to avoid replacing the old dataset.

For the default horizon=50 without padding, any individual episode shorter
than 50 frames contributes **zero** training windows. The sampler does not
cross episode boundaries.

## 4. Smoke tests

In the configured ARBiM Python environment, from the repository root:

```bash
python post_training/tests/smoke/smoke_07_flywheel_recorder.py
python post_training/tests/smoke/smoke_08_flywheel_npy.py
python post_training/tests/smoke/smoke_09_flywheel_dataset.py
python post_training/tests/smoke/smoke_10_flywheel_eval.py
```

The tests cover the Recorder/Reward, NPY converter, mixed-source
Zarr/OfflineBuffer/50-step sampling, and mocked evaluation success/failure,
Stop, exception, and normal-mode paths. The last test executes the real
`run_single_episode` function with fake ROS/MuJoCo objects; it is **not** a
substitute for a live MuJoCo + ROS evaluation.

Before declaring real-world integration complete, use a short real MuJoCo
evaluation to confirm `/simulator/success`, correct 16D command/state
alignment, recorded JPEG images, and reasonable recorder overhead.
