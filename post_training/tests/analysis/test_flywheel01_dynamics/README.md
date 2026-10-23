# Flywheel01 Dynamics / Critic Diagnostics (Single-step)

This directory is **new, read-only testing code**. It does not replace
`post_training/tests/analysis/evaluate_trained_dynamics.py`, change a training
checkpoint, or run BPPO.

## What is being compared?

The **true** state is the Frozen ACT Encoder output for the actually observed
three-RGB + 16D robot observation. The **predicted** state is a Dynamics
transition of the same Encoder's latent tokens, conditioned on the **actual
executed, normalized 16D action**.

We do **not** compare against raw MuJoCo qpos or predict RGB pixels.

- Teacher-forced one-step: T(E(o_t), a_t) vs E(o_(t+1)).
- Autoregressive open-loop: T(T(...T(E(o_t),a_t)...),a_(t+j)) vs E(o_(t+j+1)).
- Constant-from-start baseline: E(o_t) vs E(o_(t+j+1)).
- Q/V/Advantage sensitivity: same a_t, actual E(o_t) vs autoregressive
  predicted latent at t. Note this is a *consistency test*, not proof of
  Critic accuracy.
- Ensemble **elite mean** predictions, not random elite samples. This is a
  deterministic baseline: PPO's `dynamics.step()` samples ensemble members and
  Gaussian noise and may perform worse. That stochastic-path check is a
  follow-up, **not validated by the first plots**.

Outputs: `per_window.csv`, `by_step.csv`, `metadata.json`,
`latent_error_vs_horizon.png`, `advantage_disagreement.png` and
`terminal_error_distribution.png`.

## Flywheel01 paths: sourced from the uploaded **training logs**

On the Kuavo training machine:

```bash
REPO=/home/kuavo/changzhang/kdc_ARBiM
RUN=$REPO/post_training/outputs/sim_task1_flywheel01_single_step_stage12_ppo8k_bs256_20261017_073145
DATA=$REPO/data/flywheel_round_01_200Episode/sim_task1_flywheel01.zarr
IL=$REPO/outputs/train/sim_toy_pick/act_flywheel01/run_20261014_144346/epoch120
DYNAMICS=$RUN/stage1/dynamics/checkpoints/final/dynamics.pth
CRITIC_Q=$RUN/stage1/critic/checkpoints/final/Q.pt
CRITIC_V=$RUN/stage1/critic/checkpoints/final/value.pt
```

Paths above were **recorded in the original experiment config/metadata**. Their
presence on the current machine is checked at runtime. The uploaded ZIP did
not include actual model weights.

Run this checkout on the training machine. The code must be checked out on
this test branch or invoked via this branch's worktree.

```bash
cd /home/kuavo/changzhang/kdc_ARBiM
git fetch origin
git switch test/flywheel01-dynamics-diagnostics
conda activate arbim

python post_training/tests/analysis/test_flywheel01_dynamics/evaluate.py \
  --run-dir "$PWD/post_training/outputs/sim_task1_flywheel01_single_step_stage12_ppo8k_bs256_20261017_073145" \
  --source zarr \
  --split val \
  --windows 64 \
  --horizon 32 \
  --stride 32 \
  --device cuda:0 \
  --output-dir "$PWD/post_training/outputs/dynamics_diagnostics/flywheel01_offline"
```

This picks deterministic, chronologically ordered windows from the **same
episode-level validation split** as the experiment. It does not draw training
samples. Use `--windows 8` as a smaller smoke test. The seed and split come
from the resolved training config; no new model training occurs.

The `--dataset` and `--checkpoint` flags override paths from the recorded
config if the directory has moved. Stage-1 weights remain under `--run-dir`.

## MuJoCo / Base ACT real rollout comparison (no new sim runner)

This first version consumes **recorded MuJoCo Base ACT trajectories** in the
existing `arbim_rollout_staging_v2` format. It does **not** inject Dynamics
in the robot control loop.

On `feature/flywheel`, menu `9. data_wheel` produces staging NPY files:
`.../epoch..._flywheel/data_wheel/staging/*_staging.npy`, storing executed
actions plus T+1 actual observations. It saves only **successful** episodes.
For testing failed episodes too, a separate failure-inclusive recorder is
needed; avoid treating success-only performance as distribution-complete.

When those records are available, run:

```bash
python post_training/tests/analysis/test_flywheel01_dynamics/evaluate.py \
  --run-dir "$RUN" \
  --source staging \
  --staging-dir /path/to/base_act_rollouts/data_wheel/staging \
  --windows 64 --horizon 32 --stride 16 \
  --output-dir "$REPO/post_training/outputs/dynamics_diagnostics/base_act_mujoco"
```

**Alignment requirements:** staging must be made with the Base/Flywheel01 ACT
policy tested in the same simulator task, preserve T executed physical actions
and T+1 raw observations, and use the same 3-RGB camera/state contract.
For meaningful apples-to-apples results, prefer successful and failed
episodes, and include multiple simulator seeds.

## Interpretation

1. `teacher_mse` low but `rollout_mse` high: short-term model fits data,
   autoregressive rollout accumulates large errors.
2. `rollout_mse` near/above `zero_from_start_mse`: may not track state
   changes better than an unmoving baseline.
3. `adv_sign_flip`: how often the **same critic** changes its action-ranking
   sign when fed a predicted state instead of a true encoded state; both are
   estimates, so a low disagreement alone does not mean Q/V are correct.
4. Values are in learned latent units, not joint degrees/radians or physical
   displacements. Do not present MSE as a robot-space motion error.

The first Q/V plot is deliberately a sensitivity diagnostic. Validate Critic
independently against held-out returns and/or real sim outcomes before treating
an Advantage as ground truth.

## Limitations

- This test specifically matches `chunk_as_single_action=false`,
  `n_action_steps=32`, and ACT action dimension 16.
- The old analysis script constructs a 512D chunk-transition Dynamics and
  **cannot** be used unmodified for this checkpoint.
- Staging recordings must come from a trusted source
  (`np.load(..., allow_pickle=True)` loads pickled content).
- This version measures latent drift and critic sensitivity; direct
  decoder-output-action error is deferred because matching the frozen ACT
  decoder's positional embedding contract needs an additional equivalence
  test with actual Base ACT inference.
