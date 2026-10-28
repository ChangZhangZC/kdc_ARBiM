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

## Second pass: exact PPO stochastic Dynamics diagnostics

The first-pass `evaluate.py` used only the deterministic **mean over Elite
means**. PPO actually calls `EnsembleDynamics_batch.step()`, which **randomly
chooses one Elite** and adds a Gaussian noise sample at **each environment
step**. To measure this extra stochastic model error, run the new script:

```bash
cd /home/kuavo/changzhang/kdc_ARBiM
git fetch origin
git switch test/flywheel01-dynamics-diagnostics
conda activate arbim

# CPU-only synthetic tests; no checkpoints required
python post_training/tests/analysis/test_flywheel01_dynamics/test_smoke.py
python post_training/tests/analysis/test_flywheel01_dynamics/test_stochastic_smoke.py

# Same Flywheel01 validation source; 8 stochastic replicas/window
WINDOWS=64 REPEATS=8 DEVICE=cuda:0 \
  bash post_training/tests/analysis/test_flywheel01_dynamics/run_stochastic_flywheel01.sh
```

Default output is
`post_training/outputs/dynamics_diagnostics/flywheel01_stochastic/`:

- `stochastic_vs_elite_mean.png`: deterministic/stochastic rollout, one-step
  errors, and stationary baseline. Shaded 10th–90th percentiles describe the
  spread across windows and replicas; they are **not confidence intervals**.
- `stochastic_advantage_sign.png`: sign disagreement for all samples and
  for `|A_real| >= 1e-4`. Advantage is always evaluated **before** applying
  the action being scored.
- `normalized_advantage_direction.png`: per-step **batch-normalized** Advantage sign disagreement (all actions and larger-magnitude `|A_real_normalized|>=0.5`). The recorded W windows form a surrogate PPO batch, not the actual PPO minibatch.
- `advantage_margin_sensitivity.png`: sensitivity to thresholds
  `0, 5e-5, 1e-4, 2e-4, 3e-4` and the number of samples retained.
- `stochastic_by_step.csv`: aggregated means, percentiles, filtered flip
  rates, and exact eligible counts at each step.
- `stochastic_per_replica.csv`: window x replica x step records with individual
  Q, V, Advantage and latent MSE.
- `deterministic_per_window.csv`: corresponding Elite-mean baseline.
- `stochastic_metadata.json`: seed, number of windows and replicas, alignment.

Direct invocation with explicit overrides:

```bash
python post_training/tests/analysis/test_flywheel01_dynamics/evaluate_stochastic.py \
  --run-dir "$RUN" --dataset /path/to/flywheel01.zarr \
  --source zarr --split val --windows 64 --repeats 8 \
  --horizon 32 --stride 32 --seed 20261023 \
  --output-dir "$REPO/post_training/outputs/dynamics_diagnostics/flywheel01_stochastic"
```

To compare recorded Base ACT MuJoCo trajectories, replace
`--source zarr --split val` with
`--source staging --staging-dir /path/to/trusted/data_wheel/staging`.
Do not pass unknown/untrusted NPY files (`allow_pickle=True`).

**Scope and interpretation:**

- Using `dynamics.step()` duplicates the PPO model's stochastic transition
  mechanism. We still feed **recorded actions** from offline/MuJoCo data,
  **not actions resampled from PPO's old Gaussian ACT**, so this remains a
  Dynamics diagnostic, not a full replay of PPO's sampling distribution.
- `adv_horizon=0` (step=1) should show zero sign disagreement because both
  methods use the same observed pre-action latent. At `step=32`,
  `adv_horizon=31`, while the plotted state MSE is after the 32nd transition.
- Filtering out small `|A_real|` reduces the effect of near-zero signs, but
  this `A_real=Q(z,a)-V(z)` is **still a learned Critic estimate**, not
  true/ground-truth Advantage.
- Replicas of the same recorded window share observations and actions; do
  not interpret `windows*repeats` as the number of independent episodes.
- No original analysis script, model weight, training code or reward data is
  modified.

## Why normalized Advantage is reported separately

BPPO's `_normalize_advantage_by_step()` subtracts the mini-batch mean
**separately at every action position**, then divides by the position-wise
standard deviation. A raw Q−V sign flip does **not** necessarily mean an
opposite PPO gradient direction after this centering.

The new comparison constructs, for each position, a W-window batch of real
and predicted Q−V, centers/scales them separately, then compares their signs.
For a random Dynamics rollout it does this separately for every stochastic
replica. It reports both unrestricted sign disagreement and the subset where
the real batch-normalized advantage magnitude is at least 0.5. Eligible
window counts appear in `stochastic_by_step.csv`. If no samples pass a
threshold, the rate is NaN; it is not zero.

Because our input actions are from recorded data rather than Old ACT samples
and W=64 rather than the actual PPO mini-batch, these are **sensitivity
diagnostics**, not an exact reproduction of the optimizer's update directions.

## Third pass: split random Elite selection from Gaussian noise (P0 + P1)

The older stochastic run reported 32-step MSE of 3.892, while Elite-mean
rollout had MSE 0.301. This result alone **cannot** identify the source of the
extra squared error.

New independent scripts (do not overwrite the original run output):

```bash
cd /home/kuavo/changzhang/kdc_ARBiM
git fetch origin
git switch test/flywheel01-dynamics-diagnostics
git pull --ff-only
conda activate arbim

# P0: read previous already-computed normalized Advantage metrics; no GPU
python post_training/tests/analysis/test_flywheel01_dynamics/report_existing_advantage.py

# P1: no-checkpoint CPU smoke (including plot creation)
python post_training/tests/analysis/test_flywheel01_dynamics/test_components_smoke.py

# P1: trained models + same Flywheel01 validation windows; no MuJoCo
WINDOWS=64 REPEATS=8 DEVICE=cuda:0 \
  bash post_training/tests/analysis/test_flywheel01_dynamics/run_components_flywheel01.sh
```

New output directory:
`post_training/outputs/dynamics_diagnostics/flywheel01_components/`.

- `three_mode_rollout_mse.png`: **A** = average of Elite model
  predictions (no sampling), **B** = random Elite model mean, **C** = random
  Elite plus sampled Gaussian noise (the PPO state rollout).
- `teacher_one_step_components.png`: analogous one-step measurements always
  starting from the **true observation**.
- `variance_calibration.png`: mean epistemic (between Elite), aleatoric
  (within-model Gaussian), empirical error, plus moment-matched ±1σ and ±2σ
  coverage of the true latent. Reference 68% and 95% coverage assume a
  Gaussian; an Elite mixture is not necessarily Gaussian, so these are
  approximate calibration diagnostics.
- `component_advantage_direction.png`: raw Q−V and separately per-step
  batch-normalized Q−V sign differences, plus a larger-normalized-margin subset.
- `component_per_replica.csv`: exact per-window/per-seed trajectory errors,
  assigned Elite ID, and raw/normalized Advantages.
- `component_calibration.csv`: one-step predicted variance vs actual errors.
- `component_by_step.csv`: aggregated step 1–32 diagnostics.
- `component_metadata.json`: run provenance.

**Comparison protocol:** B and C start from the same observed state and
execute the same recorded action sequence. Both use the same random Elite
IDs at every step, but B uses the model's predicted mean while C calls the
actual `dynamics.step()` with Gaussian noise. After the first step B and C
enter different predicted states; their 32-step MSE difference is therefore
the **cumulative effect of adding noise**, not a per-step additive variance
term. The teacher-forced comparison isolates local conditional error more
directly.

This is still offline dataset evaluation, **not** a MuJoCo rollout or direct
sampling from BPPO's old Gaussian ACT. Model weight files are not changed.
