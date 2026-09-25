# ARBiM Post-RL tests

This directory is intentionally split by purpose. Test/diagnostic code lives here so the Post-RL training implementation can remain unchanged while validation work continues on a dedicated test branch.

## Directory contract

```text
post_training/tests/
├── smoke/      # pass/fail regression tests for executable V1 contracts
├── analysis/   # offline measurement/report scripts; no pass/fail claim implied
└── rollout/    # simulator/robot diagnostics on live rollout observations
```

## Smoke sequence

The smoke numbering follows the Post-RL data/training/export chain.

| Order | Script | Purpose | V1 status |
| --- | --- | --- | --- |
| 00 | `smoke/smoke_00_data_prepare.py` | LeRobot -> streamed NPY -> JPEG Zarr data contract | current; manual because it needs the original LeRobot dataset |
| 01 | `smoke/smoke_01_contract.py` | fixed/configurable Post-RL contract guards | current |
| 02 | `smoke/smoke_02_policy_data.py` | Offline dataset -> stochastic ACT -> shared latent frontend | current |
| 03 | `smoke/smoke_03_critic.py` | IQL Q/V forward, update, frozen encoder/target-Q checks | current |
| 04 | `smoke/smoke_04_dynamics.py` | transition model shape/update checks | current |
| 05 | `smoke/smoke_05_ppo.py` | Offline PPO ratio/advantage/update and monitoring checks | current; includes explicit unsupported-mode guard cases |
| 06 | `smoke/smoke_06_latent_cache.py` | endpoint sampling + frozen ACT latent cache reuse | current |
| 07 | `smoke/smoke_07_stage_gates.py` | Stage-1 / Stage-2 gate behavior | current |
| 08 | `smoke/smoke_08_e2e.py` | tiny Critic -> Dynamics -> PPO run, best-OPE bundle, exact resume | current |
| 09 | `smoke/smoke_09_checkpoint_export.py` | stochastic best-OPE bundle -> deterministic Kuavo ACT export | current; manual because it needs a stochastic Post-RL checkpoint |

`smoke/run_smoke.sh` runs the core 01-08 suite. Smoke 00 and 09 have different input contracts and remain explicit/manual tests.

## Analysis sequence

| Order | Script | Purpose | V1 status |
| --- | --- | --- | --- |
| 01 | `analysis/analysis_01_action_sigma.py` | estimate ACT residual scale for stochastic-policy sigma design | retained historical design diagnostic; for current V1 use `--max-log-std -2.2` |
| 02 | `analysis/analysis_02_dynamics_eval.py` | evaluate trained dynamics one-step/multi-step behavior and uncertainty | retained; Stage-1 model diagnostic |
| 03 | `analysis/analysis_03_policy_drift.py` | compare IL vs exported Post-RL deterministic weights and same-observation actions | current; primary policy-drift diagnostic |
| 04 | `analysis/analysis_04_rgb_storage_estimate.py` | estimate JPEG RGB storage before full data conversion | retained; moved out of smoke because it is capacity analysis rather than pass/fail testing |
| 05 | `analysis/analysis_05_terminal_advantage.py` | test whether late/terminal-like actions are overvalued before true episode end using cached latents and trained IQL Q/V | current; offline diagnostic only, no training changes |

### Analysis 05: Terminal / Advantage diagnostic

This diagnostic targets the hypothesis that Post-RL may enter a premature terminal-like fixed point during the second half of the task. It does not assume that ACT receives an explicit done flag. Instead it tests whether the trained critic assigns excessive value or positive advantage to hold-like or true terminal-tail action chunks before the real episode end.

It reuses the existing Offline RL Zarr, frozen ACT latent cache, Stage-1 IQL Q/V checkpoint, Base ACT checkpoint, and a Post-RL checkpoint. The analysis always scans valid chunk anchors at diagnostic stride 1, while separately reporting the configured dataset, critic, and PPO-finetune strides so sampler coverage is not silently conflated.

For each valid observation anchor it compares:

- demonstration action chunk;
- Base ACT deterministic mean;
- Post-RL deterministic mean;
- a hold proxy that repeats the first demonstration action across the chunk;
- the final H-action terminal template from the same episode.

The primary outputs are `terminal_window_coverage.csv`, `episode_action_motion.csv`, `qva_per_anchor.csv`, phase/progress summaries, plots, and `summary.json`. The key quantities are `Q`, `V`, `A=Q-V`, and whether Post-RL/hold/terminal-template actions outrank the Base ACT continuation action during the second half of the episode.

## Rollout diagnostics

| Order | Script | Purpose |
| --- | --- | --- |
| 01 | `rollout/rollout_01_shadow_policy_compare.py` | IL controls the simulator for the whole episode; Post-RL is shadow-only on the exact same observations. Measures whether PPO action output drifts on an IL-generated trajectory. No control switch is allowed. |
| 02 | `rollout/rollout_02_switch_policy_compare.py` | Post-RL controls first while IL runs in shadow, then control switches once from Post-RL to IL either at a fixed step or after fixed-point detection. Tests whether the state is still recoverable by IL. |
| 03 | `rollout/rollout_03_latent_ood.py` | Scores each live rollout observation against the already-built demonstration ACT latent cache. Uses the V1 critic representation: ACT encoder tokens followed by mean-token readout, then standardized kNN distance calibrated on held-out demonstration latents. |

### Rollout 01: IL trajectory / Post-RL shadow

This is intentionally a no-switch baseline. The original deterministic IL ACT always supplies the executed action. The exported deterministic Post-RL ACT sees the identical preprocessed observation and only contributes diagnostic actions. The CSV records IL/Post-RL disagreement and each policy's step-to-step action change.

### Rollout 02: switch-control recovery

This is the causal recovery experiment. Post-RL initially controls the robot. With `--switch-step`, control moves to IL at a specified step. With `--switch-on-freeze`, the handoff occurs when the rolling Post-RL action change is small while IL/Post-RL disagreement remains large. After the handoff, IL controls the rest of the episode and Post-RL becomes shadow-only.

A recovery after the handoff shows that the state was still recoverable by the original IL policy. Failure to recover does not prove that IL and Post-RL are equivalent: Post-RL may already have moved the robot outside IL's recoverable support.

### Rollout 03: latent-space distribution shift

This diagnostic reuses the existing frozen latent cache instead of re-encoding the demonstration images. `obs_latent.npy` stores ACT encoder token latents for the demonstration/offline dataset. The script mean-pools those tokens exactly like `ACTCriticEncoder.forward`, forms a random reference/calibration split, standardizes each latent feature using the reference split, and computes k-nearest-neighbor distance.

For each live rollout observation it records:

```text
latent_knn_distance
latent_demo_percentile
latent_above_demo_p95
latent_above_demo_p99
executed_step_delta_l2
```

`latent_demo_percentile` is a rank relative to held-out demonstration samples, not an OOD probability. A value near 99 means the live observation is farther from the demonstration reference than about 99% of held-out demonstration observations under this representation/metric.

The cache contract is checked before rollout: the live ACT encoder fingerprint must exactly match the `encoder_sha256` stored in the cache metadata. This is important because the distance is only meaningful when rollout and demonstration latents share the same frozen coordinate system.

## V1 alignment notes

- Current formal Post-RL artifacts use a complete stochastic `best_ope/` bundle and a separate deterministic exported `epochbest/` bundle.
- `smoke_09_checkpoint_export.py` tests the stochastic -> deterministic export contract directly.
- Current V1 stochastic-policy bounds are `init_log_std=-3.5`, `log_std_min=-5.0`, `log_std_max=-2.2`. `analysis_01_action_sigma.py` predates the final max bound, so pass `--max-log-std -2.2` when reproducing the V1 sigma analysis.
- Chunk size is checkpoint/config dependent. `run_smoke.sh` accepts an optional chunk-size argument so the suite can validate the current 32-step experiment as well as other aligned ACT checkpoints without changing production YAML defaults.
- The current Post-RL training/release branches are not modified by this test-suite work; changes are developed on a GPT branch and are intended to be squash-merged only into the dedicated test branch after review.
