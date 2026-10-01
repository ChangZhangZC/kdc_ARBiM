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
| 00 | `analysis/analysis_00_dataset_episode_integrity.py` | audit LeRobot episode boundaries, Zarr transition/terminal contracts, sampler windows, and one-grasp/one-release semantics for both grippers | current; dataset sanity check before Critic/PPO analysis |
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
| 06 | `analysis/analysis_06_ppo_local_advantage.py` | test whether terminal-directed perturbations inside the actual PPO Gaussian sampling neighborhood receive higher IQL advantage | current; uses PPO finetune stride and no training updates |
| 07 | `analysis/analysis_07_ppo_gradient_alignment.py` | estimate the Gaussian PPO score-function mean gradient and compare it with terminal direction and observed IL-to-Post-RL action drift | current; first-order action-mean diagnostic only, no training updates |
| 08 | `analysis/analysis_08_one_step_ppo_replay.py` | replay one production Offline PPO optimizer step from Base ACT and measure the resulting deterministic ACT output drift on fixed probe states | current; mutates only an in-memory PPO clone and never overwrites training artifacts |
| 09 | `analysis/analysis_09_multi_batch_one_step.py` | repeat independent one-step PPO replays from the same Base ACT over many shuffled finetune batches/seeds and estimate the expected deterministic action drift | current; isolates batch/sample variance before studying multi-step accumulation |

### Analysis 00: Dataset episode integrity

This diagnostic is the dataset-level prerequisite for the later Critic/PPO analyses. It audits the actual LeRobot source and Offline RL Zarr used by sim_task1 without decoding RGB, and it treats episode semantics separately from structural consistency.

The current Rosbag -> LeRobot converter contract is also documented by code inspection: `kuavo_data/CvtRosbag2Lerobot.py` iterates over selected rosbag files and calls `dataset.save_episode()` exactly once after each bag is processed. Therefore one successfully converted rosbag file maps to one LeRobot episode. Because the original bags are not available in the current debugging setup, the script cannot verify whether each original bag itself represented one complete task; it verifies the resulting LeRobot and Zarr data instead.

The structural audit checks `episode_index`, optional `frame_index`, optional `timestamp`, repeated/non-contiguous episode IDs, episode lengths, LeRobot-to-Zarr row ordering, `episode_ends`, `done`, `timeout`, terminal self-loops, `next_index`, nonterminal positive rewards, transition next-state/next-action consistency, return recurrence, and real `SequenceSampler` windows at configurable horizon/strides. This distinguishes “the conversion pipeline preserved its own boundaries” from “the boundaries were semantically correct.”

The task-semantic audit uses the 16D bimanual action contract by default: left gripper index 7 and right gripper index 15. For sim_task1, a valid complete episode is expected to contain exactly one debounced `open -> closed -> open` cycle on each side, corresponding to one grasp and one release per hand. Open/closed numerical orientation is inferred from episode starts by default and can be overridden; low/high hysteresis thresholds and a minimum dwell filter suppress continuous-action noise. Episodes that are structurally valid but fail one or both gripper cycles are reported separately rather than silently treated as successful demonstrations.

Primary outputs are `episode_integrity.csv`, `suspicious_episodes.csv`, `gripper_events.csv`, `sampler_checks.csv`, and `summary.json`. The next visualization diagnostic should use the suspicious episode IDs and detected gripper event frames to render start / grasp / release / end RGB contact sheets for manual semantic verification.

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

The expanded counterfactuals test three more links in the hypothesis. First, the terminal action is replaced by the next episode's terminal chunk while the current state is fixed, and the current episode's terminal action is also evaluated on a progress-matched state from the next episode. This separates same-episode compatibility from a generic terminal-action shortcut. Second, the normalized action-chunk RMSE from Base ACT and Post-RL to the same terminal template is compared; a negative `postrl_terminal_delta_rmse` means Post-RL moved closer to the terminal-action manifold. Third, for the 16D bimanual contract `[left7, left_gripper, right7, right_gripper]`, terminal components are injected into the Base ACT chunk one arm / joint group / gripper at a time to localize which side drives any Q overvaluation.

### Analysis 06: PPO local sampling advantage

This diagnostic tests the missing causal link left by Analysis 05. Analysis 05 can show that far terminal-like counterfactual actions receive excessive Q, but PPO only updates from action chunks that the old stochastic policy can actually sample. Analysis 06 therefore reconstructs the local Gaussian action neighborhood on the real PPO finetune anchors (using `dataset.finetune_sequence_stride`) and samples many chunks around each policy mean without changing any model weights.

For each sampled chunk it computes raw IQL advantage `Q(s,a)-V(s)`, applies the same optional PPO temperature transform used before advantage normalization, and measures the signed projection of the sampled perturbation toward the same-episode terminal action template. Correlations are computed within each state across local samples before being averaged, so state-to-state value differences do not create a false terminal correlation.

The diagnostic runs on both the initial stochastic Base ACT neighborhood and the final `best_ope` Post-RL neighborhood by default. It reports whole-action and bimanual component projections for left/right joints and grippers. Positive local correlation, positive top-score-minus-bottom-score terminal projection, and positive best-score projection mean that PPO-accessible perturbations toward the terminal manifold are systematically preferred by the Critic. Near-zero or negative values mean the far terminal-template Q anomaly from Analysis 05 is not locally reachable through PPO sampling and is less likely to explain policy drift.

### Analysis 07: PPO gradient alignment

This diagnostic turns the local correlations from Analysis 06 into a first-order mean-space PPO update estimate. For a Gaussian policy it Monte-Carlo estimates `E[score * (a-mu) / sigma^2]` on the real PPO finetune anchors, where `score` matches the Offline PPO pre-normalization advantage transform. The primary estimator subtracts the within-state sample-score mean as a control variate; because an action-independent baseline has zero expected score-function gradient, this reduces Monte-Carlo variance without changing the expected direction.

The estimated action-mean gradient is compared with two directions on the exact same cached observations:

- the same-episode terminal direction, `terminal_chunk - current_policy_mean`;
- the observed deterministic policy drift, `PostRL_mean - BaseIL_mean`.

The comparison is reported for the full chunk and separately for left/right joints and grippers. Positive `cos(g, terminal)` means the local PPO update points terminal-ward. Positive `cos(g, PostRL-drift)` means the estimated update is aligned with the final deterministic action drift actually observed after Post-RL. `cos(terminal, PostRL-drift)` checks whether that observed drift itself is terminal-ward.

This is deliberately not a full optimizer replay. It estimates the score-function direction with respect to the action mean when samples are drawn from the old/reference policy, where the PPO ratio starts at one and clipping is inactive. It does not reconstruct the Transformer parameter Jacobian, repeated clipped epochs on the same samples, or intermediate old-policy snapshots after OPE reference refreshes.

### Analysis 08: one-step real PPO replay

This diagnostic moves from action-space theory to the actual shared ACT network update. It constructs the same Stage-2 PPO object and finetune dataloader used by production training, starts from the Base ACT checkpoint, draws the first reproducible shuffled finetune batch, and calls the production `BehaviorProximalPolicyOptimization.update_distribution()` exactly once with the trained Stage-1 critic and the configured learning-rate / clip-decay flags.

Before the update it probes the deterministic Base ACT mean on all real PPO-stride anchors and also records the final best-OPE Post-RL mean on those exact cached observations. After the single real optimizer step it probes the mutated PPO policy again. The resulting one-step drift is compared with both the same-episode terminal direction and the final observed IL-to-Post-RL drift.

Metrics are reported for the full 32x16 chunk and separately for left/right joints and grippers. This is the first analysis in the terminal-bias chain that includes the real ACT decoder/action-head parameter Jacobian and parameter sharing: if a gripper-driven PPO signal also creates systematic arm-joint output drift after one optimizer step, the coupling is now observable directly rather than inferred from action-space gradients.

The script previews the exact stochastic batch action and IQL advantage, restores the RNG state, and then lets the production update sample the identical action chunk. It also verifies that the PPO policy clone exactly matches Base ACT before mutation. The scope is intentionally one optimizer step only; it does not reproduce the full 8000-step accumulation, OPE old-policy refreshes, EMA selection, or intermediate optimizer state.

### Analysis 09: multi-batch one-step PPO expectation

Analysis 08 showed that a single real PPO optimizer step can produce a complicated shared-network output drift: some gripper components may move terminal-ward while other gripper/arm components move the opposite way. Analysis 09 tests whether that pattern is systematic or just one shuffled batch / policy-sampling realization.

Each repeat starts from the exact same Base ACT checkpoint with a fresh current policy, fresh old/reference policy, fresh optimizer/scheduler state, one independently shuffled production finetune batch, and one production `update_distribution()` call. The final best-OPE Post-RL policy and the fixed probe observations are held constant across repeats. The script therefore estimates the expectation of the first PPO update rather than accumulating training history.

Two summaries are intentionally separated. First, every repeat gets its own phase-level alignment statistics, so the distribution and sign consistency across repeats are visible. Second, the action drift vectors are averaged per fixed probe state before cosine metrics are computed. This `E[delta action]` view answers whether there is a systematic expected network-output direction after marginalizing over one-step batch/sample noise.

The output also reports an expected-drift signal-to-repeat-noise ratio: the RMSE magnitude of the across-repeat mean drift divided by the RMS standard deviation across repeats for the same action group. A stable positive terminal alignment with useful SNR would support a systematic first-step mechanism. Near-zero expected drift or low sign consistency would indicate that the Analysis 08 direction was dominated by batch/sample variance and that later debugging should focus on multi-step accumulation, optimizer state, clipping, or old-policy refresh dynamics instead.

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
