# ARBiM Post-RL tests

This directory is intentionally split by purpose. Test/diagnostic code lives here so the Post-RL training implementation can remain unchanged while validation work continues on a dedicated test branch.

## Directory contract

```text
post_training/tests/
├── smoke/      # pass/fail regression tests for executable V1 contracts
├── analysis/   # offline measurement/report scripts; no pass/fail claim implied
└── rollout/    # simulator/robot shadow diagnostics on live rollout observations
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
| 03 | `analysis/analysis_03_policy_drift.py` | compare IL vs exported Post-RL deterministic weights and same-observation actions | new; primary policy-drift diagnostic |
| 04 | `analysis/analysis_04_rgb_storage_estimate.py` | estimate JPEG RGB storage before full data conversion | retained; moved out of smoke because it is capacity analysis rather than pass/fail testing |

## Rollout diagnostics

`rollout/rollout_01_shadow_policy_compare.py` always runs the original IL ACT and exported deterministic Post-RL ACT on the exact same live simulator observations. In plain shadow mode, one policy controls the robot and the other is observation-only. The CSV records Post-RL action, IL action, the actual executed action, IL/Post-RL disagreement, and each policy's step-to-step action change.

For the current Post-RL fixed-point issue, the script also supports a switch-control recovery test. Start with Post-RL controlling the robot, then switch to IL either at an explicit rollout step (`--switch-step`) or automatically when the rolling Post-RL action change is small while IL/Post-RL disagreement stays large (`--switch-on-freeze`). Once switched, IL controls the rest of that episode while both policies continue to be evaluated and logged.

The automatic detector is configurable with:

```text
--freeze-min-step
--freeze-window
--freeze-step-delta-max
--freeze-policy-delta-min
```

A recovery after the Post-RL -> IL handoff is direct evidence that the frozen simulator state is still recoverable by the original IL policy and that the Post-RL action mapping is responsible for maintaining the fixed point. Failure to recover does not by itself prove the opposite, because the robot may already have entered a state outside both policies' recoverable support.

## V1 alignment notes

- Current formal Post-RL artifacts use a complete stochastic `best_ope/` bundle and a separate deterministic exported `epochbest/` bundle.
- `smoke_09_checkpoint_export.py` tests the stochastic -> deterministic export contract directly.
- Current V1 stochastic-policy bounds are `init_log_std=-3.5`, `log_std_min=-5.0`, `log_std_max=-2.2`. `analysis_01_action_sigma.py` predates the final max bound, so pass `--max-log-std -2.2` when reproducing the V1 sigma analysis.
- Chunk size is checkpoint/config dependent. `run_smoke.sh` accepts an optional chunk-size argument so the suite can validate the current 32-step experiment as well as other aligned ACT checkpoints without changing production YAML defaults.
- The current Post-RL training/release branches are not modified by this test-suite work; changes are developed on a GPT branch and are intended to be squash-merged only into the dedicated test branch after review.
