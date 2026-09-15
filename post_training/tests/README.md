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
| 01 | `analysis/analysis_01_action_sigma.py` | estimate ACT residual scale for stochastic-policy sigma design | retained; design diagnostic, not a regression gate |
| 02 | `analysis/analysis_02_dynamics_eval.py` | evaluate trained dynamics one-step/multi-step behavior and uncertainty | retained; Stage-1 model diagnostic |
| 03 | `analysis/analysis_03_policy_drift.py` | compare IL vs exported Post-RL deterministic weights and same-observation actions | new; primary policy-drift diagnostic |
| 04 | `analysis/analysis_04_rgb_storage_estimate.py` | estimate JPEG RGB storage before full data conversion | retained; moved out of smoke because it is capacity analysis rather than pass/fail testing |

## Rollout diagnostics

`rollout/rollout_01_shadow_policy_compare.py` executes one deterministic ACT policy in the normal Kuavo simulator and shadow-runs the other policy on the exact same preprocessed observations. It records per-step IL/Post-RL action differences and each policy's step-to-step action change without allowing the shadow policy to affect the robot trajectory.

The intended first diagnosis for the current Post-RL freeze issue is:

```text
execute: Post-RL deterministic ACT
shadow : original IL deterministic ACT
```

If the Post-RL step-to-step action change collapses toward zero while the IL shadow policy still produces meaningful changes on those same observations, the freeze is policy-side rather than an environment/action-execution stall.

## V1 alignment notes

- Current formal Post-RL artifacts use a complete stochastic `best_ope/` bundle and a separate deterministic exported `epochbest/` bundle.
- `smoke_09_checkpoint_export.py` tests the stochastic -> deterministic export contract directly.
- Chunk size is checkpoint/config dependent. `run_smoke.sh` accepts an optional chunk-size argument so the suite can validate the current 32-step experiment as well as other aligned ACT checkpoints without changing production YAML defaults.
- The current Post-RL training/release branches are not modified by this test-suite work; changes are developed on a GPT branch and are intended to be squash-merged only into the dedicated test branch after review.
