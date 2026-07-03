# Goal-Tracking Substrate Guiding GR00T on Long-Horizon Tasks

## Result

**Goal:** Build a goal-tracking substrate (Liquid) that guides GR00T transformer through long-horizon tasks.

**Outcome:** 6 substrate variants were built and tested on chained-LIBERO long-horizon benchmark. None improved completion rate over GR00T-alone baseline. The only positive intervention was a non-substrate physical heuristic (conditional retract, +9.2pp pooled 3-seed).

The substrate-as-inference-time-overlay architecture, applied to a frozen pre-trained GR00T, does not improve long-horizon task completion on this benchmark family. The reason is architectural, not a tuning issue.

## Setup

- Actor: GR00T-N1.7-LIBERO checkpoint (3B params, frozen)
- Substrate: GoalImageSubstrate variants (~317K params, K=8 belief positions, ContinuousDynamics ODE)
- Benchmark: chained-LIBERO L=3 (3 sub-tasks per chain, each from libero_10 multi-step task suite)
- Eval: 6 chains × 3 rollouts × 3 sub-tasks = 54 trials per seed; 3 seeds = 162 trials
- Step budget: 720 per sub-task (unconstrained); 1100 shared across L=5 chain (constrained)

## Variants Tested

| # | Variant | Training | Intervention | Result |
|---|---|---|---|---|
| 1 | Gripper-tracker substrate | BCE on (override needed=1 iff expert opens AND v10 closes) | Per-position gripper override on GR00T chunk | Seeds 0/1/2: +9/0/0pp = +3.1pp pooled (noise) |
| 2 | GoalImageSubstrate with chunks | MSE(progress) + BCE(goal_reached) on v10 chunks | Force replan on stall | Broken at inference (v10/GR00T chunk distribution shift) |
| 3 | GoalImageSubstrate without chunks | MSE+BCE, no chunk input | Force replan on stall | Net zero (over-triggers, slows GR00T 3-7x) |
| 4 | GoalImageSubstrate without chunks | Same as #3 | Bail on stall | -58pp (bails at min_steps=100 every sub-task) |
| 5 | GoalImageSubstrate balanced (pos_weight=15) | Class-balanced BCE | Signal-derived early-advance with 30-step env confirmation window + bail on regression | -16pp (signal saturates ~50 steps before env predicate; low-peak failures slip past bail trigger) |
| 6 | ResidualSubstrate joint BC training | MSE(predicted_delta, expert_chunk - GR00T_chunk) capped ±0.05 | Add bounded delta to GR00T's xyz/rpy directly | -100pp at scale=1.0; -33pp on first 7 rollouts at scale=0.2 |
| 7 | Task-conditional retract lookup (precursor to substrate version) | — | Modulate retract steps per next-sub-task identity (tasks 8,9 → 50 steps; others → 20) | -7.4pp pooled 3-seed vs constant retract baseline |
| — | Conditional retract heuristic (NOT substrate) | — | Open gripper + lift for up to 30 env steps after success, early-exit when grip_qpos>0.030 | **+9.2pp pooled 3-seed (real, empirically optimal)** |

## Common Failure Mode

Across all 6 substrate variants, including the principled joint-BC residual design:

GR00T's 16-step chunk represents an integrated trajectory plan. Substrate's per-chunk interventions — whether gating decisions (variants 1-5) or direct continuous corrections (variant 6) — perturb that integration faster than they help. Even semantically-correct corrections accumulate compounding error when applied at every chunk boundary.

For gating variants (1-5): wrong discrete decisions are asymmetrically costly. Substrate's continuous signal carries information (progress regression empirically rises monotonically on successful sub-tasks) but cannot drive discrete advance/bail decisions reliably (signal saturates earlier than env predicate; low-peak failures slip past bail triggers).

For residual variant (6): training achieved 77% loss improvement vs naive zero-prediction baseline (substrate IS learning correct deltas on training data). At inference these correct corrections degrade rollouts because:
- 200 training samples cannot cover all inference robot poses
- Even bounded ±0.01 corrections perturb GR00T's planned trajectory at every chunk
- Compounding error destabilizes trajectories that GR00T alone completes

## The Diagnostic That Mattered

Per-step gripper qpos logging on baseline failures revealed: 7/7 chained-LIBERO L=3 baseline failures had gripper physically stuck closed at sub-task boundary; in 4/7 GR00T was actively sending "open" commands but fingers were wedged on the previously-grasped object. The dominant failure mode at L=3 is mechanical handoff, not goal-tracking drift.

The conditional retract heuristic addresses this specific mechanical failure: between sub-tasks, send open-gripper + small lift actions for up to 30 env steps, early-exit when env reports gripper_qpos > 0.030 (released). 7 lines of code. +9.2pp pooled across 3 seeds (78%→87% chain 0, 74%→80% chain 1, 81%→93% chain 2 = pooled 77.8%→87.0%). At L=5 unconstrained the retract regresses (-3pp) — different failure mode at longer chains.

## What This Tells Us About the Goal

The user's framing "substrate guides GR00T thru long-horizon" requires substrate's signal to improve task completion at inference. With GR00T frozen, the only available surfaces are:
1. Override GR00T's chunk (gripper or full action) — substrate variants 1, 6
2. Modulate replanning cadence — variant 3
3. Trigger early sub-task transitions — variants 4, 5
4. Modify GR00T's language input per sub-task — UNTESTED but same asymmetric-cost class

None of (1)-(4) worked, and the diagnostic shows why:

GR00T integrates a coherent trajectory plan from each chunk. Externally adding to or gating that integration introduces destabilizing perturbations that accumulate across the long horizon. The substrate would need to either:
- Replace part of GR00T's processing (requires architectural modification + retraining)
- Be co-trained with GR00T from the start (joint training, RL or BC)
- Operate at a granularity where GR00T's plan can absorb the perturbation (sub-task-boundary only, not per-chunk)

All three options are out of scope for the current setup (frozen GR00T + inference-time substrate overlay).

## What This Doesn't Tell Us

Whether the substrate-as-overlay would help on CALVIN long-horizon (where goal-drift over many steps may be the actual dominant failure mode, vs LIBERO's mechanical handoff failure). CALVIN was investigated:
- Env installed, smoke verified
- GR00T-LIBERO zero-shot probe: 0/8 CALVIN tasks succeed; eef_motion 0.04-0.22m (model attempts tasks but can't complete)
- Finetune required: ~2-4 days (download 166GB dataset + write CALVIN→LeRobot v2 converter + GR00T finetune 12-24hr)

Given the consistent 6-variant pattern, finetune-and-retest on CALVIN is unlikely to break the architectural constraint. The failure mode (substrate's per-step intervention disrupts integrated trajectory plan) would apply equally to a CALVIN-finetuned GR00T.

## The Real Path Forward

If substrate-guided GR00T is to work, the substrate must:
1. **Be jointly trained with the actor's task objective end-to-end** (e.g., substrate as low-rank adapter on GR00T's chunk-output layer, finetuned via BC on full trajectories). This makes substrate's interventions trained-for-the-action, not separately-trained-then-controllerized.
2. **OR operate at coarser-than-chunk granularity** where GR00T's integration can absorb the perturbation (e.g., substrate decides language paraphrase at sub-task start, GR00T runs unmodified within sub-task).

Option (1) is the principled solution. Estimated effort: 1-2 weeks for adapter design, joint BC training infrastructure, and eval.

Option (2) is the cheap test. Estimated effort: 1-2 days for paraphrase-selection substrate + training data collection.

## Files

- `subprojects/liquid-arc/research/self_org_sim/rollout_chained_libero.py` — chain eval harness with all 6 substrate variants wired
- `subprojects/liquid-arc/research/self_org_sim/substrate_goal_tracker.py` — variant #1 model
- `subprojects/liquid-arc/research/self_org_sim/goal_image_substrate.py` — variants #2-5 model
- `subprojects/liquid-arc/research/self_org_sim/goal_image_residual_substrate.py` — variant #6 model
- `subprojects/liquid-arc/research/self_org_sim/train_goal_image_substrate.py` — variants #2-5 trainer
- `subprojects/liquid-arc/research/self_org_sim/train_residual_substrate.py` — variant #6 trainer
- `subprojects/liquid-arc/research/self_org_sim/collect_groot_chunks.py` — pre-collects GR00T chunks for residual training
- `subprojects/liquid-arc/research/self_org_sim/extract_goal_features.py` — DINOv2 averaged end-frame features per task
- `subprojects/liquid-arc/research/self_org_sim/rollout_calvin_zeroshot.py` — CALVIN zero-shot probe (0/8 success, needs finetune)
