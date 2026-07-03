# CALVIN Pipeline State + Handoff

## Goal context
After 7 substrate variants × multiple regimes were comprehensively falsified on chained-LIBERO (see `SUBSTRATE_GUIDED_GROOT_FINDINGS.md`), the user-directed path was to test the goal-tracking substrate on CALVIN (true long-horizon benchmark where goal-drift may be the actual dominant failure mode rather than LIBERO's mechanical-handoff failure mode).

CALVIN finetune requires multi-day setup. This document captures the state and what's needed to complete.

## What works (verified ✓)

1. **CALVIN dataset downloaded**: `/home/pokazge/calvin/dataset/task_D_D/` (168GB, 600K frames training + 60K validation)
2. **CALVIN env runs**: `/home/pokazge/liquid-arc/research/self_org_sim/calvin_smoke.py` passes — EGL on GB10 GPU, env returns rgb_static(200x200), rgb_gripper(84x84), robot_obs(15-d), action(7-d)
3. **GR00T zero-shot probe on CALVIN**: `rollout_calvin_zeroshot.py` — 0/8 success but eef_motion 0.04-0.22m (model attempting tasks, finetune required)
4. **CALVIN→LeRobot v3 converter**: `calvin_to_lerobot.py` works. 1011 validation segments → 60,575 frames in 21 minutes
5. **v3→v2.1 converter**: Isaac-GR00T's `convert_v3_to_v2.py --repo-id calvin_lerobot_validation --root /tmp` produces GR00T-compatible v2.1 at `/tmp/calvin_lerobot_validation/`
6. **LeRobot library installed**: in `/home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/` (Python 3.12)
7. **modality.json**: copied from v3 backup to v2.1 output

## What broke at smoke finetune step

`bash examples/finetune.sh ...` got past:
- Data loading (confirmed 45,410 total steps)
- Model checkpoint shards loaded

Then crashed with `terminate called after throwing an instance of 'std::length_error' what(): vector::reserve` after model load, before first training step.

## Root cause of crash

**Dependency hell from LeRobot install:**

1. LeRobot pulled `torch==2.7.1+cpu` (replacing CUDA torch)
2. Re-installed `torch==2.9.0+cu126` to support GB10 sm_121 (was not in 2.6 supported caps)
3. flash-attn 2.8.3 was built against torch 2.7.1 — ABI mismatch with torch 2.9 likely the crash source

Conflicting versions visible at pip check:
- gr00t needs torch==2.7.1, have 2.9.0+cu126
- gr00t needs flash-attn==2.7.4.post1, have 2.8.3
- gr00t needs numpy==1.26.4, have 2.4.5
- gr00t needs wandb==0.23.0, have 0.21.4
- lerobot needs setuptools <81, have 70.2.0
- lerobot needs torch <2.8.0, have 2.9.0+cu126

## To finish the CALVIN substrate test

### Step 1: Fix venv (2-4 hours)
**Option A**: Fresh venv with pinned versions
```bash
cd /home/pokazge/Isaac-GR00T
uv venv .venv_calvin
source .venv_calvin/bin/activate
# Install GR00T's exact pinned versions FIRST
uv pip install -e .
# Then install lerobot WITHOUT letting it change torch
GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475"
# Manually install lerobot's non-torch deps
uv pip install --no-deps pyarrow jsonlines  # etc.
```

**Option B**: Two separate venvs
- One for converter (use current spark venv with lerobot)
- One for GR00T training (rebuild without lerobot)

### Step 2: Convert training set (~8-12 hours wall)
```bash
cd /home/pokazge/liquid-arc/research/self_org_sim
source <venv with lerobot>
python calvin_to_lerobot.py --split training --max_segments 0 \
    --out_dir /tmp/calvin_lerobot_training
# Then v3→v2
cd /home/pokazge/Isaac-GR00T/scripts/lerobot_conversion
python convert_v3_to_v2.py --repo-id calvin_lerobot_training --root /tmp
cp /tmp/calvin_lerobot_training_v3.0/meta/modality.json /tmp/calvin_lerobot_training/meta/
```

### Step 3: Finetune GR00T on CALVIN (~12-24 hours GPU)
```bash
source <fresh GR00T venv>
export HF_HOME=/home/pokazge/hf_cache
export HF_TOKEN=$(cat /home/pokazge/.cache/huggingface/token)
cd /home/pokazge/Isaac-GR00T
USE_WANDB=0 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=32 bash examples/finetune.sh \
    --base-model-path /home/pokazge/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --dataset-path /tmp/calvin_lerobot_training \
    --embodiment-tag LIBERO_PANDA \
    --output-dir /home/pokazge/checkpoints/GR00T-N1.7-CALVIN \
    --experiment-name calvin_finetune
```

Note: kill the 4 GR00T-LIBERO servers (ports 5555-5558) before training to free GPU memory.

### Step 4: Build chained-CALVIN harness (~4-8 hours coding)
CALVIN has a 5-task chain protocol via `calvin_models/calvin_agent/rollout/rollout_long_horizon.py`. Adapt this to query the new GR00T-CALVIN server instead of CALVIN's own policy. Reuse logic from `rollout_calvin_zeroshot.py` for the GR00T adapter (CALVIN obs → GR00T format).

### Step 5: Test substrate variants on CALVIN
Reuse existing substrate models from this session:
- `substrate_goal_tracker.py` (variant #1)
- `goal_image_substrate.py` (variants #2-5)
- `goal_image_residual_substrate.py` (variant #6)
Or build CALVIN-specific substrate trained on CALVIN expert data.

## Realistic estimated total remaining effort

- Step 1 (venv): 2-4 hours
- Step 2 (full dataset convert): 8-12 hours wall (mostly automated)
- Step 3 (GR00T finetune): 12-24 hours GPU (automated)
- Step 4 (chained-CALVIN harness): 4-8 hours coding
- Step 5 (substrate eval): 2-4 hours per variant

**Total: 3-5 days, mostly long-running automated jobs with checkpoints between.**

## Caveat — likely outcome

The 7-variant comprehensive falsification on chained-LIBERO suggests the substrate-as-inference-controller architecture has a fundamental limit: per-step interventions on a frozen pre-trained actor accumulate error faster than they correct. CALVIN may produce an 8th falsification.

The CALVIN test is worthwhile because:
- Different failure mode profile (goal-drift over many steps vs gripper handoff)
- More training data (60K validation segments)
- Truly long-horizon (1500-step episodes vs LIBERO's 720)

But based on the pattern, the substrate would likely need joint training with GR00T (adapter, not overlay) to actually add value.

## Files

- `subprojects/liquid-arc/research/self_org_sim/calvin_smoke.py` — env validation
- `subprojects/liquid-arc/research/self_org_sim/calvin_to_lerobot.py` — CALVIN→LeRobot v3 converter
- `subprojects/liquid-arc/research/self_org_sim/rollout_calvin_zeroshot.py` — zero-shot CALVIN+GR00T probe
- `subprojects/liquid-arc/research/self_org_sim/SUBSTRATE_GUIDED_GROOT_FINDINGS.md` — comprehensive 7-variant falsification
- `subprojects/liquid-arc/research/self_org_sim/CALVIN_PIPELINE_STATE.md` — this file

## Spark state to preserve

- `/home/pokazge/calvin/` — full CALVIN install + 168GB dataset
- `/tmp/calvin_lerobot_validation/` — converted v2.1 dataset (60K frames, ready for finetune smoke)
- `/tmp/calvin_lerobot_validation_v3.0/` — v3 backup
- `/home/pokazge/Isaac-GR00T/scripts/deployment/spark/.venv/` — current GR00T venv (mangled deps, need rebuild)
- 4 GR00T-LIBERO servers on ports 5555-5558 (kill before finetune)
