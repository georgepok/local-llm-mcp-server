# Neuroplastic Project Status

**Last updated:** 2026-03-10 by Claude Code
**Current phase:** Phase 2 — Experiments 001-003 complete, in-memory modification system deployed

## Phase 0 — Cartography: COMPLETE

All deliverables in `phase0_cartography/`. Architecture verified, weight baseline collected, vLLM reconfigured (0.4 GPU util, ~68GB free).

## Phase 1 — Self-Model Construction: COMPLETE

All deliverables in `phase1_artifacts/`. Detailed results in `shared/outbox/phase1/PHASE1_RESULTS.md`.

## Phase 2 — Self-Modification Experiments: IN PROGRESS

### Experiment 001: Gate Weight Amplification — No Effect
- Uniform ×1.1 scaling on layer 45 MoE gate: 83.3% → 83.3%
- See `shared/outbox/phase2/EXPERIMENT_001_RESULTS.md`

### Experiment 002: Asymmetric Gate Scaling — No Effect
- Rank-based 0.8–1.2 scaling across layers 43, 45, 47, 49: 83.3% → 83.3%
- MoE routing is insensitive to gate weight perturbations (expert homogeneity)

### Experiment 003: Mamba A_log — FIRST IMPROVEMENT
- **003b (A_log -0.5):** 83.3% → **100%** — state tracking and self-prediction fixed
- Faster SSM decay in deep layer 50 improves responsiveness to recent tokens
- See `shared/outbox/phase2/EXPERIMENTS_002_003_AND_INMEMORY_RESULTS.md`

### In-Memory Modification System — DEPLOYED
- vLLM plugin with HTTP endpoints at `/neuroplastic/*`
- **30ms modify** (vs 3 min disk restart) — 6000x speedup
- Checkpoint/restore, inspect, list all operational
- Container running with neuroplastic entrypoint on Spark
- See full details in `EXPERIMENTS_002_003_AND_INMEMORY_RESULTS.md`

### Awaiting Direction
1. Follow-up experiments on A_log (gradient search, multi-layer)
2. Nemotron self-assessment of 003b result
3. Automated experiment loops (now feasible with 30ms cycles)
