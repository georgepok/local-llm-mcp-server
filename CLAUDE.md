# Claude Code Project Guide

## What This Is

A monorepo combining a production MCP server with six research subprojects exploring geometric neural networks, continuous-time ODE models, and ARC-AGI solving — all targeting NVIDIA DGX Spark (GB10, 128GB unified memory).

## Repository Structure

```
local-llm-mcp-server/
├── src/                    # Production MCP server (TypeScript/Node.js)
├── config.json             # MCP server config (points to DGX Spark vLLM)
├── subprojects/
│   ├── fluid-geometry/     # vLLM logits processor — entropy-driven reasoning control
│   ├── fgn-v3/             # Fluid Geometry Network — Riemannian attention replacement
│   ├── liquid-arc/         # LiquidARC — continuous-time geometric ODE for ARC-AGI
│   ├── latent-oracle/      # Oracle embedding distillation via HyperNet
│   ├── neuroplastic/       # Self-introspecting neural substrate (Nemotron)
│   └── wake-sleep/         # Autonomous wake-sleep training loop
```

## Subproject Relationships

The subprojects form a research progression:

1. **fluid-geometry** — Production plugin. Single-file vLLM LogitsProcessor that adapts temperature and thinking tokens based on entropy curvature. Deployed on DGX Spark.

2. **fgn-v3** — Foundation research. Replaces transformer dot-product attention with diffusion on learned Riemannian manifold. Heat kernel attention via geodesic distances. This is the theoretical basis for LiquidARC.

3. **liquid-arc** — Primary research project. Clean-slate continuous-time ODE model where geometry IS the computation. One shared dynamics module applied 16x via Euler integration. SDPA-factored heat kernel enables FlashAttention. Key results:
   - Phase transition at CV~6.0 produces universal geometric substrate
   - Geometry distillation: 71% eval ARC (vs teacher's 54%), 72% graph coloring
   - 100x geometric LR ratio is the critical training insight

4. **latent-oracle** — Distillation. Uses oracle model (Qwen3.5-9B) embeddings as task descriptors to predict task-specific weight deltas via HyperNet.

5. **neuroplastic** — Self-modification research. Maps and modifies Nemotron-3-Nano architecture. Currently in Phase 0 (cartography).

6. **wake-sleep** — Autonomous learning. Wake-sleep algorithm for self-directed training without external supervision.

## DGX Spark Deployment

**Server:** spark-129a.local (NVIDIA DGX Spark, GB10 Superchip)
- 128GB unified CPU/GPU memory
- CUDA capability 12.1 (sm_121a)
- SSH: `ssh pokazge@spark-129a.local`

**Key containers:**
- `vllm-nemotron-serve` — Production inference (Nemotron-3-Nano-30B-A3B-FP8)
- `fgn-train` — Training container (nvcr.io/nvidia/vllm:26.01-py3), mounts fgn-v3 + liquid-arc
- `liquid-mind` — LiquidARC Mind MCP server on port 8420

**CRITICAL: torch.compile on DGX Spark:**
- Always set `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` in containers
- d=768 is max model width for Triton shared memory (101KB limit)
- ODE steps must be fixed (not randomized) at d=768 to avoid 30-60 min recompilation

**Memory management:**
- Unified memory — cap at 85% utilization
- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Never run vLLM serving + Isaac Lab training simultaneously (OOM)

## Key Technical Concepts

### LiquidARC Architecture
- **ContinuousDynamics**: Single weight-tied module, applied 16x via Euler ODE
- **Heat kernel as SDPA**: K = softmax(-D²/(4t)) factored so N×N matrix stays in SRAM
- **LTC contraction**: dh/dt = -(1/τ)(h - target) where target comes from SDPA routing
- **MetricNet**: Learned Riemannian metric g(h) determines geodesic distances
- **TauNet**: Per-position adaptive time constant τ(h)

### Phase Transition
- At step ~5000-5500, metric CV jumps from ~2 to ~6-7
- The geometry reorganizes from near-flat to richly curved
- This transition is unreliable to reproduce — hence geometry distillation

### Geometry Distillation (Key Discovery)
- Transfer MetricNet/TauNet/ContextPool weights from post-transition teacher
- Train with 100x slower LR for geometric params vs content params
- Student reaches 71% eval ARC at step 1000 (vs teacher's 54% at step 21K)
- The 100x LR ratio is the single most impactful discovery

### Structural Tau (Open Problem)
- Per-position input-independent timescale parameter
- Architecturally present but receives zero effective gradient
- Needs either direct loss, shorter gradient path, or teacher-initialized differentiation

## Building and Running

### MCP Server (main project)
```bash
npm install
npm run build
npm run start:local  # stdio mode for Claude Desktop
```

### LiquidARC Training
```bash
cd subprojects/liquid-arc
# On DGX Spark:
export PYTHONPATH=/home/pokazge/liquid-arc:/home/pokazge/fgn-v3
python scripts/train.py --config configs/liquid_arc_5m.yaml \
    --data_dir /path/to/arc-repo/data --max_steps 50000

# Geometry distillation (v2):
python scripts/train_v2.py --config configs/liquid_arc_v2.yaml \
    --teacher_checkpoint output_30m/checkpoints/step_10000.pt \
    --data_dir /path/to/arc-repo/data --max_steps 10000
```

### FGN-v3
```bash
cd subprojects/fgn-v3
python scripts/validate_minimal.py  # CPU smoke test
python scripts/train.py --config configs/small.yaml --max_steps 10000
```

### Fluid Geometry (deploy to vLLM)
```bash
cd subprojects/fluid-geometry
./deploy.sh  # SCPs to DGX Spark, restarts vLLM with --logits-processors flag
```

## Important Considerations

### torch.compile Gotchas
- No `.item()` in forward path (causes graph break)
- No variable-length loops (fixed ODE steps for compile stability)
- `_orig_mod.` prefix in compiled checkpoint state dicts — strip with `.replace("._orig_mod.", ".")`
- `return_efficiency` in euler_solve must be False during compiled path

### Thinking Model Handling (Nemotron)
- Parkinson's Law of Reasoning: thinking models expand reasoning to fill ANY token budget
- Never return `reasoning_content` as visible content (thinking leak)
- Fix: retry with `chat_template_kwargs: {enable_thinking: false}`

### ARC Task Data Format
- `generate_batch()` returns `(input_ids, labels, meta_dict)` — use `meta_dict` for model forward
- `meta_dict` contains: colors, xs, ys, roles, sep_mask, sep_types, target_mask, target_labels, grid_ids

### Alignment Philosophy (from user)
- Don't constrain model behavior with penalties — create environments where desired behavior is intrinsically rewarding
- Strategic death (in robotics) showed temporal agency, not a bug
- Every constraint (alive bonus, efficiency regularizer) produces learned helplessness
