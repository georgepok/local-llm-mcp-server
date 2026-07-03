I have comprehensive coverage now. Let me deliver the catalog.

---

# LiquidARC Research Program — Complete Findings Catalog

Citations refer to absolute paths under `/Users/George/Documents/GitHub/local-llm-mcp-server/` (`la/` = `subprojects/liquid-arc/`, `fgn/` = `subprojects/fgn-v3/`, `mem/` = `/Users/George/.claude/projects/-Users-George-Documents-GitHub-local-llm-mcp-server/memory/`).

## 1. SUBSTRATE / ARCHITECTURE FINDINGS

**1.1 ContinuousDynamics — single weight-tied module applied 16× via Euler ODE.** One shared dynamics module is the entire computation; emergence comes from iteration, not parameter count. 572K params at d=256 (embedding+context ~213K, dynamics ~356K, head 3K). Source: `la/liquid_arc/dynamics.py:26-32`, `la/liquid_arc/model.py:1-10`, MEMORY.md §"LiquidARC — Clean-Slate".

**1.2 SDPA-factored heat kernel.** `K = softmax(-D²/(4t))` factored as SDPA via row-invariance trick: `K = softmax(q·k/(2t) - ||k_j||²/(4t))` where `q=k=h·√g`. N×N matrix never materialized to HBM; stays in SRAM via FlashAttention. Validated: 18,000 tok/s vs 380 chunked Euler vs 700 DEQ — 47× speedup. `la/liquid_arc/dynamics.py:580-615`, MEMORY.md §"SDPA heat kernel (current)".

**1.3 MetricNet — learned diagonal Riemannian metric.** `[LN(h)||ctx] → Linear(2d, d_metric_bn) → GELU → Linear(d_metric_bn, d) → Softplus`. Init `bias = log(e-1)` → `Softplus → ~1.0` (identity metric at init). `la/liquid_arc/dynamics.py:54-69`.

**1.4 Optional low-rank metric overlay (fluid metric).** `g = diag(D) + L·L^T` with `metric_rank > 0`. Small random init `std=0.001` (bilinear form has zero gradient at zero — cannot be zero-init). `la/liquid_arc/dynamics.py:70-75`. Status: validated config available, used in fluid metric architecture commit (`9be961a`).

**1.5 TauNet — per-position adaptive time constant.** `Linear(d, d_metric) → GELU → Linear(d_metric, 1) → sigmoid → tau ∈ [tau_min, tau_max]`. Bias init `softplus⁻¹(1.0)` → initial τ ≈ 1.0 + tau_min. `la/liquid_arc/dynamics.py:95-100`.

**1.6 LTC contraction.** `dh/dt = -(1/τ)(h - target) + FFN(h)/n_ode_steps`. The 1/τ contraction guarantees spectral radius < 1 — enables DEQ solver. `la/liquid_arc/dynamics.py:780, 788`, `la/liquid_arc/solver.py:347` (DEQ docstring).

**1.7 Target as zero-init residual.** `target = h + W_o(routed_v)`. W_o starts normal-init (not zero — the residual `h + update` already prevents signal destruction; non-zero W_o breaks copy symmetry so positions get different perturbations). `la/liquid_arc/dynamics.py:289-291, 658, 688`.

**1.8 Identity sidechain alpha.** `alpha_logit_init = 2.2` → `sigmoid(2.2) ≈ 0.90` self-attention at init. `la/liquid_arc/dynamics.py:251`, `la/liquid_arc/config.py:57`. Note: in current code path, alpha is held in `alpha_logit` but the literal `alpha*V + (1-alpha)*SDPA` mix described in MEMORY.md is NOT visible in the current `forward`; the parameter exists as a vestigial knob.

**1.9 ContextPool.** Episode context computed once per forward (mean-pool of embedded prompt features) and supplied via `set_context()` before ODE. MetricNet receives `[LN(h) || context]`. `la/liquid_arc/context_pool.py:1-81`, `la/liquid_arc/dynamics.py:392-396`.

**1.10 Pre-norms for three paths.** `norm_geo` (for MetricNet/QK), `norm_val` (for W_v), `norm_ff` (for FFN). `la/liquid_arc/dynamics.py:44-46`.

**1.11 FFN inside dynamics, amortized.** FFN runs at every ODE step but `/n_ode_steps`; this matches a single-FFN-per-block transformer when integrated over the trajectory. `la/liquid_arc/dynamics.py:282-287, 788`.

**1.12 ARCEmbedding.** Tokenized cells: color + (x,y) + role + sep_type + grid_id + (optional) demo_pair_id. `la/liquid_arc/embedding.py`. Returns `(input_ids, labels, meta_dict)` with `meta_dict = {colors, xs, ys, roles, sep_mask, sep_types, target_mask, target_labels, grid_ids}`. MEMORY.md §"ARC Task Data Format".

**1.13 OutputHead is 3K params at d=256.** Small head — argues for substrate being the locus of capacity. `la/liquid_arc/model.py`, MEMORY.md §"572K params at d=256".

**1.14 FlatBaselineARC.** Reference 2-block transformer with matched param budget. Used for substrate ablations. `la/liquid_arc/model.py:31-80`.

## 2. SOLVER / ODE INTEGRATION FINDINGS

**2.1 `euler_solve` — standard forward Euler.** `y_{n+1} = y_n + dt · f(t,y_n)`. O(n_steps) memory, fastest forward, torch.compile compatible. Iterates plain `for i in range(n_steps)` — unrolls at trace. `la/liquid_arc/solver.py:21-83`.

**2.2 `euler_solve_chunked` — gradient checkpointing.** `chunk_size=4` blocks. O(n_steps/chunk_size) memory, ~3× compute (forward+recompute+backward). torch.compile compatible (each chunk is static). `la/liquid_arc/solver.py:303-329`.

**2.3 `euler_solve_halting` — per-position ACT halting (Tier 3, validated).** Each step returns `(dy, p_halt)`. `still_active = ∏_steps(1 - p_halt)` multiplies `dy` so halted positions freeze. `min_steps` clamps halting to 0 for first 4 steps (mandatory min computation). `n_ode_steps` becomes a MAX — easy positions exit early, hard ones use full budget. Returns `(y, ponder_cost, steps_used_per_pos[, sup])`. `la/liquid_arc/solver.py:86-186`, `la/liquid_arc/config.py:159-161`. Validated: halt+SoC at d=128 in drone DR reaches reward 119/ep_len 499 — `mem/project_drone_dr_gotchas.md`.

**2.4 `euler_solve_halting` PonderNet deep supervision mode.** Optional `label_mask` argument captures per-step intermediate state, `p_halt_stack`, `p_active_stack` for ARC label positions. Halt distribution `p_halt_dist[k] = p_active_stack[k] * p_halt_stack[k]`. Residual mass goes to final step. Enables CE_per_step weighted by halt prob — `la/liquid_arc/solver.py:178-186`, `mem/project_cold_start_bootstrap_regime.md`.

**2.5 `euler_solve_with_observer` — passive memory observation.** Observer watches `h.detach()` at each step but NEVER modifies it. Logit correction computed post-hoc via `observer.get_output_correction()`. Structurally immune to copy bias — `la/liquid_arc/solver.py:189-220`, `la/liquid_arc/working_memory.py:1-30`.

**2.6 `euler_solve_with_memory` — memory residual overlay.** Detaches `y` per step, passes through memory, residual added back to MetricNet routing via `fn._metric_overlay`. Detach cuts base-model graph; only memory module gets gradient. NOTE: detach prevents long-tape OOM but also prevents end-to-end gradient. `la/liquid_arc/solver.py:223-271`.

**2.7 `invertible_euler_solve` — O(1) memory via fixed-point reconstruction.** Forward stores only `y_final`. Backward reconstructs intermediates via `y_prev = y - dt·f(t, y_prev)` fixed-point iteration. NOT torch.compile compatible. ~7× compute. Kept as reference; `chunked` preferred. `la/liquid_arc/solver.py:458-550`.

**2.8 `deq_solve` — Deep Equilibrium via IFT.** Forward Euler under `torch.no_grad` (zero tape, fastest). Backward: solve `(I - J_f^T) z = grad` via fixed-point iteration (`n_ift_iters=30`), then single VJP for param grads. ~31 evals vs 80 invertible vs n_steps standard. Requires contractive dynamics (LTC 1/τ guarantees). `la/liquid_arc/solver.py:337-450`.

**2.9 Per-step norm homeostasis (in solver, not dynamics).** `pos_norm > norm_ref`: shrink by `1 - λ(1 - norm_ref/pos_norm)`; below ref: no change. Default `norm_ref=50`, `norm_lambda=0.1`. Applied after each step in `euler_solve` and inside `_euler_chunk_fn`. NOT applied in dynamics-level (would fight stability damping). `la/liquid_arc/solver.py:67-77, 281-298`, `la/liquid_arc/config.py:196-198`.

**2.10 Adaptive stability damping in dynamics.** `damping_factor = stability_threshold / (||dh/dt|| + stability_threshold)`, `threshold=50.0`. Bounds `dh/dt` magnitude. `la/liquid_arc/dynamics.py:791-794`.

**2.11 Progressive damping option.** `damping = 1 - damping_strength * (step/n_steps-1)`. Later ODE steps make smaller updates. Off by default. `la/liquid_arc/dynamics.py:303, 800-803`, `la/liquid_arc/config.py:227-228`.

**2.12 RK4 is OOM.** 4 steps × 4 stages = 16 distance matrices → 100GB+ autograd tape. Euler chosen explicitly to avoid this. MEMORY.md §"LiquidLayer — Continuous-Time ODE".

**2.13 Solver dispatches `reset_fast_weights` and `reset_id_history`.** Per-batch state buffers reinitialized at start of each forward. `_orig_mod` unwrap for compiled module. `la/liquid_arc/solver.py:46-50, 129-134`, `la/liquid_arc/dynamics.py:399-420`.

## 3. CRITICALITY / SoC MECHANISMS

**3.1 `compute_criticality_loss` — drives D²/(4τ) → 18 in log space.** Samples random position pairs, computes geodesic `D² = Σ_k g_k(x)(h_ik-h_jk)²`, takes medians. `ratio_loss = smooth_l1(log(ratio/target_ratio), 0)`. Plus optional D² anchor `0.1*(log D² - log d_sq_target)²` to prevent scale drift. Diagnostics: D²_median, ratio, attn_entropy, entropy_ratio, amp, D_sq_anchor. `la/liquid_arc/sustained_criticality.py:55-164`.

**3.2 `compute_curvature_diversity_loss` — CV band + metric entropy reward.** Soft quadratic hinge: `(max(0, cv_floor - cv))² + (max(0, cv - cv_ceiling))²`. Plus soft histogram entropy reward `-0.1 * entropy/log(n_bins)`. Default band [2.0, 10.0]. `la/liquid_arc/sustained_criticality.py:167-233`, `la/liquid_arc/config.py:207-209`.

**3.3 `compute_tau_quality_loss` — replaces tau_var_loss.** Two components: (a) `smooth_l1(tau_mean, 1.0)` anchors mean to productive ODE range; (b) `(log_tau_std - 0.6)²` encourages ~2× multiplicative spread between positions. Replaces variance-maximizer (which pushes to extremes). `la/liquid_arc/sustained_criticality.py:19-52`, `la/liquid_arc/config.py:216-219`, SUSTAINED_CRITICALITY.md.

**3.4 CV floor/ceiling hinge.** `cv_floor_lambda * (max(0, cv_floor - cv)² + max(0, cv - cv_ceiling)²)`. Default `cv_floor=3.0, ceiling=8.0, λ=0.1`. Used independently from `curvature_diversity_loss`. Critical at d=768 (5M model's metric diverged without it). `la/liquid_arc/config.py:97-100`, MEMORY.md §"5M Width Scaling".

**3.5 τ-CV coupling (structural, not loss).** Inside forward, `coupling_factor = 1 + alpha*(local_cv - cv_target)` clamped to [0.3, 3.0]. τ scaled multiplicatively. `tau_cv_coupling_strength=0.5, target=3.5`. NO gradient through coupling (uses `with torch.no_grad`). `la/liquid_arc/dynamics.py:738-748`, `la/liquid_arc/config.py:211-213`.

**3.6 τ-convergence coupling (structural).** `conv_factor = 1/(1 + β·residual/residual.mean())`. Positions struggling to converge (high `||h-target||`) get faster integration (lower τ). `tau_scale = floor + (1-floor)*conv_factor` modulates within [floor·τ, τ]. Default `β=1.0, floor=0.5`. `la/liquid_arc/dynamics.py:749-760`, `la/liquid_arc/config.py:222-224`.

**3.7 CriticalityController — model-based adaptive real-ARC-mix.** EMA-smoothed CV → target ratio. Zones: sub_critical (<4.5), critical (4.5-6.0), crystallizing (>6.0). Asymmetric alpha: 2× faster when below floor (sub-critical failure is the primary mode). Anticipatory term: incorporates `cv_rate`. `la/liquid_arc/criticality_controller.py:1-354`.

**3.8 LTC residual diagnostic.** `self._last_residual = (h-target).norm(dim=-1).mean(dim=-1)` — the model's own internal surprise. Used for τ-convergence coupling. `la/liquid_arc/dynamics.py:785`.

**3.9 cv·τ product conservation.** `compute_cv_tau_product(cv, tau_mean) = cv * tau_mean`. Joint diagnostic — should stay near constant through transition. Validation experiment hypothesis. `la/liquid_arc/sustained_criticality.py:236-249`, MIND_V2_1_CRITICAL_OBSERVATIONS.md §"cv_tau_product stay constant through transition".

**3.10 D²/4τ target value 18 is the bifurcation point.** Empirical: in distillation transition, the cascade fires when D²/4τ crosses ~18 (step 120-134 cascade in SUSTAINED_CRITICALITY.md introduction). Softmax breaks degeneracy; MetricNet flips amplify→compress. SUSTAINED_CRITICALITY.md "Background" section.

**3.11 Anti-pattern: tau_var_loss maximizes variance blindly.** Pushes τ to extremes — half positions freeze at tau_min, half oscillate. Replaced by `tau_quality_loss`. `la/liquid_arc/config.py:56` (the legacy `tau_var_lambda` still exists), SUSTAINED_CRITICALITY.md §"The problems with naive tau losses".

**3.12 Anti-pattern: Cartpole-style SoC penalty.** "Hinge on raw activations" — the LIBERO Liquid (`distill_groot_flow.py`) uses `soc_penalty` but it's NOT the LiquidARC criticality scaffolding; that's the v11 audit complaint. `la/research/self_org_sim/distill_groot_flow.py` (import `soc_penalty`), MEMORY.md flag.

**3.13 Tau external bias.** `_tau_external_bias` ([N] tensor): per-position τ floor injected by Mind for event-aware integration. Added to `tau_logits` pre-sigmoid. `la/liquid_arc/dynamics.py:335, 721-723`.

## 4. TRAINING REGIME FINDINGS

**4.1 100× geometric LR ratio (geometry distillation key).** MetricNet/TauNet/ContextPool LR is 100× slower than content (W_o, FFN, head) LR. Student reaches 71% eval ARC at step 1000 (vs teacher 54% at step 21K). "Single most impactful discovery." MEMORY.md §"Geometry Distillation".

**4.2 Tau freeze first 5K steps.** `freeze_tau=True` → τ=1.0 (constant) for `tau_freeze_steps=5000`. Plain bool, one recompile at unfreeze. Lets MetricNet specialize before tau adds dynamic complexity. `la/liquid_arc/config.py:58`, `la/liquid_arc/dynamics.py:338, 693-705`, MEMORY.md §"Key architectural fixes".

**4.3 Curriculum warm-up (1K steps, Q/K/V frozen, metric LR 2×).** FGN Phase 1a.1: synthetic copy-pattern at start with Q/K/V frozen and metric_lr_mult=2.0. Forces metric to develop before content gradients flow. `fgn/fgn/config.py`, MEMORY.md §"Phase 1a.1 additions".

**4.4 LR schedule inversion.** Metric LR is 2× during curriculum warm-up, then 0.1× during main training (default `metric_lr_mult=0.1` after warm-up). MEMORY.md §"LR schedule inversion".

**4.5 Anti-memorization procedural stream.** Infinite procedural ARC generator, 13 rules, 3 curriculum stages (`curriculum_stage1_end=20000`, `stage2_end=100000`). Local topology → object cohesion → composition. `la/liquid_arc/config.py:77-81`, MEMORY.md §"Anti-memorization".

**4.6 Temporal invariance — randomized ODE steps [12, 20].** Per-batch random `n_ode_steps ∈ [ode_steps_min, ode_steps_max]`. Forces dynamics to work at multiple integration depths. CRITICAL CAVEAT: at d=768 this causes 30-60 min recompilation stalls — must fix to 16 at large d. `la/liquid_arc/config.py:18-20`, MEMORY.md §"ODE steps fixed to 16 (randomization caused...)". The user flagged this finding as one I dropped.

**4.7 30% real ARC mixing.** `real_arc_mix_ratio=0.30` (default 0 in config). Fixes procedural overfit; baseline xform 40-55% vs V1's 10-20% — 3× improvement. `la/liquid_arc/config.py:103`, MEMORY.md §"TTT V2".

**4.8 Cold-start bootstrap regime.** When ANY TWO of {sparse labels, n_ode_steps>20, halting, d≥768}: enable ReZero+PonderNet deep sup+KL prior+metric_bias_init_std=0.5 from step 0. ARC alone never needed this (dense labels, d=256). `mem/project_cold_start_bootstrap_regime.md`.

**4.9 Multi-task rescues compositional length-gen.** FGN single-task: 54% @ 120 ops. FGN+H+S joint: 99% @ 120 ops at half compute. First clean substrate × process interaction finding. `mem/project_substrate_process_rescue.md`.

**4.10 Transform-weighted loss.** 5× weight on changed cells, 0.05× on unchanged. `transform_weight=5.0, copy_weight=0.05`. Inverted grid weighting breaks copy bias. `la/liquid_arc/config.py:54-55`, MEMORY.md §"Transform-weighted loss".

**4.11 Resonance hypothesis (NOT REPRODUCIBLE).** Phase transition depended on natural frequency of optimizer state × forcing rhythm of 30% ARC mix matching. Two original successes; clean reproduction FAILED in March 2026. **All post-transition checkpoints are irreplaceable artifacts.** `la/PHASE_TRANSITION_FOUNDATION.md:1-50`, `la/RESONANCE_HYPOTHESIS.md`.

**4.12 Extended training past 15K is harmful (5M model).** Eval CE degraded 1.50→1.89 from step 15K→50K; eval xform flat. Procedural overfit. Best checkpoint is step 10-15K. MEMORY.md §"Extended training (30K-50K) harmful".

**4.13 Resume strips `_orig_mod.` prefix.** `--resume` in train.py strips compiled checkpoint prefix. `la/scripts/train.py`, MEMORY.md §"`--resume` added to train.py".

**4.14 TF32 matmul precision.** `torch.set_float32_matmul_precision('high')` after `import torch` — ~30% perf gain, removes inductor warning. `mem/feedback_tf32_matmul_precision.md`.

**4.15 Aux loss weight staircase.** Step 2-3× per iteration, NOT 10× jumps — 10× to crit_lambda OOMs GB10. `mem/feedback_aux_loss_staircase.md`.

**4.16 Sweet spot ≈ 2 domains for far transfer.** At 5M/10K, 3rd domain helps close-OOD (YAML) but hurts far-OOD (dialogue). Capacity-dilution limits. `mem/project_diversity_limits.md`.

**4.17 Compression correlates INVERSELY with OOD at 5M.** Effective rank ↑ → OOD ↑. Tishby naive compression fails at under-parameterized scale; preserving capacity > compressing. `mem/project_compression_inverse.md`.

**4.18 Data diversity is first-order, architecture second-order.** Dense and chunked M=8 identical held-out eval under multi-domain training. The chunked "win" was single-domain regularization. `mem/project_architecture_vs_regime.md`.

## 5. STABILITY / NUMERICAL FINDINGS

**5.1 Zero-init residual.** W_o was originally zero-init (per MEMORY.md key fix). Current code uses normal-init with `std=0.05` for metric_net_linear2_diag weight, but the `target = h + update` pattern handles signal protection structurally. `la/liquid_arc/dynamics.py:68, 289-291, 688`.

**5.2 Identity sidechain alpha_logit=2.2.** sigmoid(2.2) ≈ 0.90 → near-identity routing at init. Vestigial param but documented as design intent. `la/liquid_arc/dynamics.py:252`.

**5.3 No detach in metric/curvature paths (FGN).** Gradient must flow to all 3 loss terms; detach breaks scale entropy regularization. MEMORY.md §"FGN v3 — NO detach()".

**5.4 Log-space heat kernels.** `softmax(log_K)` for numerical stability. Avoids overflow in `exp(-D²/4t)`. FGN Phase 1a, transferred to LiquidARC SDPA factorization. MEMORY.md §"FGN v3 — Log-space heat kernels".

**5.5 Mask log_K with -inf, NOT D² with inf.** Masking D² with inf causes `-inf/(4t) → NaN` gradient. Always mask the softmax input (log_K). `la/liquid_arc/dynamics.py:598-602`, MEMORY.md §"Gotchas — Never mask D²".

**5.6 No tanh on target.** Removed — LN already scales (in current dynamics, `target = h + W_o(routed_v)` with no tanh). MEMORY.md §"No tanh on target".

**5.7 No `.item()` in forward path (torch.compile).** `.item()` causes graph break. All scalar reads must stay as tensors. Use buffers for step indices, not int attributes. `la/liquid_arc/dynamics.py:37-39, 322-325` (`_current_step_index_buf`), MEMORY.md §"FluidNet v1 Bug fix" and §"FGN v3 torch.compile fix".

**5.8 `model.apply(_init_weights)` zeroes special inits — `reinit_special()` AFTER.** `model.apply` traverses and re-initializes; custom bias inits (TauNet bias, Softplus bias) get clobbered. Always re-apply special init AFTER apply(). MEMORY.md §"FluidNet v1 — Bug fix" and §"Gotchas".

**5.9 No variable-length loops in compiled path.** Fixed ODE step counts for compile stability — variable n_steps triggers 30-60 min recompilation at d=768. `la/liquid_arc/config.py:18-20` (with caveat in MEMORY.md), §4.6 above.

**5.10 `return_efficiency` False in compiled path.** `return_efficiency` flag in `euler_solve` adds tensor accumulator that can break compile. `la/liquid_arc/solver.py:22, 52-53, 81-83`, MEMORY.md §"torch.compile Gotchas".

**5.11 `set_step_embed` uses direct assignment, not `.copy_()`.** In-place buffer ops conflict with autograd version tracking. Each step gets a fresh tensor via direct assignment so the graph stays clean. Same pattern for `set_delta_W_o`. `la/liquid_arc/dynamics.py:426-446, 852-876`.

**5.12 `_delta_W_o` is non-persistent buffer.** Must be set BEFORE every forward pass; not saved in state_dict. `zeros_like()` for None reset (not `zero_()` — illegal on grad-tracking tensor). `la/liquid_arc/dynamics.py:328, 426-445`, MEMORY.md §"torch.compile + buffers".

**5.13 `_current_step_index_buf` and `_current_n_steps_buf` are tensors.** Plain ints become static at trace; tensors stay dynamic. `la/liquid_arc/dynamics.py:321-325`.

**5.14 TRITON_PTXAS_PATH required in all containers.** `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` on GB10/sm_121a; oracle-train container missing it baked. MEMORY.md §"Deployment — Containers & torch.compile".

**5.15 Triton shared memory 101KB limit caps d=768.** At d≥1024 the heat-kernel SDPA kernel exceeds shared mem. Hard ceiling. MEMORY.md §"d=768 is max for torch.compile".

**5.16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.** Required on DGX Spark unified memory. Cap at 85% utilization. MEMORY.md §"Memory management".

**5.17 GB10 SDPA kernel selection.** `torch.backends.cuda.enable_mem_efficient_sdp(False)` + flash_sdp(True) — cutlass mem-efficient kernels not built for sm121. `la/research/self_org_sim/distill_groot_flow.py:46-49`.

**5.18 Stability damping in dynamics.** `damping_factor = 50 / (||dh/dt||+50)` bounds update magnitude. `la/liquid_arc/dynamics.py:791-794`.

## 6. LAYER-WISE FINDINGS (from deleted inbox + project memory)

**6.1 Layer-wise ODE co-processing — one Euler step per LLM layer.** Distributes 36 ODE steps across Qwen3-4B's 36 attention layers. ODE accumulates geometric model of LLM computation as it unfolds through depth. Sensory forcing alpha=0.2 at each layer. LAYER_WISE_ODE_ARCHITECTURE.md.

**6.2 Pure perturbation architecture (the fix).** `h_ode = h_residual + ε·correction`. Correction accumulates additively through depth; D² anchored by residual stream. ε=0.1 default. `la/liquid_arc/layer_wise_ode.py:1-40`, LAYER_WISE_PERTURBATION_FIX.md.

**6.3 Layer-wise D² runaway (negative result).** Without perturbation: D² grows 268 → 56,832 through 36 layers (30,000× compounding amplification). MetricNet amp at pre-transition CV=0.4 amplifies every layer. LAYER_WISE_PHASE1_ANALYSIS.md.

**6.4 Norm anchoring failed; ε perturbation worked.** Norm anchoring in raw space can't fix metric-space D² growth (because g_ij amplifies independently of h norms). Solution was perturbation architecture limiting ODE state to residual_stream + ε·correction. LAYER_WISE_PERTURBATION_FIX.md.

**6.5 ε sweep — at ε=0.1, correction stabilizes at 1.3-2.9% of residual norm.** Structurally correct B_within/B_across separation emerges (B_within +0.13, B_across -0.26 at late depth) but too weak to change generation. LAYER_WISE_PERTURBATION_NEXT_STEPS.md.

**6.6 Layer-wise NTP delta NEGATIVE after E2E CE training.** With CV→22 from E2E CE training, ODE improves NTP consistently after ~300 steps. `mem/project_layerwise_ode.md`.

**6.7 LLM ignores attention bias (negative result).** QK^T scores dominate ±5 bias range. Bias injection alone insufficient. Residual injection destroys generation at any useful strength. `mem/project_layerwise_ode.md` "What Doesn't Work".

**6.8 SDPA doesn't propagate gradients through mask — use eager attention.** For E2E CE training through bias, must set `attn_implementation="eager"` for grad to flow. `mem/project_layerwise_ode.md`.

**6.9 d=2560 ODE matches Qwen3-4B natively; projection destroys geometry.** Tested: projecting between d_ode and d_llm broke routing. `mem/project_layerwise_ode.md`.

**6.10 Layer deltas have better root/intermediate separation than residuals.** cosine 0.04 (deltas) vs 0.90 (raw). But didn't help generation — discriminative signal at INPUT, model treats it as binary presence. `mem/project_layerwise_ode.md`, parallels v7 FiLM content-invariant finding.

## 7. CONTEXT / METRIC ARCHITECTURE FINDINGS (from inbox)

**7.1 Buffer elimination.** Token buffer (512 slots) was 91.5% curriculum noise, 8.5% conversation signal. Bias was meaningless. Replace with: h_state [1, K=32-64, d] persistent + per-call prompt deltas. ODE input = `[h_state ; prompt_deltas]`. h_state updates from ODE output[:K]. BUFFER_ELIMINATION_ARCHITECTURE.md.

**7.2 Context-prompt scaling.** User-message-only deltas (~8× reduction): 20 turns = 600 tokens vs 5000. Sparse bias scatter for user positions. Assistant tokens excluded from delta extraction. CONTEXT_PROMPT_SCALING.md.

**7.3 Revised context architecture — recent window includes assistant.** Correction to 7.2: assistant responses contain causal articulations, corrections, synthesis — structural signal. Window = last W=3-5 turns (user+assistant). h_state carries compressed history of older turns. `ode_window_max_tokens=800` cap. REVISED_CONTEXT_ARCHITECTURE.md.

**7.4 Mamba state capture (proposed, NOT IMPLEMENTED).** Use Nemotron's Mamba-2 hidden state `s_t = A·s_{t-1} + B·x_t` as natural ODE input. Fixed size, natively sequential, already compressed. Replaces delta extraction + buffer + h_state entirely. Three options: vLLM KV Connector / in-process quantized / Mamba-only forward. MAMBA_STATE_CAPTURE_ARCHITECTURE.md.

**7.5 LLM-adapted MetricNet training is the missing piece.** Architecture validated. MetricNet trained on ARC produces CV=0.51 on text deltas. Self-supervised CV reward → CV=3.87 but generic-not-useful structure. E2E CE training → CV=16-22 (useful). METRICNET_LLM_TRAINING.md, LLM_ADAPTED_METRICNET_ASSESSMENT.md.

**7.6 State cosine + displacement bias (replaces MetricNet bias).** Bias from ODE's internal state alignment via cosine similarity. The ODE's routing produces alignment that cosine extracts. Response quality "best the system has produced — clean formatting, no repetition, concise cross-domain connection." STATE_COSINE_BIAS_VALIDATION.md.

**7.7 Live B_across=0.0 (bug).** Live deployment showed B_across=0 on every call. Three candidates: event_id propagation broken, curriculum dilution, autonomous loop de-alignment. Not resolved. STATE_COSINE_BIAS_VALIDATION.md §Issue 1.

**7.8 Curriculum injection during active conversation = HIGH-PRIORITY DAMAGE.** Wikipedia articles consume buffer + ODE state + text prompt slots. Suppress curriculum within 120s of user activity. STATE_COSINE_BIAS_VALIDATION.md §Issue 2.

## 8. WORKING MEMORY / STEP-EVOLVING FINDINGS

**8.1 Step embeddings `step_embeds[20, d_metric_bn]`.** Zero-init parameter, additive after MetricNet GELU (`met_hidden + self._current_step_embed`). Linear interpolation via `set_step_embed`. Direct assignment per step (not `.copy_()`) for autograd graph cleanliness. `la/liquid_arc/dynamics.py:78-83, 525-526, 852-876`.

**8.2 Channel gate replaces TauNet (working memory experiment).** `gate_net_linear1/2 → sigmoid` produces [B, N, d] gate. LTC becomes `dh/dt = -gate * (h - target)` (channel-wise instead of scalar). Bias init 2.0 → sigmoid(2.0)≈0.88 ≈ initial 1/tau. `la/liquid_arc/dynamics.py:86-93, 691-700`, MEMORY.md §"Working Memory + Step-Evolving Metric".

**8.3 WM ablation results.** +152K params (~3% of 5M). step_only / gate_only / combined / baseline. `model.py: gate_dim_std` is key metric. `ttt.py` melts gate_net + step_embeds during TTT. MEMORY.md §"Working Memory + Step-Evolving Metric".

**8.4 Hierarchical tau schedule (init).** `init_hierarchical_tau(tau_fast=0.2, tau_slow=0.9)`. Biases tau_step_embed: spinal reflex steps 0-3 (fast), motor coord 4-10 (medium), cortical 11-15 (slow). Analytical bias computation from sigmoid logit. `la/liquid_arc/dynamics.py:340-390`.

**8.5 `tau_step_embed[20, d_metric]`.** Embedding added to TauNet hidden before linear2. Zero-init for checkpoint compat. Lets TauNet know which ODE step it's in. `la/liquid_arc/dynamics.py:103-105, 710-711`.

**8.6 WorkingMemory V4 — observe during ODE, correct at output.** Memory NEVER modifies h. Correction targets logits, not hidden states. Mean-pool observations lose position-specific info → immune to copy bias. ~42K params (slot_init + write_proj + write_gate + step_embed + read_query + correction_head). `la/liquid_arc/working_memory.py:1-154`.

**8.7 Step-conditional FiLM on MetricNet/TauNet bottlenecks (Tier 1).** γ=1, β=0 init (no-op). Per-step `t_diff_per_step` embedding. Lets each step specialize geometry. `la/liquid_arc/dynamics.py:111-133, 528-533, 712-718`, `la/liquid_arc/config.py:149-152`.

**8.8 Step-conditional FiLM on Q/K (Tier 2).** Same pattern for attention routing. Only when `routing_mode in (attention, coupled)`. `la/liquid_arc/dynamics.py:156-169, 471-479`.

## 9. GEOMETRIC AUXILIARY / SCAFFOLD FINDINGS

**9.1 Geometric loss L_geo — MSE on D² to spatial target.** Phase 1 (0-5K): target = Manhattan distance², CE=0, only MetricNet trains. Phase 2 (5K+): target interpolates manhattan→boundary over 3K steps. CE ramps 0→1. Scaffold permanent unless decayed. `la/liquid_arc/geo_loss.py:1-50`, `la/liquid_arc/config.py:83-95`.

**9.2 D² scaffold decay option (recent commit `09df3c6`).** `geo_lambda_init→geo_lambda_final` over `decay_start→decay_end`. Hypothesis: explicit Manhattan supervision over-constrains metric into grid-shaped patterns that don't generalize → 40-pt train/eval gap on 5M. Tests scaffold removal late (15K→20K). `la/scripts/train.py compute_geo_lambda`, commit `09df3c6` message.

**9.3 Structural τ — input-INDEPENDENT per-position timescale.** Ones-init → sigmoid(1)≈0.73 → s_tau≈2.3 — starts slow, differentiates via training. Modulates dynamic tau as `tau * s_tau` (multiplicative). Range [`structural_tau_min`, `structural_tau_max`]=[0.3, 3.0]. `la/liquid_arc/dynamics.py:185-189, 725-735`, `la/liquid_arc/config.py:249-251`.

**9.4 Structural_τ gradient fix (recent commit `09df3c6`).** Open research problem RESOLVED: `compute_tau(h0)` returned only TauNet output, NOT the structural_tau blend. Loss backward never touched the parameter. Fix: `compute_tau` mirrors forward()'s structural_tau branch. Validation: grad norm 4.6e-4→7.5e-3 (16×) over 5K steps; eval cell_acc 5.3%→37.0%. `la/liquid_arc/dynamics.py:927-980`, commit `09df3c6`.

**9.5 Geometric KL prior (PonderNet).** `KL(halt_dist || Geom(rate=1/16))`, default `λ=0.01`. Prevents halt collapse to step 1. `la/liquid_arc/config.py:170`.

**9.6 Connected-components BFS for Phase 2 boundaries.** CPU BFS in geo_loss for Phase 2 object boundary targets. ~200MB overhead, outside compiled path. MEMORY.md §"Geometric Auxiliary Loss".

**9.7 Geo loss outside compiled ODE.** No torch.compile interaction; `compute_model_attention` uses `torch.bmm` for materialized [B,N,N] (same SDPA factorization, but materialized). MEMORY.md §"Outside compiled ODE loop".

## 10. TTT / META-LEARNING FINDINGS

**10.1 WHERE/WHAT decomposition.** MetricNet = WHERE to route (geometric routing). W_o = WHAT transformation to apply (content transformation). TTT must unfreeze BOTH for content-transform tasks. MEMORY.md §"TTT V2 — WHERE/WHAT decomposition".

**10.2 W_o unfreeze alone: 13.7%→43.7% xform (3.2×).** Dominant TTT V2 factor. Melt set: `metric_net_linear1`, `metric_net_linear2_diag`, `tau_net_linear1`, `tau_net_linear2`, `W_o`. Optionally `ffn[-1]` (`ttt_unfreeze_ffn`). `la/liquid_arc/reptile.py:32-43`, MEMORY.md §"W_o unfreeze".

**10.3 `xform_loss` instead of `ce_loss` in TTT.** Removes copy-cell contamination. xform_loss only on transform-changed cells. MEMORY.md §"TTT fix 1".

**10.4 TTT positive across 20K-50K post-V2 fixes.** No degradation unlike V1. V1 had 6.7%→24.1% xform at step 15K, then degraded. V2 has positive lift across all checkpoints. MEMORY.md §"TTT positive across all checkpoints".

**10.5 TTT becomes destructive post-transition (5M model).** +28.5% lift pre-transition, -25.6% post-transition. TTT compensates for underdeveloped geometry; overwrites trained universal structure. MEMORY.md §"5M Width Scaling — TTT".

**10.6 Reptile meta-learning.** First-order MAML. Snapshot melt params → for K=4 tasks: clone via state_dict, run 50 inner TTT steps, accumulate deltas → meta-update `base += meta_lr * avg_delta`. `meta_lr=0.1`, start at `reptile_start_step=5000` (after geo pre-training), every 5 CE steps (~20% overhead). `la/liquid_arc/reptile.py:1-94`, `la/liquid_arc/config.py:129-138`.

**10.7 State_dict cloning, not deepcopy.** `copy.deepcopy` breaks on torch.compile non-leaf tensors. `la/liquid_arc/reptile.py:13-15`.

**10.8 D4 augmentation during TTT (Experiment B).** Optional flip/rotate augment + color perm. `ttt_d4_augment`, `ttt_d4_color_perm`. `la/liquid_arc/config.py:115-117`.

**10.9 TTT V1 root causes of degradation.** Metric rigidity (CV 7.7→3.5), procedural exhaustion, 1024 context ceiling. MEMORY.md §"TTT V1 root causes".

**10.10 100 TTT steps + 2048 context + CV floor penalty + 30% real ARC = V2 baseline.** MEMORY.md §"TTT V2 fix list".

## 11. HYPERNET / EXTERNAL DELTA FINDINGS

**11.1 Oracle HyperNet predicts W_o deltas.** `z_context [B,768] → mean(dim=0) → adapter(768→256) → LowRankHead(rank=8) → ΔW_o [768,768]`. 226K new params. Reuses `LowRankHead` from `hypernet.py`. MEMORY.md §"Oracle HyperNet".

**11.2 Direct assignment for delta (not `copy_`).** `_delta_W_o = delta` (not `_delta_W_o.copy_(delta)`) preserves autograd graph; `copy_` is illegal on grad-tracking tensor. `la/liquid_arc/dynamics.py:443-445`, MEMORY.md §"set_delta_W_o: direct assignment".

**11.3 `zeros_like` for None reset.** `torch.zeros_like()` for None reset (not `zero_()` — illegal on grad-tracking tensor). `la/liquid_arc/dynamics.py:442-443`.

**11.4 Phase-gated hypernet activation.** Phase 0: delta=None (idle, baseline). Phase 1+: delta from hypernet, CE backprops through. MEMORY.md §"Phase 0: delta=None".

**11.5 LowRankHead.** `ΔW = U @ diag(c) @ V * scale` with `U[d_out, r], V[r, d_in]`. Coefficient predictor: `task_embed → 64 → r`. Learnable log_scale init `log(scale_init=0.01)` for numerical stability. `la/liquid_arc/hypernet.py:24-65`.

**11.6 Hypernet distillation from gradient-based TTT.** Training mode: distill from `gradient TTT target deltas`. `hypernet_distill_lr=1e-3`, `distill_steps=5000`. `la/liquid_arc/config.py:120-126`.

**11.7 Smoke test passed: torch.compile OK, 5500 tok/s, 59% eval xform.** MEMORY.md §"Smoke test passed".

**11.8 Latent oracle similarity distillation FAILED.** Qwen cosine sims encode no useful ARC structure. Negative result. MEMORY.md §"Latent Oracle — Similarity distillation failed".

## 12. PHASE TRANSITION EMPIRICAL OBSERVATIONS

**12.1 Step ~5000-5500 — CV jumps from ~2 to ~6-7 (ARC).** Geometry reorganizes from near-flat to richly curved. Loss collapses 2.30→1.20. Eval xform 15-22% → 42-48%. Learning rate: 16 pp/1K steps during transition, 0.75 pp/1K post. `la/PHASE_TRANSITION_FOUNDATION.md` §"The Transition Event".

**12.2 Task-specific critical attractor.** TSP plateau CV~1.4 (same SOC scaffolding); ARC CV~6-7. D²/4τ=18 satisfied at different (CV, D², τ) configurations. `mem/project_liquid_soc_task_specific.md`.

**12.3 D²/4τ ≈ 18 bifurcation cascade.** Distillation transition cascade: Step 120 D²=427 amp=0.7× (sub-critical) → Step 123 D²=97 amp=0.2× (bifurcation) → Step 134 D²=37 amp=0.1× (post-critical). MetricNet flips amplify→compress. SUSTAINED_CRITICALITY.md §"Background".

**12.4 Post-transition: TTT becomes destructive.** +28.5% lift pre-transition → -25.6% post-transition. TTT compensates for underdeveloped geometry; overwrites the trained universal structure. MEMORY.md §"5M TTT pre/post transition".

**12.5 Phase transition is NOT reproducible.** Two original runs succeeded; clean reproduction with exact recipe failed (March 2026). All post-transition checkpoints are IRREPLACEABLE ARTIFACTS. Practical: future experiments MUST resume from existing post-transition checkpoints. `la/PHASE_TRANSITION_FOUNDATION.md`.

**12.6 Resonance hypothesis explains non-reproducibility.** Optimizer natural frequency × forcing rhythm of 30% ARC mix must match — depends on exact numerical trajectory (seed, compile behavior, CUDA version, code change). Window narrow. `la/RESONANCE_HYPOTHESIS.md`.

**12.7 Cellular automata does NOT trigger transition.** CA satisfiable at CV~3.0. Phase transition is task-contingent — fires on ARC (global spatial patterns) not local-rule tasks. `la/PHASE_TRANSITION_FOUNDATION.md` §"Key Rules".

**12.8 Small auxiliary modules (~40-70K params) on frozen base DO NOT WORK.** Tested 4 memory variants, correction nets, metric overlays — all fail or produce noise. The substrate must be trained jointly. `la/PHASE_TRANSITION_FOUNDATION.md` §"Key Rules — #3".

## 13. NEGATIVE RESULTS / WHAT DOESN'T WORK

**13.1 Liquid LOSES on TSP (FALSIFIED prior claim).** Matched 10K: FGN 0.412, Flat 0.388, Liquid 0.263 non-unreach acc. Liquid under-compresses (rank 150→92 vs FGN/Flat 124→3); for 7-bucket decision, signal is diluted across irrelevant dims. `mem/project_liquid_wins_tsp.md`.

**13.2 Under-compression hurts on low-rank tasks; predicted to help OOD/high-rank.** Anisotropy: Flat 0.78, FGN 0.77, Liquid 0.49. Architectural property, not collapse. `mem/project_liquid_rank_preservation.md`.

**13.3 FGN heat-kernel locality bias BREAKS length-gen.** Flat 100% on 120 ops; FGN 81.9%. FGN |κ| increases 0.04→0.17 with chain length — metric over-curves on OOD. MEMORY.md §"Phase 2 — LENGTH GENERALIZATION".

**13.4 RK4 OOMs from autograd tape.** 4 steps × 4 stages = 16 distance matrices → 100GB+. Euler 1-eval/step is the chosen tradeoff. MEMORY.md §"LiquidLayer".

**13.5 Latent oracle similarity distillation FAILED.** Qwen3.5-9B cosine sims encode no useful ARC structure. MEMORY.md §"Latent Oracle".

**13.6 Single-domain training architecture differences vanish under multi-domain.** Dense ≈ chunked M=8 statistically identical. Earlier "chunked win" was single-domain regularization. `mem/project_architecture_vs_regime.md`.

**13.7 LLM ignores attention bias (layer-wise).** QK^T scores dominate ±5 bias. Residual injection at useful strength destroys generation. `mem/project_layerwise_ode.md`.

**13.8 Norm anchoring in raw space cannot fix metric-space D² runaway.** Clipping h norms does nothing to g-weighted distances. LAYER_WISE_PERTURBATION_FIX.md.

**13.9 Within-chain causal ordering: learnable but numerically unstable.** Gradients 0.5-2500; NaN within 250 steps. `mem/project_layerwise_ode.md`.

**13.10 Geometric coupling text channel = teacher-forcing pathology.** v3 (soft tokens, text removed) → 0%. Soft tokens insufficient channel without generation-based loss. Commit `9c6a07b` message.

**13.11 Hand-coded runtime controllers are anti-pattern.** V8-V11 stacked control logic; V7b (cadence dropout only, pressure-landscape training) was model-natural high-water mark. `mem/feedback_pressure_landscape_design.md`.

**13.12 Long sleep + many curriculum injections during conversation actively damage state cosine.** B_across=0 in live deployment partly attributable to ODE de-alignment from curriculum cycles. STATE_COSINE_BIAS_VALIDATION.md.

**13.13 d=2048 ODE → d=2560 LLM projection destroys geometry.** Must use d_ode == d_llm (match native dimension). `mem/project_layerwise_ode.md`.

**13.14 v7 FiLM is content-invariant.** Trained FiLM weights are LARGE (norm 34.26) but trained-config produces 0.03× content/noise ratio. Model uses FiLM as normalizer, not task content. `mem/project_v7_film_content_invariant.md`.

## 14. LIBERO-SPECIFIC FINDINGS

**14.1 v9 DINOv2 + scaffolding fixes = first multi-suite success.** libero_spatial 0%→47% (every prior variant 0%). Required THREE combined fixes: image flip removed, DINOv2 frozen encoder, rollout passes `pretrained_vision` (was loading only 79/437 tensors). `mem/project_v9_dinov2_breakthrough.md`.

**14.2 v10-DEMO trajectory variety = session record 50.85% avg.** K=4 random demo frames per chunk during adaptive eval. libero_10 53.3%, spatial 70.2%, goal 67%, object 13%. Implementation ~50 lines. `mem/project_v10_demo_replay_record.md`.

**14.3 v10-DEMOTASK task-matched filtering = -8.1% regression.** Narrowing to current task's language: object +4 but spatial -13, libero_10 -16, goal -7. Net 42.75% vs DEMO 50.85%. **Variety IS the signal**; narrowing removes the breadth Liquid's adaptive SGD depends on. `mem/project_v10_demotask_negative.md`.

**14.4 LIBERO Liquid is NOT LiquidARC.** Current `distill_groot_flow.py` is vanilla ODE with cartpole-style soc_penalty. Lacks MetricNet/TauNet/SoC criticality/halting/Reptile/structural_tau/step embeddings. The v11 redesign target. `la/research/self_org_sim/distill_groot_flow.py`, MEMORY.md flag.

**14.5 NO TEXT to Liquid (hard rule).** Liquid student operates from vision+state ONLY. No language, no task_id-as-text-proxy. Animal-level competence framing. Past distillation violated; restart with `--n_tasks 0`. `mem/feedback_no_text_to_liquid.md`.

**14.6 Memory-augmented Liquid (DAgger + kNN bank): 74% libero_10.** GR00T offline distills into both Liquid weights AND episodic memory bank; Liquid retrieves+blends at runtime with adaptive α. Beats every text-conditioned variant. OOD libero_spatial = 0% (teacher itself only 4%). Architecture sound; data quality is the OOD limit. `mem/project_dagger_libero_breakthrough.md`.

**14.7 LIBERO image flip was a 180° rotation bug.** `[::-1,::-1]` mismatched training (no flip). MSE: as-is=160, flipped=6224 (40× worse). Removing one line: v7 22%→47% on libero_10, spatial sim6 0%→67%. THIS was dominant bug; all architectural fixes were downstream. `mem/project_libero_image_flip_bug.md`.

**14.8 Liquid distillation collapses binary gripper.** Flow matching treats binary {-1, +1} as continuous, converges to constant ~-1 (marginal mode). Explains 0% on libero_spatial/object/goal across v4-v8B. Fix: separate sigmoid gripper head + BCE loss. Predicts 30-50% on suites needing actual grasping. `mem/project_gripper_collapse_bug.md`.

**14.9 v10 goal-image conditioning — bimodal.** libero_spatial +6%, libero_10 +7%; libero_object -33%, libero_goal -17%. Net -10%. Goal-image as carrier brittle when visual goals similar (all "obj in basket"). `mem/project_v10_goal_image_mixed_result.md`.

**14.10 System 1/2: V7b frozen / V8 adaptive / Stage 1 pressure-landscape.** V7b (cadence dropout {0,1,3,7}): 64/60/50/56% at K∈{1,4,16,0}. V8 (+per-episode SGD on drift+tau+z_groot_proj): 70/64/52/44% wins K∈{1,4,16}, loses K=0. Stage 1 (V7b architecture, cadence_dropout {0..64} + z_groot_drop_prob=0.2, NO controllers): 58/52/48/60% — K=0 record at 60%. `mem/project_s1s2_k_cliff.md`.

**14.11 z_vl is task-discriminative in input but model treats it as binary presence flag.** Diagnosed in 30 min vs 12+ hours of recipe tweaks. Fix is FiLM modulation, not more training. `mem/feedback_diagnose_before_iterating.md`.

**14.12 Physical signal > learned policies for 1-bit decisions.** S1/S2 cadence: 14 PPO probes peaked at 58%; physics-cadence (cond drift > median, self-calibrating) gave 62% with zero learned params. `mem/feedback_physical_signal_over_RL.md`.

## 15. ROUTING / MULTI-SUBSTRATE / SPARSE / FAST-WEIGHTS / IDENTITY-ROUTING

**15.1 Routing mode: metric / attention / coupled.** `metric`: heat-kernel from learned g. `attention`: asymmetric Q·K inside ODE. `coupled`: both paths run, learned sigmoid gate `W_gate` per-position decides mix (gate=1 attention, gate=0 metric). `la/liquid_arc/dynamics.py:223-235, 467-503, 649-653`, `la/liquid_arc/config.py:29-35`.

**15.2 Sparse activation (Probe 4).** `sparse_fraction > 0` → only top-k positions (by learned activity score) update per step; others hold. `W_activity` bias init 2.0 (sigmoid≈0.88 — most active early). Straight-through estimator for gradient. Biological sparsity hypothesis. `la/liquid_arc/dynamics.py:240-246, 807-829`, `la/liquid_arc/config.py:42-44`.

**15.3 Sparse activation Probe 7 (random ablation).** `sparse_random=True` uses uniformly random top-k instead of learned. Ablation: is learned choice load-bearing? `la/liquid_arc/dynamics.py:810-818`, `la/liquid_arc/config.py:45-46`.

**15.4 Multi-timescale Hebbian fast-weights overlay on W_o.** Per-batch low-rank `F = U·V^T` accumulates `outer(post, pre)` rule WITHOUT gradient. Decays per step. Reset each forward. Compile-safe via FIXED random projections (sketching). `fast_weights_rank=4, eta=0.01, decay=0.05`. `la/liquid_arc/dynamics.py:199-221, 664-687`, `la/liquid_arc/config.py:172-178`.

**15.5 Self-organizing identity routing (validated toy substrate).** Parallel SDPA branch with raw h-similarity (no g scaling) + per-batch EMA across ODE steps. Toy substrate showed similarity-based EMA self-organizes structure to match input topology. `identity_routing_alpha_init=0, decay=0.1`. `la/liquid_arc/dynamics.py:269-280, 626-645`, `la/liquid_arc/config.py:180-187`, commit `9c6a07b`.

**15.6 ReZero gate on dh/dt.** `sigmoid(rezero_gate_logit)`, init `-5.0` → sigmoid≈0.0067. Identity at step 0. Gate grows iff dynamics reduces loss. Required at cold-start bootstrap regime. `la/liquid_arc/dynamics.py:255-263, 832-835`, `la/liquid_arc/config.py:165-166`.

**15.7 Metric bias init std.** `metric_bias_init_std=0.5` (in bootstrap regime) breaks flat-metric fixed point at init — adds Normal(0, std) noise to bias so per-dim metric variance is non-zero from step 0. `la/liquid_arc/dynamics.py:63-67`, `la/liquid_arc/config.py:167`.

**15.8 MultiSubstrateDynamics — K parallel weight-untied dynamics with lateral coupling.** K=1 isolated MSE 0.32 → K=2 coupled 3-step MSE 0.09 (71% reduction). Substrates differentiate (cos_sim=0.38, ablation_spread=1.58). Coupling + multiple inner iterations both required. State: `[B, N, K*d]` concatenated. `la/liquid_arc/multi_substrate.py:1-100`.

**15.9 MicroCircuitWrapper — compress to M=32 slots.** Cross-attention compress → dynamics on M slots → cross-attention expand. Inner loop M² not T² (32² vs 512²). Slot specialization potential. `la/liquid_arc/microcircuit.py:1-100`. Wikitext distill: ppl 12, hard_acc 0.19 — global-latent and summary-routing both dead. `mem/project_chunked_microcircuit.md`.

**15.10 System 2 — EMA of metric weights for multi-timescale routing.** `_ema_w1/b1/w2/b2` from `metric_net_linear1/2`. Momentum 0.995, max alpha 0.4. Blend at step ramp: `alpha = (step_idx/(n_total-1)) * max_alpha`. Slow-metric path runs under `no_grad`. `la/liquid_arc/dynamics.py:313-320, 543-571`, `la/liquid_arc/config.py:234-237`.

**15.11 Metric freeze step.** Cache `g` at `metric_freeze_step`, reuse for later steps. `metric_freeze_after_training_step` gates when freeze becomes active. Plain bool flag — one recompile at flip. `la/liquid_arc/dynamics.py:307-311, 515-541`, `la/liquid_arc/config.py:230-232`.

## 16. GEOMETRIC COUPLING (LiquidARC × Qwen3-4B)

**16.1 GeometricCoupling — vector projections, no tokenization.** `W_inject: d_arc(768) → n_virtual_tokens(8)*d_qwen(2560)` = 31.48M params. `W_read` symmetric. Small init `std=0.01`. `la/liquid_arc/coupling.py:1-50`.

**16.2 58.6% PPL improvement (Run 2, step 2000).** lr=1e-4, state_pred=0.01. NTP loss ~2.7, state pred ~6-11. CV oscillating 0.32-0.75, tau ~0.9-1.2. Random prefix control: 0% improvement. `mem/project_geometric_coupling.md`.

**16.3 ReflectionLimiter — 33% cap.** Hybrid Interface: 3-channel generation (text context + geometric prefix + metadata). MEMORY.md §"Geometric Coupling".

**16.4 CoupledSystem freezes Qwen3 completely.** Only coupling layers (and optionally LiquidARC dynamics at 100× slower LR) trained. Gradient checkpointing default. Persistent ODE state `_h_state` survives across calls. `la/liquid_arc/coupled_system.py:30-60`.

**16.5 Soft-prompt bridge (geodesic experiments).** Text-grounded ODE hint: 99% on adversarial graphs. Soft-prompt v1/v2: 50% (bridge learns partial translation but LLM mixes with text channel). Text removed (v3): 0% (teacher-forcing pathology). Caveat: "ODE" was hand-coded min-plus, not actual ContinuousDynamics. Commit `9c6a07b`.

## 17. DEPLOYMENT / OPERATIONAL FINDINGS

**17.1 Strip `_orig_mod.` prefix from compiled checkpoints.** `.replace("._orig_mod.", ".")` when loading. MEMORY.md §"torch.compile Gotchas".

**17.2 Curriculum injection breaks during active conversation.** STATE_COSINE_BIAS_VALIDATION.md §Issue 2.

**17.3 Bash rejection doesn't kill remote SSH/docker.** Orphan processes on Spark — verify ps + kill explicitly. `mem/feedback_bash_rejection_doesnt_kill_remote.md`.

**17.4 Containers & TRITON_PTXAS_PATH.** `fgn-train` has it baked. `oracle-train` MISSING — must export. ALWAYS set in any container on GB10/sm_121a. MEMORY.md §"Deployment".

**17.5 Never run vLLM serving + Isaac Lab training simultaneously (OOM).** Unified memory. MEMORY.md §"Memory management".

**17.6 Always check memory before launch.** `nvidia-smi + free -h` before every new training job; GB10 unified memory OOMs silently. `mem/feedback_check_memory_before_launch.md`.

**17.7 No extra processes during training.** `mem/feedback_no_extra_processes.md`.

**17.8 Bind mount awareness.** Check mounts before `docker cp/rm`. `mem/feedback_bind_mounts.md`.

**17.9 Default checkpointing infrastructure from v1.** RL/robotics trainers need save+resume from version 1. Never start every test from random init. `mem/feedback_default_checkpointing.md`.

---

## Notable cross-cutting observations

- **Open problem still active even after `09df3c6`**: structural_τ gets gradient now (16× growth) but range only expanded modestly (0.720-0.735 → 0.673-0.735). Differentiation still mild after the gradient fix.
- **Variable ODE depth `[12,20]` IS a validated training mechanism** (temporal invariance) but conflicts with d≥768 torch.compile — config keeps both knobs (`n_ode_steps`, `ode_steps_min/max`) so the user can choose.
- **"Pressure-landscape over runtime controllers"** is the design philosophy MEMORY repeatedly reinforces — design training pressures + minimum specs, let models negotiate interfaces. Multiple anti-patterns documented (V8-V11 controllers, 14 PPO probes for 1-bit decision).
- **Substrate × process interaction is real**: FGN+H+S joint multi-task 99% vs FGN-alone 54% on 120-op compositional. Substrate alone is not enough.
- **Liquid's spectral signature** is rank preservation (150→92 vs Flat 124→3). This is the architectural identity — predictive of OOD/high-rank tasks helping, low-rank classification hurting. Untested but predicted: untying weight-tied 16-step into k-groups should narrow the low-rank gap.
- **Memory-augmented retrieval (74% libero_10)** is the strongest LIBERO result and uses NO text — purely vision+state + kNN over GR00T trajectory bank, adaptive α blend at runtime. The pattern the v11 redesign should generalize from.
