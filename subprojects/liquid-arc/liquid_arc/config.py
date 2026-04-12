"""LiquidARC configuration."""

from dataclasses import dataclass

import yaml


@dataclass
class LiquidARCConfig:
    # Model dimensions
    d_model: int = 256
    d_metric: int = 64       # metric bottleneck
    d_ffn: int = 512         # FFN hidden dim (in dynamics)
    max_seq_len: int = 1024

    # ODE integration
    n_ode_steps: int = 16    # Euler steps (weight-tied) — also used as eval default
    ode_steps_min: int = 12  # temporal invariance: sample from [min, max] during training
    ode_steps_max: int = 20
    integration_time: float = 2.0  # ODE integration interval T (dt = T / n_ode_steps)
    tau_min: float = 0.5     # minimum time constant — prevents hyper-viscous collapse
    tau_max: float = 1.0     # hard ceiling via sigmoid — prevents ODE freeze escape
    t_diffusion_init: float = 1.0

    # Fluid metric (low-rank + wider bottleneck)
    d_metric_bottleneck: int = 0  # 0 = use d_metric (backward compat), >0 = override MetricNet bottleneck
    metric_rank: int = 0          # 0 = diagonal only, >0 = diagonal + low-rank L·L^T

    # Geometry
    chunk_size: int = 256    # for chunked distance computation

    # Regularization
    dropout: float = 0.1
    curvature_lambda: float = 0.05  # hard curvature penalty: lambda * |kappa|.mean()
    transform_weight: float = 5.0  # extra weight on cells that changed
    copy_weight: float = 0.05      # near-zero weight on unchanged cells (breaks copy bias)
    tau_var_lambda: float = 0.001  # tau variance maximization weight (Var(tau) across positions)
    alpha_logit_init: float = 2.2  # sigmoid(2.2) ~ 0.90 identity residual in heat kernel
    tau_freeze_steps: int = 5000  # freeze tau=1.0 for this many steps (Fix B)

    # Solver
    use_torch_compile: bool = True
    ode_chunk_size: int = 4  # steps per checkpointed block (memory = n_steps/chunk_size)
    invertible_solver: bool = False  # O(1) memory invertible Euler (incompatible with compile)
    n_fp_iters: int = 5  # fixed-point iterations for invertible solver reconstruction
    deq_solver: bool = False  # DEQ: no_grad forward + IFT backward (fastest)
    deq_ift_iters: int = 30  # fixed-point iterations for IFT in DEQ backward

    # ARC-specific
    n_colors: int = 10
    n_roles: int = 8         # input/output/test_input/test_output/separator etc.
    n_sep_types: int = 4
    max_grid_size: int = 30
    max_grids: int = 16  # max number of grids per sequence (for grid_id_embed)

    # Curriculum
    use_procedural: bool = True  # use procedural generator instead of static ARC
    curriculum_stage1_end: int = 20000   # local topology → object cohesion
    curriculum_stage2_end: int = 100000  # object cohesion → composition

    # Geometric auxiliary loss (L_geo) — MSE on D²
    geo_loss_enabled: bool = False
    geo_cutoff_step: int = 0  # hard cutoff: geo dies at this step (0 = never cut off)
    geo_wall_distance: float = 50.0    # cross-object D² target (Phase 2). e^{-50/4}≈0
    geo_lambda_init: float = 1.0       # initial geo loss weight
    geo_lambda_final: float = 1.0      # PERMANENT scaffold — never decay
    geo_ce_ramp_start: int = 5000      # CE starts ramping from 0
    geo_ce_ramp_end: int = 15000       # CE reaches 1.0
    geo_lambda_decay_start: int = 15000  # (unused — scaffold is permanent)
    geo_lambda_decay_end: int = 20000    # (unused — scaffold is permanent)
    geo_phase2_start: int = 5000       # object boundary supervision starts
    geo_phase2_interp_steps: int = 3000  # steps to interpolate manhattan→boundary targets
    geo_use_h0: bool = True            # supervise at h0 (pre-ODE), not h_final
    geo_detach_h: bool = False         # if True, gradients only flow to MetricNet

    # Metric plasticity — CV floor/ceiling penalty
    cv_floor_target: float = 3.0   # hinge: penalize when CV drops below this
    cv_ceiling_target: float = 8.0 # hinge: penalize when CV rises above this
    cv_floor_lambda: float = 0.1   # weight for CV floor/ceiling hinge loss

    # Real ARC data mixing (training only)
    real_arc_mix_ratio: float = 0.0  # probability of sampling real ARC vs procedural (0 = off)

    # Test-Time Training
    ttt_enabled: bool = False
    ttt_steps: int = 100
    ttt_lr: float = 1e-3
    ttt_curvature_lambda: float = 0.01
    ttt_early_stop_threshold: float = 0.01

    # V3 Experiment A: FFN plasticity — unfreeze FFN[-1] during TTT
    ttt_unfreeze_ffn: bool = False

    # V3 Experiment B: D4 augmentation during TTT
    ttt_d4_augment: bool = False
    ttt_d4_color_perm: bool = False  # also apply random color perms

    # V3 Experiment C: Hypernetwork (amortized TTT)
    hypernet_enabled: bool = False
    hypernet_rank: int = 8
    hypernet_scale_init: float = 0.01
    hypernet_include_ffn: bool = False
    hypernet_training_mode: str = "distillation"  # or "end_to_end"
    hypernet_distill_lr: float = 1e-3
    hypernet_distill_steps: int = 5000

    # Reptile meta-learning (first-order MAML for TTT plasticity)
    reptile_enabled: bool = False
    reptile_start_step: int = 5000       # after geo pre-training
    reptile_every: int = 5               # 1 meta-step per N CE steps (~20% overhead)
    reptile_n_tasks: int = 4             # K tasks per meta-step
    reptile_meta_lr: float = 0.1         # outer LR (scales param deltas, not gradients)
    reptile_warmup_steps: int = 1000     # linear warmup from 0 to meta_lr
    reptile_ttt_steps: int = 50          # inner TTT steps (fewer than eval's 100)
    reptile_ttt_lr: float = 1e-3         # inner TTT learning rate
    reptile_include_ffn: bool = True     # include FFN[-1] in melt set
    reptile_use_train_split: bool = True # sample from ARC train split

    # Working memory / ODE heterogeneity
    step_embed_enabled: bool = False   # step-evolving metric (ODE heterogeneity)
    n_step_embeds: int = 20            # number of learnable step embeddings
    channel_gate_enabled: bool = False # channel-wise gate (working memory, replaces scalar tau)

    # Oracle distillation (representation similarity)
    oracle_distill_enabled: bool = False
    oracle_distill_lambda: float = 1.0       # weight for distillation loss
    oracle_distill_ramp_start: int = 0       # step to start distillation
    oracle_distill_ramp_end: int = 5000      # step when distill reaches full weight
    oracle_similarity_path: str = ""         # path to precomputed similarity matrices

    # Norm homeostasis: soft ODE-level decay prevents h runaway in deployment
    norm_ref: float = 50.0    # per-position L2 reference scale (~sqrt(d))
    norm_lambda: float = 0.1  # restoring force strength above norm_ref

    # Sustained criticality (self-organized phase transition)
    criticality_loss_enabled: bool = False
    criticality_loss_lambda: float = 0.01
    criticality_target_ratio: float = 18.0  # D²/4τ target
    criticality_D_sq_target: float = 60.0   # D² median anchor target

    curvature_diversity_loss_enabled: bool = False
    curvature_diversity_lambda: float = 0.01
    curvature_cv_floor: float = 2.0
    curvature_cv_ceiling: float = 10.0

    tau_cv_coupling_enabled: bool = False
    cv_coupling_target: float = 3.5    # target local CV (near critical)
    cv_coupling_strength: float = 0.5  # coupling strength alpha

    # Tau quality (REPLACES tau_var_loss)
    tau_quality_loss_enabled: bool = False
    tau_quality_lambda: float = 0.05
    tau_mean_target: float = 0.0  # 0 = auto: T / n_ode_steps * 16 (scales with integration params)
    tau_log_spread_target: float = 0.6

    # Tau-convergence coupling (structural, not loss)
    tau_convergence_coupling_enabled: bool = False
    tau_convergence_beta: float = 1.0
    tau_convergence_floor: float = 0.5  # min modulation factor (0.5 = halve tau, 0.3 = aggressive)

    # Progressive damping: later ODE steps make smaller updates
    progressive_damping: bool = False
    damping_strength: float = 0.5  # 0.0 = no damping, 1.0 = last step is zero

    # Metric freeze: stop recomputing routing after this ODE step. -1 = disabled
    metric_freeze_step: int = -1
    metric_freeze_after_training_step: int = 0  # only apply freeze after this training step

    # System 2: multi-timescale routing (EMA metric blending)
    system2_enabled: bool = False
    system2_ema_momentum: float = 0.995
    system2_max_alpha: float = 0.4

    # Cellular automata task (alternative to procedural ARC)
    use_cellular_automata: bool = False
    # Conditional transforms task
    use_conditional_transforms: bool = False
    # Multi-domain ratios (when >0, enables multi-domain sampling)
    procedural_ratio: float = 0.0
    ca_ratio: float = 0.0
    conditional_ratio: float = 0.0

    # Structural tau (v2 geometry distillation)
    structural_tau_enabled: bool = False
    structural_tau_min: float = 0.3
    structural_tau_max: float = 3.0

    # Layer-wise ODE co-processing
    sensory_alpha: float = 0.2     # coupling: residual → ODE per layer step
    bias_lambda: float = 1.0       # attention bias scaling
    persistent_slots: int = 0      # persistent state positions across turns

    # Training config (v2 — used when present in YAML, ignored otherwise)
    base_lr: float = 3e-4
    structural_lr_ratio: float = 0.01
    warmup_steps: int = 500
    weight_decay: float = 0.01
    batch_size: int = 4
    grad_accum_steps: int = 1

    # Model type for flat baseline
    model_type: str = "liquid"  # "liquid" or "flat"

    @classmethod
    def from_yaml(cls, path: str) -> "LiquidARCConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
