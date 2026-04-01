"""FGN v3 configuration."""

from dataclasses import dataclass
from typing import Tuple

import yaml


@dataclass
class FGNConfig:
    # Model dimensions
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    vocab_size: int = 50304
    max_seq_len: int = 2048

    # Metric
    metric_type: str = "diagonal"
    metric_rank: int = 16
    metric_activation: str = "softplus"

    # Heat kernel attention
    n_scales: int = 3
    t_init: Tuple[float, ...] = (0.1, 1.0, 10.0)
    scale_entropy_alpha: float = 0.01

    # Curvature regularization
    curvature_lambda: float = 0.01
    # eta is derived: eta = lambda * correlation_length_init^2
    correlation_length_init: float = 5.0
    # Phase 1a.1: curvature reward (-mu * mean(|kappa|)) encourages geometry
    # Set > 0 to enable reward mode, 0 for legacy variance penalty
    curvature_reward_mu: float = 0.0

    # Metric learning rate
    metric_lr_mult: float = 0.1       # Metric LR multiplier during main training (0.1 = 10x slower)

    # Curriculum training (Phase 1a.1)
    curriculum_steps: int = 0          # Steps of synthetic warmup (0 = disabled)
    curriculum_metric_lr_mult: float = 1.0  # Metric LR multiplier during curriculum
    curriculum_freeze_qkv: bool = False     # Freeze Q/K/V during curriculum phase

    # Model type
    model_type: str = "fgn"          # "fgn" or "flat" — selects model class

    # Dropout
    dropout: float = 0.1

    # Phase 2
    transport_rank: int = 8

    # v4 architecture
    architecture_version: str = "v3"    # "v3" or "v4" — selects layer/model classes
    geo_heads: int = 1                  # number of geometric routing heads (v4 only)
    gate_init: float = 3.0              # initial gate_geo_raw value (v4 only)
    phase0_steps: int = 1000            # geometric pre-training duration (v4 only)
    geo_aux_loss_alpha: float = 0.0     # geometric auxiliary loss weight (0 = disabled)
    geo_metric_type: str = "learned"    # "learned" or "flat" — flat uses g=1 constant (v4 ablation)

    # v5 hierarchical architecture
    escalation_mode: str = "soft"             # "soft" (differentiable), "fixed" (hard threshold)
    escalation_threshold: float = 0.7         # threshold for "fixed" mode
    escalation_sharpness_init: float = 1.0    # initial sharpness for "soft" mode
    escalation_sharpness_final: float = 10.0  # final sharpness after annealing
    escalation_sharpness_steps: int = 5000    # steps to anneal sharpness
    escalation_penalty_alpha: float = 0.01    # penalty weight for high escalation rate
    sparse_attention_mode: str = "dense_masked"  # "dense_masked" or "loop"

    # v6 budget-based architecture
    attention_budgets: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.10, 0.20, 0.30)

    # FluidNet architecture
    d_metric: int = 64           # metric network hidden dim
    d_ffn_fluid: int = 512       # FFN hidden dim (2x d_model, smaller than v6's 1024)
    n_diffusion_iters: int = 1   # diffusion iterations per layer (1=single-shot, >1=iterative)
    diffusion_floor: float = 0.0 # uniform attention floor (0=disabled, e.g. 0.1 = 10% uniform)

    # Resonance parameters
    structural_energy_lambda: float = 0.0     # weight of structural energy loss
    structural_energy_max_pairs: int = 2048   # max context pairs for energy computation
    structural_energy_d_proj: int = 0         # projection dim for structural energy (0=use diagonal metric)
    structural_energy_proj_mlp: bool = False  # use MLP projection (nonlinear) instead of linear

    # Auxiliary distance prediction
    aux_distance_max_hops: int = 0           # max hop classes (0=disabled, e.g. 10)
    aux_distance_weight: float = 0.0         # weight of aux distance loss

    # Curvature floor (keeps geometry active)
    kappa_floor: float = 0.0                 # minimum |κ| target (0=disabled)
    kappa_floor_mu: float = 0.0              # penalty weight for κ below floor

    # v7 sandwich architecture
    sandwich_mode: bool = False
    sandwich_bottom_geo_layers: int = 2
    sandwich_middle_attn_layers: int = 4
    sandwich_top_geo_layers: int = 2
    sandwich_separate_top_metric: bool = True
    sandwich_middle_iters: int = 1          # iterations through middle attn block (1=single pass)
    n_refine_iters: int = 1                  # outer refinement loop (1=single pass, >1=recursive self-refinement)

    # Liquid mode (continuous-time ODE)
    liquid_mode: bool = False           # True = LiquidLayer, False = FluidLayer
    n_ode_steps: int = 4                # RK4 steps per liquid layer
    tau_min: float = 0.1                # minimum time constant
    t_diffusion_init: float = 1.0       # initial diffusion timescale

    # GB10 optimizations
    use_fp8_metric: bool = False
    use_torch_compile: bool = True

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @classmethod
    def from_yaml(cls, path: str) -> "FGNConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        if "t_init" in data:
            data["t_init"] = tuple(data["t_init"])
        if "attention_budgets" in data:
            data["attention_budgets"] = tuple(data["attention_budgets"])
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        data = {k: v for k, v in self.__dict__.items()}
        if "t_init" in data:
            data["t_init"] = list(data["t_init"])
        if "attention_budgets" in data:
            data["attention_budgets"] = list(data["attention_budgets"])
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


if __name__ == "__main__":
    cfg = FGNConfig()
    print(f"d_model={cfg.d_model}, d_head={cfg.d_head}, n_scales={cfg.n_scales}")
    print(f"correlation_length_init={cfg.correlation_length_init}")

    # Verify YAML round-trip
    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), "fgn_test.yaml")
    cfg.to_yaml(path)
    cfg2 = FGNConfig.from_yaml(path)
    assert cfg == cfg2, "YAML round-trip failed"
    os.unlink(path)
    print("Config OK")
