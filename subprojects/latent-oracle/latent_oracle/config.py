"""Latent Oracle configuration — extends LiquidARCConfig with distillation fields."""

import sys
from dataclasses import dataclass
from pathlib import Path

# Import base config from liquid-arc (sibling subproject)
_LIQUID_ARC_ROOT = str(Path(__file__).resolve().parent.parent.parent / "liquid-arc")
if _LIQUID_ARC_ROOT not in sys.path:
    sys.path.insert(0, _LIQUID_ARC_ROOT)

from liquid_arc.config import LiquidARCConfig

import yaml


@dataclass
class LatentOracleConfig(LiquidARCConfig):
    """LiquidARC config + oracle distillation fields."""

    # Oracle
    oracle_dim: int = 4096
    oracle_embeddings_path: str = ""

    # Projection head
    proj_d_hidden: int = 1024
    proj_lr: float = 3e-4

    # Kappa distillation (legacy)
    lambda_kappa: float = 0.1
    lambda_kappa_decay: float = 0.01   # Phase 2 reduced weight multiplier
    warmup_steps: int = 2000           # Phase 0: projection head only
    distill_end_step: int = 20000      # Phase 1→2 transition

    # Representation similarity distillation (oracle → heat kernel)
    similarity_path: str = ""          # path to precomputed similarity_matrices.pt
    lambda_distill: float = 1.0        # weight for similarity distillation loss
    distill_ramp_start: int = 0        # step to start ramping distillation
    distill_ramp_end: int = 5000       # step when distill reaches full weight

    # Oracle HyperNet — task-specific W_o deltas from oracle embeddings
    oracle_hypernet_enabled: bool = False
    hypernet_task_dim: int = 256       # adapter bottleneck dimension
    hypernet_rank: int = 8             # rank of low-rank W_o delta
    hypernet_scale_init: float = 0.01  # initial scale for delta magnitude

    # ODE training
    ode_lr: float = 1e-4

    # Training
    batch_size: int = 16
    max_steps: int = 50000
    log_every: int = 50
    eval_every: int = 2500
    save_every: int = 5000
    grad_clip: float = 1.0
    weight_decay: float = 0.01

    @classmethod
    def from_yaml(cls, path: str) -> "LatentOracleConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        # Cast values to declared types (YAML safe_load treats 3e-4 as str)
        typed = {}
        for k, v in data.items():
            if k in cls.__dataclass_fields__:
                expected = cls.__dataclass_fields__[k].type
                if expected is float and isinstance(v, str):
                    v = float(v)
                elif expected is int and isinstance(v, str):
                    v = int(v)
                elif expected is bool and isinstance(v, str):
                    v = v.lower() in ("true", "1", "yes")
                typed[k] = v
        return cls(**typed)
