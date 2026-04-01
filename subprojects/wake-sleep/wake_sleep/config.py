"""Wake-Sleep configuration — extends LiquidARCConfig with WS-specific fields."""

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
class WakeSleepConfig(LiquidARCConfig):
    """LiquidARC config + Wake-Sleep self-distillation fields."""

    # Wake-Sleep self-distillation
    ws_enabled: bool = False
    ws_z_dim: int = 128            # latent concept dimension (information bottleneck)
    ws_d_enc: int = 32             # encoder CNN embedding dim
    ws_d_dec: int = 64             # decoder CNN hidden dim
    ws_wake_steps: int = 100       # wake steps per cycle
    ws_sleep_steps: int = 400      # sleep steps per cycle
    ws_wake_lr: float = 3e-4       # encoder + decoder + z_proj LR
    ws_sleep_lr: float = 1e-4      # ODE + z_proj LR during sleep
    ws_concept_bank_size: int = 10000
    ws_z_noise_std: float = 0.1    # dream perturbation noise (epsilon)
    ws_interp_alpha_min: float = 0.2   # interpolation range for z_dream
    ws_interp_alpha_max: float = 0.8
    ws_wake_only_steps: int = 5000     # pure Wake pre-training before alternation
    ws_dream_grid_min: int = 3         # random grid size range
    ws_dream_grid_max: int = 10
    ws_dream_ttt_steps: int = 40       # TTT steps at inference (30-50)
    ws_dream_ttt_lr: float = 1e-3      # TTT learning rate

    # V2: VQ-VAE
    ws_vq_n_embeddings: int = 64       # codebook size K (shrunk from 512; sequence prevents collapse)
    ws_vq_n_tokens: int = 8            # number of spatial tokens per task (L)
    ws_vq_beta: float = 0.25           # commitment loss weight
    ws_vq_decay: float = 0.99          # EMA decay for codebook
    ws_vq_dead_restart_every: int = 50   # re-init dead codes interval (steps)
    ws_vq_entropy_weight: float = 0.1  # entropy regularization (reduced; sequence naturally prevents collapse)

    # V2: AR Decoder
    ws_ar_d_model: int = 256           # transformer hidden dim
    ws_ar_n_heads: int = 4
    ws_ar_n_layers: int = 4
    ws_ar_dropout: float = 0.1

    # V2: Sleep hybridization
    ws_real_arc_mix_ratio: float = 0.5  # fraction of real ARC in sleep

    # V2: W_o unfreeze (applies to both sleep and dream-TTT)
    ws_unfreeze_wo: bool = True         # melt W_o during sleep phase

    @classmethod
    def from_yaml(cls, path: str) -> "WakeSleepConfig":
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
