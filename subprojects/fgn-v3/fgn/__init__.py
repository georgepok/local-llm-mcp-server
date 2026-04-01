"""FGN v3 — Fluid Geometry Network with Riemannian Attention."""

from .config import FGNConfig
from .metric import MetricNetwork
from .curvature import CurvatureEngine
from .attention import HeatKernelAttention
from .transport import ParallelTransport, HolonomyDetector
from .layer import FGNTransformerLayer
from .model import FGNModel
from .losses import CurvatureRegularization
from .flat_model import FlatTransformerModel

__all__ = [
    "FGNConfig",
    "MetricNetwork",
    "CurvatureEngine",
    "HeatKernelAttention",
    "ParallelTransport",
    "HolonomyDetector",
    "FGNTransformerLayer",
    "FGNModel",
    "CurvatureRegularization",
    "FlatTransformerModel",
]
