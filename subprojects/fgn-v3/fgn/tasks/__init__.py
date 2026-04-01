"""Multi-task generator package for FGN experiments."""

from .temporal import TemporalReasoningTask
from .pattern_search import PatternSearchTask
from .interleaved import InterleavedTask
from .multihop import MultiHopTask
from .compound import CompoundReasoningTask
from .arithmetic import ArithmeticChainTask
from .parity import ParityTask
from .state_tracking import StateTrackingTask
from .permutation import PermutationTask
from .affine import AffineGroupTask
from .random_dfa import RandomDFATask
from .gridworld import GridWorldTask
from .continuous_gridworld import ContinuousGridWorldTask
from .arc import ARCTask

TASK_REGISTRY = {
    # Phase 1b tasks (A-F)
    "A": TemporalReasoningTask,
    "B": PatternSearchTask,
    "C": InterleavedTask,
    "D": MultiHopTask,
    "E": CompoundReasoningTask,
    "F": ArithmeticChainTask,
    # Phase 2 tasks (P, S, G, H)
    "P": ParityTask,
    "S": StateTrackingTask,
    "G": PermutationTask,
    "H": AffineGroupTask,
    # v4 tasks
    "R": RandomDFATask,
    "W": GridWorldTask,
    # FluidNet tasks
    "CW": ContinuousGridWorldTask,
    # ARC tasks
    "ARC": ARCTask,
}

def get_task(name: str, tokenizer, seq_len: int = 512, **kwargs):
    """Get a task generator by name."""
    return TASK_REGISTRY[name](tokenizer, seq_len, **kwargs)
