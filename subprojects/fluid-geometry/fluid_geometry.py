"""
Entropy-Driven Fluid Geometry LogitsProcessor for vLLM v1.

Dynamically switches between 'Curved' (Mamba/Sequential) and 'Flat'
(Attention/Reasoning) modes by monitoring the entropy of the
logits distribution.

Compatible with vLLM 0.13+ v1 LogitsProcessor interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# Configuration constants (can be overridden via environment or config)
# Higher threshold = less likely to trigger thinking (more conservative)
# Typical entropy range: 0-10+, with 1-3 being "certain" and 5+ being "confused"
HIGH_ENTROPY_THRESHOLD = 4.5  # Only trigger thinking when quite uncertain
LOW_ENTROPY_THRESHOLD = 1.5   # Collapse thinking when reasonably confident
GEOMETRY_BIAS = 15.0          # Soft nudge rather than hard switch
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"


class FluidGeometryRequestProcessor:
    """
    Per-request logits processor implementing entropy-driven geometry switching.

    This processor monitors the entropy of the token distribution and:
    - Boosts <think> token probability when entropy is high (confusion)
    - Boosts </think> token probability when entropy is low (resolved)
    """

    def __init__(
        self,
        think_start_id: int,
        think_end_id: int,
        high_threshold: float = HIGH_ENTROPY_THRESHOLD,
        low_threshold: float = LOW_ENTROPY_THRESHOLD,
        bias: float = GEOMETRY_BIAS,
    ):
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.bias = bias

    def _calculate_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute Shannon Entropy: H(x) = -sum(p(x) * log(p(x)))
        """
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.item()

    def _is_thinking(self, tokens: list[int]) -> bool:
        """
        Check if we're inside <think>...</think> by scanning backwards.
        """
        for token in reversed(tokens):
            if token == self.think_end_id:
                return False
            if token == self.think_start_id:
                return True
        return False

    def __call__(
        self,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply geometry switching based on entropy.

        3-argument signature: (prompt_ids, output_ids, logits) -> logits
        """
        # Combine all tokens for state detection
        all_tokens = (prompt_token_ids or []) + output_token_ids
        is_in_thinking = self._is_thinking(all_tokens)

        # Calculate entropy of current distribution
        current_entropy = self._calculate_entropy(logits)

        # Apply geometry switching
        if not is_in_thinking and current_entropy > self.high_threshold:
            # High uncertainty in flow mode -> trigger thinking
            logits[self.think_start_id] += self.bias
        elif is_in_thinking and current_entropy < self.low_threshold:
            # Low uncertainty in thinking mode -> collapse to flow
            logits[self.think_end_id] += self.bias

        return logits


class FluidGeometryLogitsProcessor(AdapterLogitsProcessor):
    """
    vLLM v1 LogitsProcessor that enables entropy-driven geometry switching.

    This wraps FluidGeometryRequestProcessor for per-request processing
    while conforming to the vLLM v1 batch logits processor interface.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ):
        super().__init__(vllm_config, device, is_pin_memory)

        # Load the tokenizer from path
        from transformers import AutoTokenizer

        tokenizer_path = vllm_config.model_config.tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )

        # Get token IDs for think tags
        start_tokens = tokenizer.encode(THINK_START_TOKEN, add_special_tokens=False)
        end_tokens = tokenizer.encode(THINK_END_TOKEN, add_special_tokens=False)

        self.think_start_id = start_tokens[0] if start_tokens else None
        self.think_end_id = end_tokens[0] if end_tokens else None

        if self.think_start_id is None or self.think_end_id is None:
            raise ValueError(
                f"Could not resolve token IDs for {THINK_START_TOKEN} and {THINK_END_TOKEN}. "
                f"Got start_id={self.think_start_id}, end_id={self.think_end_id}. "
                "Ensure the model's tokenizer supports these tokens."
            )

    def is_argmax_invariant(self) -> bool:
        """
        This processor modifies logits based on entropy, so it can
        change the argmax result.
        """
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        """
        Create a new per-request logits processor.

        Returns the FluidGeometryRequestProcessor for all requests.
        """
        return FluidGeometryRequestProcessor(
            think_start_id=self.think_start_id,
            think_end_id=self.think_end_id,
            high_threshold=HIGH_ENTROPY_THRESHOLD,
            low_threshold=LOW_ENTROPY_THRESHOLD,
            bias=GEOMETRY_BIAS,
        )
