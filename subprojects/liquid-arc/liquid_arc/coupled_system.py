"""CoupledSystem — LiquidARC × Qwen3-4B geometric integration.

LiquidARC provides persistent curved-space dynamics.
Qwen3-4B provides stateless flat-space knowledge lookup.
The coupling is geometric — learned projections between representation spaces.

Flow:
    1. LiquidARC observes event → updates ODE state h(t)
    2. h(t) → W_inject → virtual prefix tokens in Qwen3's space
    3. Qwen3 processes [prefix + text] → generates output
    4. Qwen3's prefix-position hidden states → W_read → sensory signal
    5. Signal enters LiquidARC as forcing for next ODE integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .config import LiquidARCConfig
from .model import LiquidARCModel
from .context_pool import ContextPool
from .dynamics import ContinuousDynamics
from .solver import euler_solve
from .coupling import GeometricCoupling


class CoupledSystem(nn.Module):
    """Wraps LiquidARC + frozen Qwen3 + trainable coupling.

    Only the coupling layers (and optionally LiquidARC dynamics at 100x slower LR)
    are trained. Qwen3 is completely frozen.
    """

    def __init__(self, arc_model: LiquidARCModel, qwen_model, qwen_tokenizer,
                 coupling: GeometricCoupling, gradient_checkpointing: bool = True):
        super().__init__()
        self.arc = arc_model
        self.qwen = qwen_model
        self.tokenizer = qwen_tokenizer
        self.coupling = coupling
        self.gradient_checkpointing = gradient_checkpointing

        # Freeze Qwen3 completely
        for p in self.qwen.parameters():
            p.requires_grad_(False)
        self.qwen.eval()

        if gradient_checkpointing and hasattr(self.qwen, 'gradient_checkpointing_enable'):
            self.qwen.gradient_checkpointing_enable()

        # Persistent ODE state — the system's continuous identity
        self._h_state = None  # [1, N, d_arc] current ODE state
        self._context = None  # [1, d_arc] pooled context

    def reset_state(self):
        """Reset persistent ODE state."""
        self._h_state = None
        self._context = None

    @torch.no_grad()
    def observe_event_arc(self, text: str, device: torch.device) -> torch.Tensor:
        """Process text through LiquidARC to update ODE state.

        Uses the same text→ODE pipeline as the Mind: tokenize with GPT-2,
        embed through TextEmbedding (if available) or create simple embeddings,
        run through ODE dynamics.

        Returns:
            pooled_state: [d_arc] — pooled ODE state after integrating this event
        """
        # Tokenize with Qwen3's tokenizer for the ODE input
        tokens = self.tokenizer(text, return_tensors='pt', truncation=True,
                                max_length=128).to(device)
        input_ids = tokens['input_ids']  # [1, seq_len]
        B, N = input_ids.shape
        d = self.arc.config.d_model

        # Create simple embeddings for ODE: random init from token IDs
        # (The real system would use trained TextEmbedding, but for coupling
        # training we use a simpler approach that still provides varied input)
        h0 = torch.randn(B, N, d, device=device, dtype=torch.bfloat16) * 0.1

        # If we have previous state, blend it in
        if self._h_state is not None:
            prev_pooled = self._h_state.mean(dim=1, keepdim=True)  # [1, 1, d]
            h0 = h0 + 0.1 * prev_pooled.expand_as(h0)

        # Pool context and run ODE
        raw_model = getattr(self.arc, '_orig_mod', self.arc)
        context = raw_model.context_pool(h0)  # [B, d]
        raw_model.dynamics.set_context(context, mask=None)
        raw_model.dynamics.set_n_steps(raw_model.config.n_ode_steps)

        h_final = euler_solve(raw_model.dynamics, h0,
                              t_span=(0.0, 1.0),
                              n_steps=raw_model.config.n_ode_steps)

        self._h_state = h_final
        self._context = context

        # Pool to single vector
        pooled = h_final.mean(dim=1)  # [1, d] → simple mean pooling
        return pooled.squeeze(0)  # [d]

    def coupled_forward(self, h_arc: torch.Tensor, input_text: str,
                        device: torch.device) -> Dict[str, torch.Tensor]:
        """One step of the coupled system.

        1. LiquidARC's state → virtual prefix tokens
        2. Qwen3 processes input with prefix context
        3. Qwen3's prefix output → sensory signal for LiquidARC

        Args:
            h_arc: LiquidARC pooled state [d_arc]
            input_text: Text to process through Qwen3
            device: torch device

        Returns:
            dict with: logits, arc_signal, prefix_output, input_ids
        """
        # Step 1: Inject LiquidARC state into Qwen3
        prefix_embeds = self.coupling.inject(h_arc)  # [1, n_vt, d_qwen]

        # Step 2: Qwen3 forward with prefix
        tokens = self.tokenizer(input_text, return_tensors='pt', truncation=True,
                                max_length=512, padding=False).to(device)
        input_ids = tokens['input_ids']  # [1, seq_len]

        # Get Qwen3's input embeddings (frozen)
        with torch.no_grad():
            input_embeds = self.qwen.model.embed_tokens(input_ids)  # [1, seq_len, d_qwen]

        # Prepend virtual prefix tokens (these carry gradients through coupling)
        combined_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
        # [1, n_vt + seq_len, d_qwen]

        # Forward through Qwen3 — frozen but prefix_embeds carry gradients
        outputs = self.qwen(
            inputs_embeds=combined_embeds,
            output_hidden_states=True,
            use_cache=False,
        )

        # Step 3: Read Qwen3's response at prefix positions
        last_hidden = outputs.hidden_states[-1]  # [1, n_vt + seq_len, d_qwen]
        n_vt = self.coupling.n_virtual_tokens
        prefix_output = last_hidden[:, :n_vt, :]  # [1, n_vt, d_qwen]

        arc_signal = self.coupling.read(prefix_output)  # [d_arc]

        # Logits for NTP loss (only on text positions, shifted)
        text_logits = outputs.logits[:, n_vt:, :]  # [1, seq_len, vocab_size]

        return {
            'logits': text_logits,
            'arc_signal': arc_signal,
            'prefix_output': prefix_output,
            'input_ids': input_ids,
        }

    def compute_ntp_loss(self, logits: torch.Tensor,
                         input_ids: torch.Tensor) -> torch.Tensor:
        """Next-token prediction loss on text positions.

        Args:
            logits: [1, seq_len, vocab_size] — Qwen3 logits at text positions
            input_ids: [1, seq_len] — input token IDs

        Returns:
            scalar NTP loss
        """
        # Shift: predict token i+1 from position i
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def compute_state_pred_loss(self, arc_signal: torch.Tensor,
                                h_next: torch.Tensor) -> torch.Tensor:
        """State prediction loss — Qwen3's read-back should predict next ARC state.

        Args:
            arc_signal: [d_arc] — signal read back from Qwen3
            h_next: [d_arc] — LiquidARC's actual next state (detached)

        Returns:
            scalar state prediction loss
        """
        return (arc_signal - h_next.detach()).norm()

    @torch.no_grad()
    def baseline_forward(self, input_text: str,
                         device: torch.device) -> torch.Tensor:
        """Qwen3 forward WITHOUT prefix — baseline for perplexity comparison.

        Returns:
            logits: [1, seq_len, vocab_size]
        """
        tokens = self.tokenizer(input_text, return_tensors='pt', truncation=True,
                                max_length=512).to(device)
        outputs = self.qwen(input_ids=tokens['input_ids'])
        return outputs.logits, tokens['input_ids']

    @torch.no_grad()
    def random_prefix_forward(self, input_text: str,
                              device: torch.device) -> torch.Tensor:
        """Qwen3 forward with RANDOM prefix — control for prefix length effect.

        Returns:
            logits: [1, seq_len, vocab_size], input_ids
        """
        tokens = self.tokenizer(input_text, return_tensors='pt', truncation=True,
                                max_length=512).to(device)
        input_ids = tokens['input_ids']

        # Random prefix of same shape as coupling would produce
        n_vt = self.coupling.n_virtual_tokens
        d_qwen = self.coupling.d_qwen
        random_prefix = torch.randn(1, n_vt, d_qwen, device=device,
                                    dtype=torch.bfloat16) * 0.01

        input_embeds = self.qwen.model.embed_tokens(input_ids)
        combined = torch.cat([random_prefix, input_embeds], dim=1)

        outputs = self.qwen(inputs_embeds=combined, use_cache=False)
        text_logits = outputs.logits[:, n_vt:, :]
        return text_logits, input_ids
