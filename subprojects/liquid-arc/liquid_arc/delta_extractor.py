"""Delta Extractor — converts LLM hidden states into trajectory deltas for LiquidARC.

Feeds LiquidARC the LLM's velocity through meaning-space, not its position.
State deltas naturally cluster: function words → small Δh, content words → medium,
topic shifts → large. This heavy-tailed distribution puts pairs below D²/4τ≈18
without engineering.

Pipeline:
  text → Qwen3 forward (output_hidden_states=True) → extract h_i at middle layer
  → Δh_i = LayerNorm(h_i - h_{i-1}) → project to d_arc → LiquidARC ODE

Per-token granularity: every token becomes an individual ODE position.
The token buffer in Mind manages a sliding window of the most recent 512 tokens.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List


class DeltaExtractor(nn.Module):
    """Extract per-token state deltas from a frozen LLM and project to LiquidARC space.

    Loads the LLM once, runs forward passes to get hidden states,
    computes per-position deltas, projects to d_arc.

    Every token is a separate ODE position — no mean-pooling.
    """

    def __init__(
        self,
        model_path: str,
        d_arc: int,
        extract_layer: Optional[int] = None,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_arc = d_arc
        self.device = device

        # Load frozen LLM
        print(f"  DeltaExtractor: loading {model_path}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device,
        )
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad_(False)

        d_llm = self.llm.config.hidden_size
        n_layers = self.llm.config.num_hidden_layers
        self.extract_layer = extract_layer or (n_layers // 2)
        self.d_llm = d_llm

        print(f"  DeltaExtractor: d_llm={d_llm}, extract_layer={self.extract_layer}/{n_layers}")

        # Projection: d_llm → d_arc (learned linear, or identity if dims match)
        if d_llm != d_arc:
            self.proj = nn.Linear(d_llm, d_arc, bias=False).to(device)
            nn.init.normal_(self.proj.weight, std=0.02)
            print(f"  DeltaExtractor: projection {d_llm} → {d_arc}")
        else:
            self.proj = None

        # LayerNorm for delta normalization (on same device as LLM)
        self.delta_norm = nn.LayerNorm(d_llm).to(device)

    @torch.no_grad()
    def extract(self, text: str, max_tokens: int = 512) -> Dict:
        """Extract per-token state deltas from text.

        Every token becomes an individual ODE position — no mean-pooling.

        Args:
            text: input text string
            max_tokens: max tokens to process

        Returns:
            dict with:
                delta_h: [1, N, d_arc] — normalized per-token state deltas
                token_texts: list[str] — decoded text for each token
                h_norms: [1, N, 1] — per-position hidden state norms
                residuals: [1, N, 1] — ||Δh_i - Δh_{i-1}|| local dynamics quality
                n_tokens: int — number of tokens
        """
        # Tokenize
        inputs = self.tokenizer(
            text, return_tensors='pt', truncation=True,
            max_length=max_tokens, padding=False,
        ).to(self.device)

        # Forward with hidden states
        outputs = self.llm(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [1, N, d_llm] per layer

        # Extract target layer
        h = hidden_states[self.extract_layer]  # [1, N, d_llm]
        N = h.shape[1]

        # Decode token strings for each position
        input_ids = inputs['input_ids'][0]  # [N]
        token_texts = [
            self.tokenizer.decode([tok_id], skip_special_tokens=False)
            for tok_id in input_ids.tolist()
        ]

        # Compute deltas: Δh_i = h_i - h_{i-1}, with Δh_0 = h_0
        h_prev = torch.cat([h[:, :1, :], h[:, :-1, :]], dim=1)  # shift right, pad first
        delta_h = h - h_prev  # [1, N, d_llm]

        # Normalize deltas: mean-subtract per sequence, scale to unit RMS.
        # NOT LayerNorm — that destroys per-token magnitude variation.
        # Function words → small Δh, content words → large Δh. We preserve this.
        delta_h = delta_h.float()
        delta_h = delta_h - delta_h.mean(dim=1, keepdim=True)  # zero-center across tokens
        rms = delta_h.pow(2).mean().sqrt().clamp(min=1e-8)
        delta_h = delta_h / rms  # scale to unit RMS (preserves relative magnitudes)
        delta_h = delta_h.to(h.dtype)

        # Per-position norms (from original hidden states, for scale info)
        h_norms = h.norm(dim=-1, keepdim=True)  # [1, N, 1]

        # Local dynamics quality: ||Δh_i - Δh_{i-1}||
        delta_prev = torch.cat([delta_h[:, :1, :], delta_h[:, :-1, :]], dim=1)
        residuals = (delta_h - delta_prev).norm(dim=-1, keepdim=True)  # [1, N, 1]

        # Project to d_arc
        if self.proj is not None:
            delta_h = self.proj(delta_h.float()).to(delta_h.dtype)

        print(f"  [tokens] extracted {N} tokens from text ({len(text)} chars)")

        return {
            'delta_h': delta_h,        # [1, N, d_arc]
            'token_texts': token_texts, # list[str], len=N
            'h_norms': h_norms,         # [1, N, 1]
            'residuals': residuals,     # [1, N, 1]
            'n_tokens': N,
        }

    @torch.no_grad()
    def extract_and_append(
        self,
        text: str,
        existing_tokens: int,
        max_total: int = 512,
    ) -> Dict:
        """Extract deltas and return positions to append to ODE token buffer.

        Args:
            text: new text to process
            existing_tokens: how many tokens already in ODE buffer
            max_total: max total tokens in ODE (oldest drop off if exceeded)

        Returns:
            delta_h: [1, N_new, d_arc] — new token deltas to append
            token_texts: list[str] — decoded strings for new tokens
            n_new: number of new tokens extracted
            n_drop: number of old tokens to drop from buffer front (if full)
        """
        result = self.extract(text, max_tokens=max_total)
        delta_h = result['delta_h']      # [1, N_new, d_arc]
        token_texts = result['token_texts']
        n_new = result['n_tokens']

        total_after = existing_tokens + n_new
        n_drop = max(0, total_after - max_total)

        print(f"  [tokens] append {n_new} new tokens, drop {n_drop} old "
              f"(existing={existing_tokens}, max={max_total})")

        return {
            'delta_h': delta_h,         # [1, N_new, d_arc]
            'token_texts': token_texts, # list[str], len=N_new
            'n_new': n_new,
            'n_drop': n_drop,
        }
