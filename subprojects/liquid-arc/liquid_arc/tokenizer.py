"""Mind's own tokenizer and trainable token embedding table.

The Mind perceives text through its own learned representations,
not through an external encoder. The embedding table is trainable
through the Mind's prediction error signal.

Uses a standard tokenizer (HuggingFace) — the vocabulary doesn't
need to match any specific LLM. The Mind's embedding table is its own.
"""

import torch
import torch.nn as nn
from typing import Tuple


class MindTokenizer(nn.Module):
    """Tokenizer + learned embedding table for the Mind."""

    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 32000,
        max_tokens: int = 64,
        tokenizer_path: str = None,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.token_pos_embed = nn.Embedding(max_tokens, d_model)
        self.norm = nn.LayerNorm(d_model)

        self._tokenizer = None
        self._tokenizer_path = tokenizer_path

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return

        try:
            from transformers import AutoTokenizer
            path = self._tokenizer_path or "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
            self._tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True,
            )
            # Update vocab_size to match actual tokenizer
            actual_vocab = self._tokenizer.vocab_size
            if actual_vocab > self.vocab_size:
                # Resize embedding if tokenizer has more tokens
                old_embed = self.token_embed
                self.token_embed = nn.Embedding(
                    actual_vocab, self.d_model, padding_idx=self.pad_token_id
                ).to(old_embed.weight.device)
                # Copy old weights
                n = min(old_embed.num_embeddings, actual_vocab)
                self.token_embed.weight.data[:n] = old_embed.weight.data[:n]
                self.vocab_size = actual_vocab
            print(f"  MindTokenizer: loaded tokenizer (vocab={actual_vocab})")
        except Exception as e:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
                print(f"  MindTokenizer: GPT-2 fallback (vocab={self._tokenizer.vocab_size})")
            except Exception as e2:
                print(f"  MindTokenizer: no tokenizer available ({e}, {e2})")
                self._tokenizer = None

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text into token IDs. Returns [max_tokens] (padded)."""
        self._load_tokenizer()

        if self._tokenizer is None:
            # Emergency fallback: character-level
            tokens = [ord(c) % self.vocab_size for c in text[:self.max_tokens]]
        else:
            tokens = self._tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_tokens,
            )

        if len(tokens) < self.max_tokens:
            tokens = tokens + [self.pad_token_id] * (self.max_tokens - len(tokens))

        return torch.tensor(tokens[:self.max_tokens], dtype=torch.long)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed token IDs.

        Args:
            token_ids: [B, T]

        Returns:
            embeddings: [B, T, d_model]
            mask: [B, T] (True = real token)
        """
        B, T = token_ids.shape
        mask = token_ids != self.pad_token_id
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(token_ids) + self.token_pos_embed(positions)
        return self.norm(h), mask
