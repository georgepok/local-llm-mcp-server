"""ARDecoder — Autoregressive Transformer Decoder for discrete grid generation.

Replaces V1 DreamDecoder CNN with GPT-style transformer that generates output
grids pixel-by-pixel in raster order. Produces exact integer outputs (0-9),
eliminating the "blurry imagination" problem of V1's CNN decoder.

Uses TransformerEncoder (no cross-attention) as a decoder-only architecture.
Rule tokens are embedded directly in the sequence — no separate memory needed.

Sequence format (sequence VQ, L rule tokens):
  [rule_0, rule_1, ..., rule_{L-1}] [input_cells...] [BOS] [output_cells...]

- rule_tokens: Linear(z_q[l]) for each of the L spatial positions — all attend
  bidirectionally to one another, providing compositional rule conditioning
- input cells: color_embed + pos_x + pos_y + phase(0), bidirectional attention
- BOS: special token (color_id=10) marking output start
- output cells: color_embed + pos_x + pos_y + phase(1), causal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ARDecoder(nn.Module):
    """GPT-style autoregressive decoder: rule_token + input -> output pixel-by-pixel.

    Decoder-only architecture using TransformerEncoder (self-attention only).
    Custom mask: input tokens bidirectional, output tokens causal.
    """

    def __init__(
        self,
        z_dim: int = 128,
        d_ar: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        n_colors: int = 10,
        max_grid_cells: int = 900,
        max_grid_size: int = 30,
        dropout: float = 0.1,
        n_rule_tokens: int = 8,
    ):
        super().__init__()
        self.d_ar = d_ar
        self.n_colors = n_colors
        self.max_grid_cells = max_grid_cells
        self.n_rule_tokens = n_rule_tokens  # L — number of spatial rule tokens

        # Rule token projection: z_q per position -> d_ar (applied per-token)
        self.rule_proj = nn.Linear(z_dim, d_ar)

        # Color embedding: 10 colors (0-9) + PAD(10) + BOS(11)
        self.color_embed = nn.Embedding(n_colors + 2, d_ar)

        # Position embeddings (separate x, y for 2D structure)
        self.pos_x_embed = nn.Embedding(max_grid_size, d_ar)
        self.pos_y_embed = nn.Embedding(max_grid_size, d_ar)

        # Phase embedding: 0=input, 1=output
        self.phase_embed = nn.Embedding(2, d_ar)

        # Layer norm before transformer
        self.ln_in = nn.LayerNorm(d_ar)

        # Decoder-only transformer (TransformerEncoder = self-attention only, no cross-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_ar,
            nhead=n_heads,
            dim_feedforward=4 * d_ar,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: predict colors 0-9 only
        self.output_head = nn.Linear(d_ar, n_colors)

        self.BOS_ID = n_colors + 1  # 11

    def _build_sequence(
        self,
        z_q: torch.Tensor,
        input_grid: torch.Tensor,
        target_output: torch.Tensor = None,
    ):
        """Build token sequence for the transformer.

        Args:
            z_q: [B, L, z_dim] sequence of rule embeddings (L spatial positions)
            input_grid: [B, H_in, W_in] int tensor
            target_output: [B, H_out, W_out] int tensor (None for dream mode)

        Returns:
            tokens: [B, seq_len, d_ar] embedded token sequence
            n_input: int, number of non-output tokens (L rule tokens + input cells + BOS)
            H_out, W_out: output grid dimensions
        """
        B = z_q.shape[0]
        L = z_q.shape[1]  # number of spatial rule tokens
        device = z_q.device
        H_in, W_in = input_grid.shape[1], input_grid.shape[2]

        # If target provided, use its dims; otherwise assume same as input
        if target_output is not None:
            H_out, W_out = target_output.shape[1], target_output.shape[2]
        else:
            H_out, W_out = H_in, W_in

        tokens = []

        # 1. Rule tokens: L spatial positions, each projected independently
        # z_q: [B, L, z_dim] -> rule_proj applied to last dim -> [B, L, d_ar]
        rule_toks = self.rule_proj(z_q)  # [B, L, d_ar]
        tokens.append(rule_toks)  # already [B, L, d_ar]

        # 2. Input cells (raster scan: row-major)
        for y in range(H_in):
            for x in range(W_in):
                c = input_grid[:, y, x]  # [B]
                tok = (
                    self.color_embed(c)
                    + self.pos_x_embed(torch.tensor(x, device=device))
                    + self.pos_y_embed(torch.tensor(y, device=device))
                    + self.phase_embed(torch.tensor(0, device=device))
                )
                tokens.append(tok.unsqueeze(1))  # [B, 1, d_ar]

        # 3. BOS token
        bos_id = torch.full((B,), self.BOS_ID, dtype=torch.long, device=device)
        bos_tok = self.color_embed(bos_id)  # [B, d_ar]
        tokens.append(bos_tok.unsqueeze(1))  # [B, 1, d_ar]

        n_input = L + H_in * W_in + 1  # L rule tokens + input cells + BOS

        # 4. Output cells (teacher-forced if target provided)
        if target_output is not None:
            for y in range(H_out):
                for x in range(W_out):
                    c = target_output[:, y, x].clamp(0, self.n_colors - 1)  # [B]
                    tok = (
                        self.color_embed(c)
                        + self.pos_x_embed(torch.tensor(x, device=device))
                        + self.pos_y_embed(torch.tensor(y, device=device))
                        + self.phase_embed(torch.tensor(1, device=device))
                    )
                    tokens.append(tok.unsqueeze(1))

        seq = torch.cat(tokens, dim=1)  # [B, seq_len, d_ar]
        return seq, n_input, H_out, W_out

    def _make_causal_mask(self, seq_len: int, n_input: int, device: torch.device):
        """Build attention mask: input bidirectional, output causal.

        Shape: [seq_len, seq_len], True = masked (cannot attend).
        Positions 0..n_input-1: can attend to all positions 0..n_input-1
        Positions n_input..end: can attend to 0..n_input-1 + self and prior output
        """
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

        # Input tokens attend to all input tokens (bidirectional)
        mask[:n_input, :n_input] = False

        # Output tokens attend to all input tokens
        mask[n_input:, :n_input] = False

        # Output tokens: causal within output block (lower-triangular)
        out_len = seq_len - n_input
        if out_len > 0:
            causal = torch.triu(
                torch.ones(out_len, out_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            mask[n_input:, n_input:] = causal

        return mask

    def forward(self, z_q, input_grid, target_output):
        """Teacher-forced training. Returns logits for output positions only.

        Args:
            z_q: [B, L, z_dim] quantized rule embedding sequence (L spatial positions)
            input_grid: [B, H_in, W_in] int tensor (colors 0-9)
            target_output: [B, H_out, W_out] int tensor (colors 0-9)

        Returns:
            logits: [B, H_out*W_out, n_colors] predictions for output positions
        """
        # Build full sequence: [rule, input..., BOS, out_0, ..., out_{T-1}]
        seq, n_input, H_out, W_out = self._build_sequence(z_q, input_grid, target_output)

        # Logits at BOS predict out[0], at out[0] predict out[1], etc.
        total_len = seq.shape[1]
        n_out = H_out * W_out

        # Apply LN and transformer (self-attention only, no cross-attention)
        seq = self.ln_in(seq)
        mask = self._make_causal_mask(total_len, n_input, seq.device)
        h = self.transformer(seq, mask=mask)

        # Extract logits at output-predicting positions:
        # Position n_input-1 (BOS) predicts out[0]
        # Position n_input (out[0]) predicts out[1]
        # ...
        # Position n_input+n_out-2 (out[T-2]) predicts out[T-1]
        output_positions = h[:, n_input - 1 : n_input - 1 + n_out, :]  # [B, n_out, d_ar]
        logits = self.output_head(output_positions)  # [B, n_out, n_colors]

        return logits

    @torch.no_grad()
    def dream(self, z_q, input_grid, temperature=0.0):
        """Autoregressive generation. Returns crisp integer grid.

        Maintains raw (un-normalized) token sequence and applies ln_in once
        before each transformer call, matching the training-time forward() flow.

        Args:
            z_q: [B, L, z_dim] quantized rule embedding sequence (L spatial positions)
            input_grid: [B, H_in, W_in] int tensor
            temperature: 0.0 = argmax (deterministic), >0 = sampling

        Returns:
            output_grid: [B, H_out, W_out] int tensor (exact integers 0-9)
        """
        B = z_q.shape[0]
        device = z_q.device
        H_out, W_out = input_grid.shape[1], input_grid.shape[2]

        # Build input portion: rule + input cells + BOS (raw, un-normalized)
        seq_raw, n_input, _, _ = self._build_sequence(z_q, input_grid, target_output=None)

        # Generate output tokens one at a time
        generated = []
        for t in range(H_out * W_out):
            total_len = seq_raw.shape[1]
            mask = self._make_causal_mask(total_len, n_input, device)

            # Normalize entire sequence before transformer (matches training)
            seq_normed = self.ln_in(seq_raw)
            h = self.transformer(seq_normed, mask=mask)

            # Predict next token from last position
            logits = self.output_head(h[:, -1, :])  # [B, n_colors]

            if temperature <= 0.0:
                next_color = logits.argmax(dim=-1)  # [B]
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_color = torch.multinomial(probs, 1).squeeze(-1)  # [B]

            generated.append(next_color)

            # Build next token embedding (raw, un-normalized — will be normalized with full seq)
            y = t // W_out
            x = t % W_out
            next_tok = (
                self.color_embed(next_color)
                + self.pos_x_embed(torch.tensor(x, device=device))
                + self.pos_y_embed(torch.tensor(y, device=device))
                + self.phase_embed(torch.tensor(1, device=device))
            )
            seq_raw = torch.cat([seq_raw, next_tok.unsqueeze(1)], dim=1)

        # Reshape to grid
        output_flat = torch.stack(generated, dim=1)  # [B, H*W]
        output_grid = output_flat.reshape(B, H_out, W_out)

        return output_grid


if __name__ == "__main__":
    print("Testing ARDecoder (sequence VQ, L=8 rule tokens)...")

    L = 8  # n_rule_tokens
    dec = ARDecoder(z_dim=128, d_ar=256, n_heads=4, n_layers=4, n_colors=10, n_rule_tokens=L)
    n_params = sum(p.numel() for p in dec.parameters())
    print(f"  Params: {n_params:,}")

    B = 2
    z_q = torch.randn(B, L, 128)  # [B, L, z_dim]
    input_grid = torch.randint(0, 10, (B, 3, 3))
    target_output = torch.randint(0, 10, (B, 3, 3))

    # Teacher-forced forward
    logits = dec(z_q, input_grid, target_output)
    assert logits.shape == (B, 9, 10), f"Got {logits.shape}"
    print(f"  Teacher-forced logits shape: {logits.shape}")

    # Loss computation
    target_flat = target_output.reshape(B, -1)  # [B, 9]
    loss = F.cross_entropy(logits.reshape(-1, 10), target_flat.reshape(-1))
    print(f"  Initial CE loss: {loss.item():.4f}")

    # Loss decreases over training
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    loss0 = loss.item()
    for step in range(30):
        opt.zero_grad()
        logits = dec(z_q, input_grid, target_output)
        loss = F.cross_entropy(logits.reshape(-1, 10), target_flat.reshape(-1))
        loss.backward()
        opt.step()
    print(f"  Loss after 30 steps: {loss0:.4f} -> {loss.item():.4f}")
    assert loss.item() < loss0, "Loss did not decrease"

    # Autoregressive dream
    dream_output = dec.dream(z_q, input_grid, temperature=0.0)
    assert dream_output.shape == (B, 3, 3), f"Got {dream_output.shape}"
    assert dream_output.min() >= 0 and dream_output.max() <= 9, \
        f"Values out of range: [{dream_output.min()}, {dream_output.max()}]"
    print(f"  Dream output shape: {dream_output.shape}")
    print(f"  Dream values range: [{dream_output.min()}, {dream_output.max()}]")

    # Dream with sampling
    dream_sampled = dec.dream(z_q, input_grid, temperature=0.5)
    assert dream_sampled.shape == (B, 3, 3)
    print(f"  Sampled dream shape: {dream_sampled.shape}")

    print("ARDecoder OK")
