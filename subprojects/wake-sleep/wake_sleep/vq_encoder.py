"""VQ-VAE Encoder — CNN encoder + Vector Quantizer with STE + EMA codebook.

Replaces V1 ConceptEncoder with discrete codebook bottleneck.
Forces crisp, discrete concept representations for dream generation.

V2 Sequence VQ: each task encodes to L spatial tokens [B, L, z_dim] rather than
a single [B, z_dim] vector. The 2×4 spatial pool creates 8 positional slots that
capture different regions of the input/output relationship, forming compositional
"sentences" (sequences of codebook words) instead of single "words". This
naturally prevents codebook collapse by distributing assignments across L×B items
per batch instead of just B.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Straight-Through Estimator + EMA codebook updates.

    Codebook updated via exponential moving average (more stable than gradient-based
    for small batch sizes). Dead code restart re-initializes unused codes from batch.
    Entropy regularization prevents codebook collapse.
    """

    def __init__(
        self,
        n_embeddings: int = 512,
        z_dim: int = 128,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        entropy_weight: float = 0.1,
        z_buffer_size: int = 2048,
    ):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.z_dim = z_dim
        self.beta = beta  # commitment loss weight
        self.decay = decay
        self.eps = eps
        self.entropy_weight = entropy_weight

        self.embedding = nn.Embedding(n_embeddings, z_dim)
        # Initialize with larger range — encoder outputs have norm ~10+
        # Small init causes all inputs to map to same nearest entry
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)

        # EMA buffers
        self.register_buffer("cluster_size", torch.zeros(n_embeddings))
        self.register_buffer("embed_avg", self.embedding.weight.data.clone())

        # z_e buffer for dead code restart (accumulate diverse samples)
        self.z_buffer_size = z_buffer_size
        self.register_buffer("z_buffer", torch.zeros(z_buffer_size, z_dim))
        self.register_buffer("z_buffer_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("z_buffer_count", torch.zeros(1, dtype=torch.long))

        # EMA-accumulated soft assignment distribution for long-horizon entropy
        # Solves the B<<K problem: entropy computed over accumulated history,
        # not just current batch. Gradient flows through current batch's contribution.
        self.register_buffer("soft_avg", torch.ones(n_embeddings) / n_embeddings)

    def forward(self, z_e: torch.Tensor):
        """Quantize continuous z_e to nearest codebook entry.

        Args:
            z_e: [B, z_dim] continuous encoder output

        Returns:
            z_q: [B, z_dim] quantized (with STE gradient bypass)
            vq_loss: scalar (commitment + entropy regularization)
            indices: [B] codebook indices
        """
        # Distances: ||z_e - e_j||^2 = ||z_e||^2 - 2*z_e*e_j + ||e_j||^2
        # [B, n_embeddings]
        d = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_e @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = d.argmin(dim=1)  # [B]
        z_q_actual = self.embedding(indices)  # [B, z_dim]

        # EMA codebook update (training only)
        if self.training:
            self._ema_update(z_e, indices)
            self._update_z_buffer(z_e)

        # Commitment loss: encourage encoder output to stay near codebook
        commitment_loss = self.beta * F.mse_loss(z_e, z_q_actual.detach())

        # Entropy regularization: encourage uniform codebook usage
        # Use EMA-accumulated soft assignments to compute long-horizon entropy.
        # This solves B << K: even with B=8, the accumulated distribution
        # reflects hundreds of past assignments, giving meaningful entropy.
        # Gradient flows through the current batch's soft_probs contribution.
        entropy_temp = d.detach().mean().clamp(min=1.0) * 0.1  # adaptive temperature
        soft_probs = F.softmax(-d / entropy_temp, dim=1)  # [B, K]
        batch_avg = soft_probs.mean(dim=0)  # [K] — current batch distribution

        # Mix current batch (differentiable) with EMA history (detached)
        alpha = 0.1  # current batch weight in accumulated distribution
        accumulated = alpha * batch_avg + (1 - alpha) * self.soft_avg.detach()

        # Update EMA buffer for next forward pass
        if self.training:
            with torch.no_grad():
                self.soft_avg.mul_(1 - alpha).add_(batch_avg.detach(), alpha=alpha)

        # Entropy on accumulated distribution
        entropy = -(accumulated * (accumulated + 1e-10).log()).sum()
        max_entropy = torch.tensor(self.n_embeddings, device=z_e.device).float().log()
        entropy_loss = self.entropy_weight * (max_entropy - entropy)

        vq_loss = commitment_loss + entropy_loss

        # STE: gradients bypass quantization
        z_q = z_e + (z_q_actual - z_e).detach()

        return z_q, vq_loss, indices

    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor):
        """EMA codebook update — no gradient needed."""
        with torch.no_grad():
            # One-hot encoding of assignments
            encodings = F.one_hot(indices, self.n_embeddings).float()  # [B, K]

            # Update cluster sizes
            self.cluster_size.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )

            # Update embedding averages
            embed_sum = encodings.t() @ z_e  # [K, z_dim]
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Laplace smoothing to avoid division by zero
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps)
                / (n + self.n_embeddings * self.eps)
                * n
            )

            # Update codebook
            self.embedding.weight.data.copy_(self.embed_avg / cluster_size.unsqueeze(1))

    def _update_z_buffer(self, z_e: torch.Tensor):
        """Accumulate encoder outputs into ring buffer for diverse dead code restart."""
        with torch.no_grad():
            B = z_e.shape[0]
            ptr = self.z_buffer_ptr.item()
            # Write as many as fit from current position
            space = self.z_buffer_size - ptr
            n_write = min(B, space)
            self.z_buffer[ptr:ptr + n_write] = z_e[:n_write].detach()
            if B > space:
                # Wrap around
                overflow = B - space
                self.z_buffer[:overflow] = z_e[space:space + overflow].detach()
            self.z_buffer_ptr[0] = (ptr + B) % self.z_buffer_size
            self.z_buffer_count[0] = min(
                self.z_buffer_count.item() + B, self.z_buffer_size
            )

    def restart_dead_codes(self, z_e: torch.Tensor = None, threshold: float = None):
        """Re-initialize dead codebook entries from z_e buffer (not just current batch).

        Uses accumulated z_e buffer for diverse replacements. Falls back to z_e arg
        if buffer is empty. Noise scaled to z_e standard deviation for meaningful
        perturbation.

        Args:
            z_e: [B, z_dim] optional fallback batch of encoder outputs
            threshold: codes with cluster_size below this are considered dead.
                       If None, uses adaptive threshold (10% of mean).
        """
        with torch.no_grad():
            mean_cs = self.cluster_size.mean()
            if threshold is None:
                # Aggressive: dead if < 10% of mean cluster size
                threshold = max(mean_cs.item() * 0.10, 1e-6)

            dead_mask = self.cluster_size < threshold
            n_dead = dead_mask.sum().item()
            if n_dead == 0:
                return 0

            # Use z_e buffer if available, else fall back to z_e argument
            buf_count = self.z_buffer_count.item()
            if buf_count > 0:
                pool = self.z_buffer[:int(buf_count)]
            elif z_e is not None:
                pool = z_e
            else:
                return 0

            # Sample replacements from pool
            pool_size = pool.shape[0]
            replace_idxs = torch.randint(0, pool_size, (int(n_dead),), device=pool.device)
            replacements = pool[replace_idxs].clone()

            # Add noise proportional to z_e distribution spread (not fixed 0.01)
            z_std = pool.std(dim=0, keepdim=True).clamp(min=0.01)  # [1, z_dim]
            noise = torch.randn_like(replacements) * z_std * 0.5  # 50% of per-dim std
            replacements = replacements + noise

            dead_indices = dead_mask.nonzero(as_tuple=True)[0]
            self.embedding.weight.data[dead_indices] = replacements
            self.embed_avg[dead_indices] = replacements
            self.cluster_size[dead_indices] = mean_cs  # match average

            return int(n_dead)

    def codebook_usage(self) -> float:
        """Fraction of codebook entries actively used (above 10% of mean cluster size)."""
        mean_cs = self.cluster_size.mean()
        if mean_cs < 1e-8:
            return 0.0
        return (self.cluster_size > mean_cs * 0.10).float().mean().item()


class VQEncoder(nn.Module):
    """CNN encoder -> sequence of continuous z_e tokens -> VQ -> discrete z_q tokens.

    Sequence VQ: replaces the single [B, z_dim] global pooling with [B, L, z_dim]
    spatial tokens from a 2×4 adaptive pool. Each of the L=8 positions captures a
    different spatial region, creating compositional concept "sentences" that cover
    the full input/output grid pair. VectorQuantizer receives B*L items so entropy
    regularization is meaningful even with small task batch sizes.
    """

    def __init__(
        self,
        z_dim: int = 128,
        d_enc: int = 32,
        n_colors: int = 11,
        n_embeddings: int = 64,
        n_tokens: int = 8,
        beta: float = 0.25,
        decay: float = 0.99,
        entropy_weight: float = 0.1,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.n_tokens = n_tokens  # L — spatial positions (2*4 = 8)
        self.color_embed = nn.Embedding(n_colors, d_enc)  # 11 colors (10 + PAD)
        # CNN on stacked [input, output]: [B, 2*d_enc, H, W]
        self.conv1 = nn.Conv2d(2 * d_enc, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        # 2×4 pool produces L=8 spatial positions — flatten spatial before proj
        self.pool = nn.AdaptiveAvgPool2d((2, 4))  # -> [B, 128, 2, 4]
        # proj applied per-position: [B, L, 128] -> [B, L, z_dim]
        self.proj = nn.Sequential(
            nn.Linear(128, z_dim),
            nn.LayerNorm(z_dim),
        )

        # Vector quantizer — processes [B*L, z_dim] (no internal changes needed)
        self.vq = VectorQuantizer(
            n_embeddings=n_embeddings, z_dim=z_dim, beta=beta, decay=decay,
            entropy_weight=entropy_weight,
        )

    def encode_pair(
        self, input_grid: torch.Tensor, output_grid: torch.Tensor
    ) -> torch.Tensor:
        """Encode one (input, output) pair -> z_e [B, L, z_dim] (continuous, pre-VQ).

        Args:
            input_grid: [B, H, W] int tensor (colors 0-9, pad=10)
            output_grid: [B, H, W] int tensor

        Returns:
            z_e: [B, L, z_dim] continuous encoder output (L=n_tokens spatial positions)
        """
        in_emb = self.color_embed(input_grid).permute(0, 3, 1, 2)
        out_emb = self.color_embed(output_grid).permute(0, 3, 1, 2)
        x = torch.cat([in_emb, out_emb], dim=1)  # [B, 2*d_enc, H, W]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.pool(x)             # [B, 128, 2, 4]
        x = x.flatten(2).transpose(1, 2)  # [B, 8, 128]
        return self.proj(x)          # [B, L, z_dim]

    def forward(
        self, demo_pairs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode K demo pairs, mean-pool across demos, quantize each spatial token.

        Args:
            demo_pairs: list of (input_grid [B, H, W], output_grid [B, H, W]) tuples

        Returns:
            z_e: [B, L, z_dim] continuous (pre-VQ) — for commitment loss
            z_q: [B, L, z_dim] quantized (STE) — for decoder and concept bank
            vq_loss: scalar commitment loss
            indices: [B, L] codebook indices per spatial position
        """
        # Each encode_pair returns [B, L, z_dim]; mean-pool across demos
        zs = [self.encode_pair(inp, out) for inp, out in demo_pairs]
        z_e = torch.stack(zs).mean(dim=0)  # [B, L, z_dim]

        B, L, z_dim = z_e.shape

        # Flatten spatial dimension so VQ sees [B*L, z_dim]
        z_e_flat = z_e.reshape(B * L, z_dim)
        z_q_flat, vq_loss, indices_flat = self.vq(z_e_flat)

        # Reshape back to sequence format
        z_q = z_q_flat.reshape(B, L, z_dim)
        indices = indices_flat.reshape(B, L)

        return z_e, z_q, vq_loss, indices


if __name__ == "__main__":
    print("Testing VQEncoder (sequence VQ)...")

    L = 8  # n_tokens = 2*4
    enc = VQEncoder(z_dim=128, d_enc=32, n_embeddings=64, n_tokens=L)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"  Params: {n_params:,}")

    B = 4
    pairs = [
        (torch.randint(0, 10, (B, 5, 5)), torch.randint(0, 10, (B, 5, 5))),
        (torch.randint(0, 10, (B, 5, 5)), torch.randint(0, 10, (B, 5, 5))),
    ]

    z_e, z_q, vq_loss, indices = enc(pairs)
    assert z_e.shape == (B, L, 128), f"z_e shape: {z_e.shape}, expected ({B}, {L}, 128)"
    assert z_q.shape == (B, L, 128), f"z_q shape: {z_q.shape}, expected ({B}, {L}, 128)"
    assert indices.shape == (B, L), f"indices shape: {indices.shape}, expected ({B}, {L})"
    assert vq_loss.ndim == 0, f"vq_loss should be scalar, got {vq_loss.shape}"
    print(f"  z_e shape: {z_e.shape}")
    print(f"  z_q shape: {z_q.shape}")
    print(f"  vq_loss: {vq_loss.item():.4f}")
    print(f"  indices shape: {indices.shape}")
    print(f"  unique codes used: {indices.unique().numel()} / {enc.vq.n_embeddings}")

    # Gradient flow through STE
    loss = z_q.sum() + vq_loss
    loss.backward()
    has_grad = sum(1 for p in enc.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for _ in enc.parameters())
    print(f"  Gradients: {has_grad}/{total} params have nonzero grad")

    # Check STE: z_q should have gradients w.r.t. encoder params
    enc.zero_grad()
    z_e2, z_q2, vq_loss2, _ = enc(pairs)
    (z_q2.sum()).backward()
    assert enc.conv1.weight.grad is not None, "STE failed: no grad through conv1"
    print(f"  STE gradient through conv1: OK")

    # EMA update happens during forward (training mode)
    enc.train()
    for _ in range(10):
        z_e, z_q, vq_loss, _ = enc(pairs)
    usage = enc.vq.codebook_usage()
    print(f"  Codebook usage after 10 steps: {usage:.4f}")

    # Dead code restart (uses accumulated z_e buffer)
    enc.eval()  # stop EMA updates
    n_restarted = enc.vq.restart_dead_codes()
    print(f"  Dead codes restarted: {n_restarted}")
    print(f"  z_buffer count: {enc.vq.z_buffer_count.item()}")

    print("VQEncoder OK")
