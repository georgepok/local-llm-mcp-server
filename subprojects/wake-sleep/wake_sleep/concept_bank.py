"""ConceptBank — store, retrieve, and interpolate concept token sequences.

CPU storage to save GPU memory. Moved to GPU on sample().
The core Sleep innovation: interpolate between real concepts to dream novel rules.

V2 Sequence VQ: stores flattened [L*z_dim] concept sequences (where L = n_tokens).
Interpolation occurs in flattened space; sequences are reshaped on retrieval to
[n, L, z_dim] for the decoder. Backward-compatible with L=1 (scalar z_dim case).
"""

import torch


class ConceptBank:
    """Store, retrieve, and interpolate concept token sequences.

    Internally stores flattened sequences [L*z_dim] on CPU. On retrieval,
    reshapes to [n, L, z_dim] for the decoder. Supports both the legacy
    single-vector case (L=1) and the sequence case (L>1).
    """

    def __init__(self, max_size: int = 10000, z_dim: int = 128, n_tokens: int = 1):
        """Initialize concept bank.

        Args:
            max_size: maximum number of stored concepts (FIFO when full)
            z_dim: latent dimension per token
            n_tokens: number of spatial tokens per concept (L); 1 = legacy scalar mode
        """
        self.z_dim = z_dim
        self.n_tokens = n_tokens
        self.flat_dim = n_tokens * z_dim  # storage dimension per concept
        self.bank = torch.zeros(max_size, self.flat_dim)  # CPU
        self.count = 0
        self.max_size = max_size

    def add(self, z: torch.Tensor):
        """Add z vectors to bank (FIFO if full).

        Args:
            z: [B, z_dim] (legacy, n_tokens=1) or [B, L, z_dim] (sequence)
               Both are accepted — [B, z_dim] is treated as [B, 1, z_dim].
        """
        z_cpu = z.detach().cpu()
        if z_cpu.ndim == 2:
            # Legacy scalar mode [B, z_dim] — treat as single-token sequence
            z_cpu = z_cpu.unsqueeze(1)  # [B, 1, z_dim]
        # z_cpu is [B, L, z_dim]; flatten to [B, L*z_dim]
        B = z_cpu.shape[0]
        z_flat = z_cpu.reshape(B, self.flat_dim)
        for i in range(B):
            idx = self.count % self.max_size
            self.bank[idx] = z_flat[i]
            self.count += 1

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Random sample n concepts, returned as [n, L, z_dim] on device."""
        valid = min(self.count, self.max_size)
        idxs = torch.randint(0, valid, (n,))
        flat = self.bank[idxs].to(device)  # [n, L*z_dim]
        return flat.reshape(n, self.n_tokens, self.z_dim)  # [n, L, z_dim]

    def sample_interpolated(
        self,
        n: int,
        device: torch.device,
        alpha_min: float = 0.2,
        alpha_max: float = 0.8,
        noise_std: float = 0.1,
    ) -> torch.Tensor:
        """z_dream = alpha * z_A + (1-alpha) * z_B + epsilon — the core Sleep innovation.

        Interpolation in flattened [L*z_dim] space preserves positional structure.

        Returns:
            z_dream: [n, L, z_dim] interpolated concept sequences
        """
        valid = min(self.count, self.max_size)
        idxs_a = torch.randint(0, valid, (n,))
        idxs_b = torch.randint(0, valid, (n,))
        z_a = self.bank[idxs_a].to(device)  # [n, L*z_dim]
        z_b = self.bank[idxs_b].to(device)  # [n, L*z_dim]
        alpha = torch.rand(n, 1, device=device) * (alpha_max - alpha_min) + alpha_min
        z_dream_flat = alpha * z_a + (1 - alpha) * z_b
        z_dream_flat = z_dream_flat + torch.randn_like(z_dream_flat) * noise_std
        return z_dream_flat.reshape(n, self.n_tokens, self.z_dim)  # [n, L, z_dim]

    @property
    def size(self) -> int:
        return min(self.count, self.max_size)


if __name__ == "__main__":
    print("Testing ConceptBank (sequence VQ)...")

    L = 8  # n_tokens
    z_dim = 128
    bank = ConceptBank(max_size=100, z_dim=z_dim, n_tokens=L)

    # Add 50 diverse z sequences [B, L, z_dim]
    for _ in range(50):
        z = torch.randn(1, L, z_dim)
        bank.add(z)
    assert bank.size == 50, f"Expected 50, got {bank.size}"
    print(f"  Bank size: {bank.size}")

    # Sample raw — should return [n, L, z_dim]
    s = bank.sample(10, torch.device("cpu"))
    assert s.shape == (10, L, z_dim), f"Got {s.shape}, expected (10, {L}, {z_dim})"
    assert not torch.isnan(s).any()
    print(f"  Sample shape: {s.shape}")

    # Sample interpolated — should return [n, L, z_dim]
    z_dream = bank.sample_interpolated(20, torch.device("cpu"))
    assert z_dream.shape == (20, L, z_dim), f"Got {z_dream.shape}, expected (20, {L}, {z_dim})"
    assert not torch.isnan(z_dream).any()
    print(f"  Interpolated shape: {z_dream.shape}")
    norms = z_dream.reshape(20, -1).norm(dim=-1)
    print(f"  z_dream flat-norm range: [{norms.min():.3f}, {norms.max():.3f}]")

    # Backward-compatible: add [B, z_dim] (L=1 legacy mode)
    bank1 = ConceptBank(max_size=100, z_dim=z_dim, n_tokens=1)
    for _ in range(10):
        bank1.add(torch.randn(2, z_dim))  # [B, z_dim]
    assert bank1.size == 20
    s1 = bank1.sample(5, torch.device("cpu"))
    assert s1.shape == (5, 1, z_dim), f"Got {s1.shape}"
    print(f"  Legacy (L=1) sample shape: {s1.shape}")

    # FIFO overflow
    for _ in range(200):
        bank.add(torch.randn(1, L, z_dim))
    assert bank.size == 100  # capped at max_size
    print(f"  After overflow: size={bank.size}, count={bank.count}")

    print("ConceptBank OK")
