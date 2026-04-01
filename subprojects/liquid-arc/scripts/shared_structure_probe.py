"""Shared Structure Probe — do domains share abstract computational structure?

Three levels of analysis:
1. FFN activation subspace overlap (SVD): Are active neuron patterns in the same subspace?
2. Cross-domain transfer matrix: Does training on one domain help another?
3. Representation CKA: Do hidden states share structure across domains?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import create_model
from liquid_arc.tasks.sorting import SortingTask
from liquid_arc.tasks.logic_inference import LogicInferenceTask
from liquid_arc.tasks.pattern_completion import PatternCompletionTask
from liquid_arc.tasks.graph_coloring import GraphColoringTask


def load_model(path, config, device):
    model = create_model(config, device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(cleaned)
    model.eval()
    return model, ckpt["step"]


def get_tasks(seq_len):
    tasks = {
        "sorting": SortingTask(seq_len=seq_len, augment=False, n_demos=2),
        "logic": LogicInferenceTask(seq_len=seq_len, augment=False, n_demos=2),
        "pattern": PatternCompletionTask(seq_len=seq_len, augment=False, n_demos=2),
        "graph": GraphColoringTask(seq_len=seq_len, augment=False, n_demos=2),
    }
    for t in tasks.values():
        t._seed_counter = 42
    return tasks


def run_forward(model, task, device, n_batches=3, batch_size=2):
    """Run forward pass, return logits at target positions."""
    all_logits = []
    with torch.no_grad():
        for _ in range(n_batches):
            _, _, meta = task.generate_batch(batch_size, device=device)
            result = model(
                colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                roles=meta["roles"], sep_mask=meta["sep_mask"],
                sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                target_labels=meta["target_labels"], context_mask=meta["context_mask"],
                grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                target_input_colors=meta.get("target_input_colors"),
            )
            logits = result["logits"]  # [B, N, 10]
            mask = meta["target_mask"]
            for b in range(batch_size):
                pos = mask[b].nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    all_logits.append(logits[b, pos].float().cpu())
    return torch.cat(all_logits, dim=0)  # [n_targets, 10]


# ─── Analysis 1: FFN activation subspace overlap via SVD ──────────────

def probe_ffn_subspace(model, tasks, device, n_batches=5, batch_size=2):
    """Capture FFN activations per domain, compute SVD, measure subspace overlap."""
    print("\n=== FFN Activation Subspace Analysis (SVD) ===")
    dynamics = model.dynamics
    ffn = dynamics.ffn

    # Hook post-GELU activations
    captured = {"acts": []}

    def hook_fn(module, inp, out):
        # out: [B, N, d_ffn] post-GELU
        captured["acts"].append(out.detach().float().cpu())

    hook_target = ffn[1]  # GELU layer
    handle = hook_target.register_forward_hook(hook_fn)

    domain_acts = {}
    for dname, task in tasks.items():
        captured["acts"] = []
        with torch.no_grad():
            for _ in range(n_batches):
                _, _, meta = task.generate_batch(batch_size, device=device)
                model(
                    colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                    roles=meta["roles"], sep_mask=meta["sep_mask"],
                    sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                    target_labels=meta["target_labels"], context_mask=meta["context_mask"],
                    grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                    target_input_colors=meta.get("target_input_colors"),
                )
        # Flatten: [total_tokens, d_ffn]
        all_acts = torch.cat(captured["acts"], dim=0)  # [sum(B*N), d_ffn]
        # Subsample to manageable size
        n_total = all_acts.shape[0] * all_acts.shape[1]
        flat = all_acts.reshape(-1, all_acts.shape[-1])
        if flat.shape[0] > 2000:
            idx = torch.randperm(flat.shape[0])[:2000]
            flat = flat[idx]
        domain_acts[dname] = flat
        print(f"  {dname}: {flat.shape[0]} tokens, d_ffn={flat.shape[1]}")

    handle.remove()

    # SVD of each domain's activations
    # Extract top-k principal components, measure subspace overlap
    k = 50  # top-50 directions
    domains = list(domain_acts.keys())
    bases = {}
    for dname in domains:
        X = domain_acts[dname]
        X = X - X.mean(dim=0)  # center
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        bases[dname] = Vh[:k]  # [k, d_ffn] — top-k right singular vectors
        # Report variance explained
        total_var = (S ** 2).sum()
        topk_var = (S[:k] ** 2).sum()
        print(f"  {dname}: top-{k} explains {topk_var/total_var*100:.1f}% variance")

    # Subspace overlap: ||P_A @ P_B^T||_F^2 / k
    # This measures how much of domain A's subspace overlaps with B's
    print(f"\nSubspace overlap (top-{k} SVD directions, Grassmann similarity):")
    header = "            "
    for d in domains:
        header += f"  {d:>10s}"
    print(header)
    for d1 in domains:
        line = f"  {d1:10s}"
        for d2 in domains:
            # Grassmann: sum of squared cosines of principal angles
            M = bases[d1] @ bases[d2].T  # [k, k]
            singular_values = torch.linalg.svdvals(M)
            # Normalized overlap: mean of squared singular values
            overlap = (singular_values ** 2).mean().item()
            line += f"  {overlap:10.3f}"
        print(line)

    print("\n  1.0 = identical subspaces, 1/d_ffn = random overlap")

    # Also: which neurons are EXCLUSIVELY active for one domain?
    print(f"\nDomain-specific neurons (>2x mean activation vs other domains):")
    mean_acts = {d: domain_acts[d].mean(dim=0) for d in domains}
    for d in domains:
        own = mean_acts[d]
        others = torch.stack([mean_acts[d2] for d2 in domains if d2 != d]).mean(dim=0)
        ratio = own / (others + 1e-8)
        n_specific = (ratio > 2.0).sum().item()
        n_suppressed = (ratio < 0.5).sum().item()
        print(f"  {d:10s}: {n_specific:4d} specific (>2x), {n_suppressed:4d} suppressed (<0.5x)")


# ─── Analysis 2: Cross-domain accuracy (zero-shot transfer) ──────────

def probe_cross_domain_accuracy(model, tasks, device, n_batches=5, batch_size=2):
    """Measure accuracy on each domain — the trained combined model should
    show which domains share computational structure via correlated accuracy."""
    print("\n=== Cross-Domain Accuracy (current combined model) ===")

    for dname, task in tasks.items():
        total_correct = 0
        total_xform = 0
        total_cells = 0
        total_xform_cells = 0
        with torch.no_grad():
            for _ in range(n_batches):
                _, _, meta = task.generate_batch(batch_size, device=device)
                result = model(
                    colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                    roles=meta["roles"], sep_mask=meta["sep_mask"],
                    sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                    target_labels=meta["target_labels"],
                    context_mask=meta["context_mask"],
                    grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                    target_input_colors=meta.get("target_input_colors"),
                )
                ca = result.get("cell_accuracy", torch.tensor(0.0))
                if isinstance(ca, torch.Tensor):
                    ca = ca.item()
                xa = result.get("transform_accuracy", torch.tensor(0.0))
                if isinstance(xa, torch.Tensor):
                    xa = xa.item()
                nx = result.get("n_transform", torch.tensor(0))
                if isinstance(nx, torch.Tensor):
                    nx = nx.item()
                n_tgt = (meta["target_labels"] != -100).sum().item()
                total_correct += int(ca * n_tgt)
                total_cells += n_tgt
                total_xform += int(xa * nx)
                total_xform_cells += nx

        cell = total_correct / max(total_cells, 1)
        xform = total_xform / max(total_xform_cells, 1)
        print(f"  {dname:10s}: cell={cell:.3f}  xform={xform:.3f}")


# ─── Analysis 3: Representation CKA ──────────────────────────────────

def probe_representation_cka(model, tasks, device):
    """Linear CKA between output representations across domains."""
    print("\n=== Output Representation CKA ===")

    domain_reps = {}
    for dname, task in tasks.items():
        domain_reps[dname] = run_forward(model, task, device, n_batches=5, batch_size=2)
        print(f"  {dname}: {domain_reps[dname].shape[0]} target tokens")

    def linear_cka(X, Y):
        n = min(X.shape[0], Y.shape[0], 500)
        X, Y = X[:n], Y[:n]
        X = X - X.mean(dim=0)
        Y = Y - Y.mean(dim=0)
        hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
        hsic_xx = (X @ X.T * (X @ X.T)).sum()
        hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()
        return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)).item()

    domains = list(domain_reps.keys())
    print(f"\nLinear CKA (output logits at target positions):")
    header = "            "
    for d in domains:
        header += f"  {d:>10s}"
    print(header)
    for d1 in domains:
        line = f"  {d1:10s}"
        for d2 in domains:
            cka = linear_cka(domain_reps[d1], domain_reps[d2])
            line += f"  {cka:10.3f}"
        print(line)
    print("\n  1.0 = identical structure, 0.0 = unrelated")


# ─── Analysis 4: Gradient subspace overlap ────────────────────────────

def probe_gradient_subspace(model, tasks, device, n_batches=3, batch_size=2):
    """Like gradient cosine, but check if gradients share a LOW-RANK subspace
    even when individual gradient vectors are orthogonal.

    Stack multiple per-sample gradients → SVD → compare subspaces.
    """
    print("\n=== Gradient Subspace Overlap (multi-sample SVD) ===")

    dynamics = model.dynamics
    domains = list(tasks.keys())

    # Collect per-sample FFN gradients
    domain_grad_matrices = {}
    for dname, task in tasks.items():
        grad_rows = []
        for _ in range(n_batches):
            model.zero_grad()
            _, _, meta = task.generate_batch(batch_size, device=device)
            result = model(
                colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                roles=meta["roles"], sep_mask=meta["sep_mask"],
                sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                target_labels=meta["target_labels"], context_mask=meta["context_mask"],
                grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                target_input_colors=meta.get("target_input_colors"),
            )
            result["loss"].backward()
            g = []
            for name, p in dynamics.named_parameters():
                if "ffn" in name and p.grad is not None:
                    g.append(p.grad.detach().float().flatten())
            grad_rows.append(torch.cat(g))
            model.zero_grad()

        G = torch.stack(grad_rows)  # [n_batches, d_params]
        domain_grad_matrices[dname] = G
        print(f"  {dname}: {G.shape[0]} gradient samples, dim={G.shape[1]}")

    # SVD of each domain's gradient matrix
    k = min(n_batches, 3)
    bases = {}
    for dname in domains:
        G = domain_grad_matrices[dname]
        G = G - G.mean(dim=0)
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        bases[dname] = Vh[:k]
        var_explained = (S[:k] ** 2).sum() / (S ** 2).sum()
        print(f"  {dname}: top-{k} explains {var_explained*100:.1f}% of gradient variance")

    print(f"\nFFN gradient subspace overlap (top-{k} directions):")
    header = "            "
    for d in domains:
        header += f"  {d:>10s}"
    print(header)
    for d1 in domains:
        line = f"  {d1:10s}"
        for d2 in domains:
            M = bases[d1] @ bases[d2].T
            sv = torch.linalg.svdvals(M)
            overlap = (sv ** 2).mean().item()
            line += f"  {overlap:10.3f}"
        print(line)
    print("  High overlap + low gradient cosine = shared subspace, different directions")
    print("  Low overlap + low gradient cosine = truly separate structures")


def main():
    config = LiquidARCConfig.from_yaml("configs/universality_combined.yaml")
    device = torch.device("cpu")
    model, step = load_model(
        "output_universality/combined_transfer/checkpoints/best.pt", config, device)
    print(f"Loaded step {step}")

    tasks = get_tasks(config.max_seq_len)

    probe_cross_domain_accuracy(model, tasks, device)
    probe_ffn_subspace(model, tasks, device)
    probe_representation_cka(model, tasks, device)
    probe_gradient_subspace(model, tasks, device)

    print("\n=== SUMMARY ===")
    print("If FFN subspace overlap is HIGH but gradient cosine is LOW:")
    print("  → Domains share the same neural circuitry but activate it differently")
    print("  → This indicates shared abstract structure (generalization)")
    print("If FFN subspace overlap is LOW and gradient cosine is LOW:")
    print("  → Domains use completely separate neuron populations")
    print("  → No shared abstraction — the model is partitioning capacity")


if __name__ == "__main__":
    main()
