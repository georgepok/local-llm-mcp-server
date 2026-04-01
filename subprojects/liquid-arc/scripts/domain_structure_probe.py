"""Domain Structure Probe — analyze whether FFN/metric develop separate or shared structures.

Three analyses:
1. FFN activation overlap: Which neurons fire for which domains?
2. Gradient direction similarity: Do domains push weights in the same direction?
3. Metric routing divergence: How different are heat kernel patterns across domains?

Usage:
    python scripts/domain_structure_probe.py \
        --checkpoint output_universality/combined_transfer/checkpoints/best.pt \
        --config configs/universality_combined.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liquid_arc.config import LiquidARCConfig
from liquid_arc.model import LiquidARCModel, create_model
from liquid_arc.tasks.sorting import SortingTask
from liquid_arc.tasks.logic_inference import LogicInferenceTask
from liquid_arc.tasks.pattern_completion import PatternCompletionTask
from liquid_arc.tasks.graph_coloring import GraphColoringTask


def load_model(checkpoint_path, config, device):
    model = create_model(config, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.eval()
    return model


def get_domain_tasks(seq_len):
    tasks = {
        "sorting": SortingTask(seq_len=seq_len, augment=False, n_demos=2),
        "logic": LogicInferenceTask(seq_len=seq_len, augment=False, n_demos=2),
        "pattern": PatternCompletionTask(seq_len=seq_len, augment=False, n_demos=2),
        "graph": GraphColoringTask(seq_len=seq_len, augment=False, n_demos=2),
    }
    for t in tasks.values():
        t._seed_counter = 42  # deterministic
    return tasks


# ─── Analysis 1: FFN Activation Overlap ───────────────────────────────

def probe_ffn_activations(model, tasks, device, n_batches=10, batch_size=4):
    """Record which FFN neurons activate (>0 after GELU) per domain."""
    dynamics = model.dynamics._orig_mod if hasattr(model.dynamics, '_orig_mod') else model.dynamics

    # Hook the FFN hidden layer to capture activations
    ffn_activations = {}

    def hook_fn(domain_name):
        def _hook(module, input, output):
            # output is post-activation [B, N, d_ffn]
            if domain_name not in ffn_activations:
                ffn_activations[domain_name] = []
            # Mean activation magnitude per neuron across batch and positions
            ffn_activations[domain_name].append(
                output.detach().float().mean(dim=(0, 1)).cpu()  # [d_ffn]
            )
        return _hook

    # Find the FFN's first linear + activation (GELU)
    # The FFN is dynamics.ffn — typically nn.Sequential(Linear, GELU, Linear)
    ffn = dynamics.ffn
    # Hook the output of the second module (post-GELU)
    hook_target = ffn[1] if len(ffn) > 1 else ffn[0]

    print("\n=== FFN Activation Analysis ===")

    domain_mean_acts = {}
    for dname, task in tasks.items():
        handle = hook_target.register_forward_hook(hook_fn(dname))
        ffn_activations[dname] = []

        with torch.no_grad():
            for _ in range(n_batches):
                _, _, meta = task.generate_batch(batch_size, device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=(device.type == "cuda")):
                    model(
                        colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                        roles=meta["roles"], sep_mask=meta["sep_mask"],
                        sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                        target_labels=meta["target_labels"],
                        context_mask=meta["context_mask"],
                        grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                        target_input_colors=meta.get("target_input_colors"),
                    )
        handle.remove()

        # Average activation per neuron across all batches
        acts = torch.stack(ffn_activations[dname]).mean(dim=0)  # [d_ffn]
        domain_mean_acts[dname] = acts

    # Compute active neuron sets (above median activation)
    domains = list(domain_mean_acts.keys())
    d_ffn = domain_mean_acts[domains[0]].shape[0]

    # Top-K active neurons per domain (top 20%)
    k = d_ffn // 5
    active_sets = {}
    for dname in domains:
        topk_idx = domain_mean_acts[dname].topk(k).indices
        active_sets[dname] = set(topk_idx.tolist())

    # Jaccard similarity between domain active neuron sets
    print(f"\nFFN active neuron overlap (top {k}/{d_ffn} neurons, Jaccard similarity):")
    print(f"{'':12s}", end="")
    for d in domains:
        print(f"  {d:>10s}", end="")
    print()
    for d1 in domains:
        print(f"  {d1:10s}", end="")
        for d2 in domains:
            jaccard = len(active_sets[d1] & active_sets[d2]) / len(active_sets[d1] | active_sets[d2])
            print(f"  {jaccard:10.3f}", end="")
        print()

    # Cosine similarity of activation patterns
    print(f"\nFFN activation pattern cosine similarity:")
    print(f"{'':12s}", end="")
    for d in domains:
        print(f"  {d:>10s}", end="")
    print()
    for d1 in domains:
        print(f"  {d1:10s}", end="")
        for d2 in domains:
            cos = F.cosine_similarity(
                domain_mean_acts[d1].unsqueeze(0),
                domain_mean_acts[d2].unsqueeze(0)
            ).item()
            print(f"  {cos:10.3f}", end="")
        print()

    return domain_mean_acts


# ─── Analysis 2: Gradient Direction Similarity ────────────────────────

def probe_gradient_directions(model, tasks, device, n_batches=5, batch_size=4):
    """Compute per-domain gradients on FFN weights and measure cosine similarity."""
    print("\n=== Gradient Direction Analysis ===")

    dynamics = model.dynamics._orig_mod if hasattr(model.dynamics, '_orig_mod') else model.dynamics

    # Collect gradients for FFN weights per domain
    domain_grads = {}
    for dname, task in tasks.items():
        model.zero_grad()
        total_loss = 0.0
        for _ in range(n_batches):
            _, _, meta = task.generate_batch(batch_size, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(
                    colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                    roles=meta["roles"], sep_mask=meta["sep_mask"],
                    sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                    target_labels=meta["target_labels"],
                    context_mask=meta["context_mask"],
                    grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                    target_input_colors=meta.get("target_input_colors"),
                )
            result["loss"].backward()
            total_loss += result["loss"].item()

        # Flatten all FFN parameter gradients into one vector
        grad_vec = []
        for name, param in dynamics.named_parameters():
            if 'ffn' in name and param.grad is not None:
                grad_vec.append(param.grad.detach().float().flatten())
        grad_vec = torch.cat(grad_vec)
        domain_grads[dname] = grad_vec.clone()
        model.zero_grad()

    domains = list(domain_grads.keys())

    print(f"\nFFN gradient cosine similarity (do domains push weights the same way?):")
    print(f"{'':12s}", end="")
    for d in domains:
        print(f"  {d:>10s}", end="")
    print()
    for d1 in domains:
        print(f"  {d1:10s}", end="")
        for d2 in domains:
            cos = F.cosine_similarity(
                domain_grads[d1].unsqueeze(0),
                domain_grads[d2].unsqueeze(0)
            ).item()
            print(f"  {cos:10.3f}", end="")
        print()

    # Also check metric network gradients
    metric_grads = {}
    for dname, task in tasks.items():
        model.zero_grad()
        for _ in range(n_batches):
            _, _, meta = task.generate_batch(batch_size, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(
                    colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                    roles=meta["roles"], sep_mask=meta["sep_mask"],
                    sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                    target_labels=meta["target_labels"],
                    context_mask=meta["context_mask"],
                    grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                    target_input_colors=meta.get("target_input_colors"),
                )
            result["loss"].backward()

        grad_vec = []
        for name, param in dynamics.named_parameters():
            if 'metric' in name and param.grad is not None:
                grad_vec.append(param.grad.detach().float().flatten())
        if grad_vec:
            metric_grads[dname] = torch.cat(grad_vec).clone()
        model.zero_grad()

    if metric_grads:
        print(f"\nMetricNet gradient cosine similarity:")
        print(f"{'':12s}", end="")
        for d in domains:
            print(f"  {d:>10s}", end="")
        print()
        for d1 in domains:
            print(f"  {d1:10s}", end="")
            for d2 in domains:
                cos = F.cosine_similarity(
                    metric_grads[d1].unsqueeze(0),
                    metric_grads[d2].unsqueeze(0)
                ).item()
                print(f"  {cos:10.3f}", end="")
            print()

    # W_o gradients (content transformation)
    wo_grads = {}
    for dname, task in tasks.items():
        model.zero_grad()
        for _ in range(n_batches):
            _, _, meta = task.generate_batch(batch_size, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(
                    colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                    roles=meta["roles"], sep_mask=meta["sep_mask"],
                    sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                    target_labels=meta["target_labels"],
                    context_mask=meta["context_mask"],
                    grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                    target_input_colors=meta.get("target_input_colors"),
                )
            result["loss"].backward()

        grad_vec = []
        for name, param in dynamics.named_parameters():
            if 'W_o' in name and param.grad is not None:
                grad_vec.append(param.grad.detach().float().flatten())
        if grad_vec:
            wo_grads[dname] = torch.cat(grad_vec).clone()
        model.zero_grad()

    if wo_grads:
        print(f"\nW_o gradient cosine similarity:")
        print(f"{'':12s}", end="")
        for d in domains:
            print(f"  {d:>10s}", end="")
        print()
        for d1 in domains:
            print(f"  {d1:10s}", end="")
            for d2 in domains:
                cos = F.cosine_similarity(
                    wo_grads[d1].unsqueeze(0),
                    wo_grads[d2].unsqueeze(0)
                ).item()
                print(f"  {cos:10.3f}", end="")
            print()

    return domain_grads


# ─── Analysis 3: Hidden State Representation Similarity ───────────────

def probe_representations(model, tasks, device, n_batches=10, batch_size=4):
    """Compare post-ODE hidden state distributions across domains using CKA."""
    print("\n=== Representation Similarity (post-ODE hidden states) ===")

    domain_hiddens = {}
    for dname, task in tasks.items():
        hiddens = []
        with torch.no_grad():
            for _ in range(n_batches):
                _, _, meta = task.generate_batch(batch_size, device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=(device.type == "cuda")):
                    result = model(
                        colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
                        roles=meta["roles"], sep_mask=meta["sep_mask"],
                        sep_types=meta["sep_types"], target_mask=meta["target_mask"],
                        target_labels=meta["target_labels"],
                        context_mask=meta["context_mask"],
                        grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
                        target_input_colors=meta.get("target_input_colors"),
                    )
                # Get logits as proxy for final hidden state
                logits = result["logits"]  # [B, N, n_colors]
                # Only take target positions
                mask = meta["target_mask"]
                for b in range(batch_size):
                    tgt_pos = mask[b].nonzero(as_tuple=True)[0]
                    if len(tgt_pos) > 0:
                        hiddens.append(logits[b, tgt_pos].float().cpu())

        # Concatenate all target hidden states
        domain_hiddens[dname] = torch.cat(hiddens, dim=0)  # [n_total_targets, d]

    domains = list(domain_hiddens.keys())

    # Linear CKA (centered kernel alignment)
    def linear_cka(X, Y):
        """CKA between two [n, d] matrices (subsample to same n)."""
        n = min(X.shape[0], Y.shape[0], 500)
        X = X[:n]
        Y = Y[:n]
        X = X - X.mean(dim=0)
        Y = Y - Y.mean(dim=0)
        hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
        hsic_xx = (X @ X.T * (X @ X.T)).sum()
        hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()
        return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)).item()

    print(f"\nLinear CKA (1.0 = identical representations, 0.0 = orthogonal):")
    print(f"{'':12s}", end="")
    for d in domains:
        print(f"  {d:>10s}", end="")
    print()
    for d1 in domains:
        print(f"  {d1:10s}", end="")
        for d2 in domains:
            cka = linear_cka(domain_hiddens[d1], domain_hiddens[d2])
            print(f"  {cka:10.3f}", end="")
        print()

    return domain_hiddens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = LiquidARCConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)

    tasks = get_domain_tasks(config.max_seq_len)
    print(f"Domains: {list(tasks.keys())}")

    probe_ffn_activations(model, tasks, device)
    probe_gradient_directions(model, tasks, device)
    probe_representations(model, tasks, device)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
