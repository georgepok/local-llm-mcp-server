"""Quick gradient direction probe — runs on CPU, 1 batch per domain."""
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

config = LiquidARCConfig.from_yaml("configs/universality_combined.yaml")
device = torch.device("cpu")
model = create_model(config, device)
ckpt = torch.load("output_universality/combined_transfer/checkpoints/best.pt",
                   map_location=device, weights_only=False)
cleaned = {k.replace("._orig_mod.", "."): v for k, v in ckpt["model"].items()}
model.load_state_dict(cleaned)
model.eval()
print(f"Model loaded (step {ckpt['step']})")

tasks = {
    "sorting": SortingTask(seq_len=2048, augment=False, n_demos=2),
    "logic": LogicInferenceTask(seq_len=2048, augment=False, n_demos=2),
    "pattern": PatternCompletionTask(seq_len=2048, augment=False, n_demos=2),
    "graph": GraphColoringTask(seq_len=2048, augment=False, n_demos=2),
}
for t in tasks.values():
    t._seed_counter = 42

dynamics = model.dynamics
domain_grads = {}

print("\nComputing per-domain gradients (1 batch each)...")
for dname, task in tasks.items():
    model.zero_grad()
    _, _, meta = task.generate_batch(2, device=device)
    result = model(
        colors=meta["colors"], xs=meta["xs"], ys=meta["ys"],
        roles=meta["roles"], sep_mask=meta["sep_mask"],
        sep_types=meta["sep_types"], target_mask=meta["target_mask"],
        target_labels=meta["target_labels"], context_mask=meta["context_mask"],
        grid_ids=meta.get("grid_ids"), lengths=meta.get("lengths"),
        target_input_colors=meta.get("target_input_colors"),
    )
    result["loss"].backward()

    grads = {}
    for name, p in dynamics.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float().flatten()
        for key in ["ffn", "metric", "W_o", "W_v", "tau", "gate"]:
            if key in name:
                grads.setdefault(key, []).append(g)
                break

    for key in grads:
        grads[key] = torch.cat(grads[key])
    domain_grads[dname] = grads
    model.zero_grad()
    print(f"  {dname}: {', '.join(f'{k}={v.shape[0]}' for k, v in grads.items())}")

domains = list(domain_grads.keys())
modules = sorted(set(k for g in domain_grads.values() for k in g.keys()))

for module in modules:
    print(f"\n{module} gradient cosine similarity:")
    header = "            "
    for d in domains:
        header += f"  {d:>10s}"
    print(header)
    for d1 in domains:
        line = f"  {d1:10s}"
        for d2 in domains:
            g1 = domain_grads[d1].get(module)
            g2 = domain_grads[d2].get(module)
            if g1 is not None and g2 is not None:
                cos = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
            else:
                cos = float('nan')
            line += f"  {cos:10.3f}"
        print(line)

print("\nInterpretation:")
print("  ~1.0 = domains push this module in the SAME direction (shared structure)")
print("  ~0.0 = orthogonal gradients (separate structures, no interference)")
print("  <0.0 = conflicting gradients (interference between domains)")
