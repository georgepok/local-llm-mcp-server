#!/usr/bin/env python3
"""Probe all checkpoints for geometric stats."""
import os, sys, torch, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fgn.config import FGNConfig
from fgn.model_fluid import FluidNetModel
from fgn.tasks import get_task
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Probe both iter3 and baseline
runs = [
    ("Iter3 (K=3)", "configs/grokking_iter3.yaml", "output_grokking_iter3/checkpoints"),
    ("Baseline (K=1)", "configs/criticality_starved.yaml", "output_grokking/checkpoints"),
]

task_kwargs = {
    "n_rooms_max": 5, "n_objects": 2, "space_size": 60.0,
    "connect_radius": 30.0, "locked_door_prob": 0.2,
    "n_rooms_min": 4, "min_steps": 2, "max_steps": 5, "min_state_changes": 1
}

tokenizer = AutoTokenizer.from_pretrained("gpt2")

for run_name, config_path, ckpt_dir in runs:
    config = FGNConfig.from_yaml(config_path)
    task = get_task("CW", tokenizer, seq_len=config.max_seq_len, **task_kwargs)

    # Generate one probe episode
    random.seed(42)
    torch.manual_seed(42)
    input_ids = None
    for _ in range(500):
        ep = task._generate_valid_episode()
        if ep is None:
            continue
        episode_text, _actions, _n_steps, _opt_cost, _step_costs, _world = ep
        ids, labels, ctx_end, _, _ = task._tokenize_episode(episode_text)
        if len(ids) <= config.max_seq_len:
            pad_id = tokenizer.eos_token_id or 0
            pad_len = config.max_seq_len - len(ids)
            ids += [pad_id] * pad_len
            n_sup = sum(1 for l in labels if l != -100)
            if n_sup >= 5:
                input_ids = ids
                break

    if input_ids is None:
        print(f"  {run_name}: Could not generate probe episode")
        continue

    input_t = torch.tensor([input_ids], device=device)

    # Find all checkpoints
    ckpt_files = sorted(os.listdir(ckpt_dir))
    ckpt_files = [f for f in ckpt_files if f.endswith(".pt")]

    # Sort by step number
    def step_key(name):
        if name == "final.pt":
            return 999999
        return int(name.replace("step_", "").replace(".pt", ""))
    ckpt_files.sort(key=step_key)

    print()
    print("=" * 70)
    print(f"  {run_name}")
    print("=" * 70)
    header = f"  {'Checkpoint':<14} {'|k|':>8} {'CV':>8} {'t_loc':>8} {'t_mid':>8} {'t_glo':>8}"
    print(header)
    print("  " + "-" * 60)

    for fname in ckpt_files:
        path = os.path.join(ckpt_dir, fname)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = FluidNetModel(config).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        with torch.no_grad():
            result = model(input_t)

        kappa = result["avg_kappa"].item()
        cv_val = result["metric_cv"] if isinstance(result["metric_cv"], float) else result["metric_cv"].item()
        t_loc = result["avg_t_local"].item()
        t_mid = result["avg_t_medium"].item()
        t_glo = result["avg_t_global"].item()

        label = fname.replace(".pt", "")
        print(f"  {label:<14} {kappa:8.4f} {cv_val:8.4f} {t_loc:8.3f} {t_mid:8.3f} {t_glo:8.3f}")

        del model, ckpt
        torch.cuda.empty_cache()

print()
print("Done.")
