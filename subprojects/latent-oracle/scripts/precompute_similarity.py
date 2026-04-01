"""Oracle similarity matrix extraction — per-position cosine similarity from Qwen.

Extracts hidden states from Qwen3.5-9B (VLM or text), maps them to ARC grid
cells, and computes pairwise cosine similarity matrices. These matrices serve
as distillation targets for LiquidARC's learned heat kernel geometry.

Architecture-agnostic: uses hidden state similarity (cos(h_i, h_j)), not
attention maps. Works with any layer type (attention, DeltaNet, convolution).

Visual path (default):
  Render ARC task → feed composite image → extract vision token hidden states
  → map patches to grid cells → average per cell → cosine similarity

Text fallback:
  Serialize task → tokenize → map BPE tokens to grid cells via serializer
  structure → average per cell → cosine similarity

Usage:
    python scripts/precompute_similarity.py \
        --model_id Qwen/Qwen3.5-9B-Base \
        --data_dir /workspace/fgn-v3/data/arc \
        --output /workspace/latent-oracle/similarity_matrices.pt \
        --mode visual \
        --d4_augment

    # Text fallback (if visual probe fails):
    python scripts/precompute_similarity.py \
        --model_id Qwen/Qwen3.5-9B-Base \
        --data_dir /workspace/fgn-v3/data/arc \
        --output /workspace/latent-oracle/similarity_matrices_text.pt \
        --mode text
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Import visual rendering from sibling script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from latent_oracle.serialize import apply_d4, serialize_task


# ── Visual rendering (inlined from precompute_visual.py) ─────────────────────

def _import_visual():
    """Import visual rendering utilities from precompute_visual.py."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from precompute_visual import (
        render_task_composite,
        render_grid,
        patch_vision_conv3d,
        load_arc_tasks_raw,
        PROMPT_TEXT,
    )
    return render_task_composite, render_grid, patch_vision_conv3d, load_arc_tasks_raw, PROMPT_TEXT


# ── Grid cell geometry ───────────────────────────────────────────────────────

def compute_grid_cell_positions(
    task: dict, d4_idx: int, test_idx: int,
    cell_size: int = 20, padding: int = 12, arrow_w: int = 30, lw: int = 1,
) -> List[Tuple[int, int, int, int, int]]:
    """Compute pixel center (cx, cy) and grid identity for each cell in the composite.

    Returns list of (cx, cy, row, col, grid_id) for each grid cell across all
    grids in the composite image.

    Grid layout matches render_task_composite exactly:
      - Each row: [padding] [input_grid] [padding+arrow+padding] [output_grid] [padding]
      - Vertical: [padding] [row0] [padding] [row1] ...
    """
    demos = task["train"]
    tests = task["test"]
    test_pair = tests[min(test_idx, len(tests) - 1)]

    # Build list of grid pairs (same order as composite rendering)
    grid_pairs = []
    for demo in demos:
        inp = apply_d4(demo["input"], d4_idx)
        out = apply_d4(demo["output"], d4_idx)
        grid_pairs.append((inp, out))

    test_inp = apply_d4(test_pair["input"], d4_idx)
    grid_pairs.append((test_inp, None))

    cells = []
    grid_id = 0
    y = padding  # vertical cursor

    for inp_grid, out_grid in grid_pairs:
        H_in = len(inp_grid)
        W_in = len(inp_grid[0]) if H_in > 0 else 0
        img_w_in = W_in * cell_size + (W_in + 1) * lw
        img_h_in = H_in * cell_size + (H_in + 1) * lw

        # Compute row height for vertical centering
        row_h = img_h_in
        if out_grid is not None:
            H_out = len(out_grid)
            W_out = len(out_grid[0]) if H_out > 0 else 0
            img_h_out = H_out * cell_size + (H_out + 1) * lw
            row_h = max(row_h, img_h_out)

        # Input grid cells
        x_origin = padding
        y_origin = y + (row_h - img_h_in) // 2
        for r in range(H_in):
            for c in range(W_in):
                cx = x_origin + c * (cell_size + lw) + lw + cell_size // 2
                cy = y_origin + r * (cell_size + lw) + lw + cell_size // 2
                cells.append((cx, cy, r, c, grid_id))
        grid_id += 1

        # Output grid cells (if present)
        if out_grid is not None:
            H_out = len(out_grid)
            W_out = len(out_grid[0]) if H_out > 0 else 0
            img_h_out = H_out * cell_size + (H_out + 1) * lw

            x_origin = padding + img_w_in + padding + arrow_w + padding
            y_origin = y + (row_h - img_h_out) // 2
            for r in range(H_out):
                for c in range(W_out):
                    cx = x_origin + c * (cell_size + lw) + lw + cell_size // 2
                    cy = y_origin + r * (cell_size + lw) + lw + cell_size // 2
                    cells.append((cx, cy, r, c, grid_id))
            grid_id += 1

        y += row_h + padding

    return cells


def compute_grid_dims(
    task: dict, d4_idx: int, test_idx: int,
) -> List[Tuple[int, int]]:
    """Get (H, W) for each grid in the task composite."""
    demos = task["train"]
    tests = task["test"]
    test_pair = tests[min(test_idx, len(tests) - 1)]

    dims = []
    for demo in demos:
        inp = apply_d4(demo["input"], d4_idx)
        dims.append((len(inp), len(inp[0]) if inp else 0))
        out = apply_d4(demo["output"], d4_idx)
        dims.append((len(out), len(out[0]) if out else 0))

    test_inp = apply_d4(test_pair["input"], d4_idx)
    dims.append((len(test_inp), len(test_inp[0]) if test_inp else 0))
    return dims


# ── Visual path: patch → grid cell mapping ───────────────────────────────────

def map_patches_to_cells(
    patch_positions: List[Tuple[int, int]],
    cell_positions: List[Tuple[int, int, int, int, int]],
) -> Dict[int, List[int]]:
    """Map vision patch indices to grid cell indices by nearest center.

    Args:
        patch_positions: [(px, py)] pixel center of each vision patch
        cell_positions: [(cx, cy, row, col, grid_id)] from compute_grid_cell_positions

    Returns:
        cell_idx → [patch_indices] mapping. Patches outside all cells are dropped.
    """
    cell_to_patches: Dict[int, List[int]] = {}

    for p_idx, (px, py) in enumerate(patch_positions):
        best_dist = float("inf")
        best_cell = -1
        for c_idx, (cx, cy, _r, _c, _gid) in enumerate(cell_positions):
            dist = abs(px - cx) + abs(py - cy)  # Manhattan
            if dist < best_dist:
                best_dist = dist
                best_cell = c_idx

        if best_cell >= 0:
            cell_to_patches.setdefault(best_cell, []).append(p_idx)

    return cell_to_patches


def extract_vision_patch_positions(
    model, processor, image_size: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Compute pixel center of each vision patch given model's patch configuration.

    Uses the model's vision config to determine patch size and spatial layout.
    Returns pixel centers in the original image coordinate system.

    Args:
        model: VLM model (Qwen3.5-9B or similar)
        image_size: (width, height) of the input image

    Returns:
        List of (px, py) pixel centers for each patch in sequence order.
    """
    visual = getattr(model.model, "visual", None) or getattr(model, "visual", None)
    if visual is None:
        raise ValueError("Model has no visual encoder")

    vision_config = getattr(visual, "config", None)
    if vision_config is None:
        vision_config = model.config.vision_config

    patch_size = getattr(vision_config, "patch_size", 14)
    # Qwen VL models use temporal_patch_size for the 3D conv
    temporal_patch = getattr(vision_config, "temporal_patch_size", 2)

    img_w, img_h = image_size

    # Number of patches in each dimension
    n_patches_w = img_w // patch_size
    n_patches_h = img_h // patch_size

    positions = []
    for py_idx in range(n_patches_h):
        for px_idx in range(n_patches_w):
            cx = px_idx * patch_size + patch_size // 2
            cy = py_idx * patch_size + patch_size // 2
            positions.append((cx, cy))

    return positions


# ── Text path: BPE token → grid cell mapping ────────────────────────────────

def map_tokens_to_cells_text(
    task: dict, d4_idx: int, test_idx: int,
    tokenizer,
) -> Tuple[List[int], List[Tuple[int, int, int]], List[Tuple[int, int]]]:
    """Map tokenized text positions to grid cells.

    Serializes the task, tokenizes, then maps each cell value token back to
    its grid position using the serializer's structured format.

    Returns:
        token_ids: full token id list
        cell_token_map: list of (token_idx, cell_idx) for cell-value tokens
        cell_coords: list of (row, col, grid_id) per cell
    """
    text = serialize_task(task, d4_idx=d4_idx, test_idx=test_idx)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Parse grid structure from serialized text
    demos = task["train"]
    tests = task["test"]
    test_pair = tests[min(test_idx, len(tests) - 1)]

    # Build grid list in serialization order
    grids = []
    for demo in demos:
        grids.append(("input", apply_d4(demo["input"], d4_idx)))
        grids.append(("output", apply_d4(demo["output"], d4_idx)))
    grids.append(("input", apply_d4(test_pair["input"], d4_idx)))

    # For each grid, build cell coordinates
    cell_coords = []  # (row, col, grid_id)
    grid_id = 0
    for _label, grid_data in grids:
        for r, row in enumerate(grid_data):
            for c, _val in enumerate(row):
                cell_coords.append((r, c, grid_id))
        grid_id += 1

    # Map tokens to cells using character offsets
    # Each cell value is a single digit (0-9) in the serialized text
    # Format: "0 8 0\n8 8 8" — digits separated by spaces, rows by newlines
    char_offsets = _compute_char_offsets(tokenizer, token_ids, text)

    # Find all cell-value character positions in the text
    cell_char_positions = []
    lines = text.split("\n")
    char_pos = 0
    cell_idx = 0

    for line in lines:
        line_start = char_pos
        # Detect grid data lines (lines that are just space-separated digits)
        stripped = line.strip()
        if stripped and all(c in "0123456789 " for c in stripped) and any(c.isdigit() for c in stripped):
            # This is a grid data line — find digit positions within this line
            strip_offset = line.index(stripped[0]) if stripped else 0
            pos_in_stripped = 0
            vals = stripped.split()
            for v in vals:
                offset_in_stripped = stripped.index(v, pos_in_stripped)
                abs_pos = line_start + strip_offset + offset_in_stripped
                cell_char_positions.append((abs_pos, cell_idx))
                cell_idx += 1
                pos_in_stripped = offset_in_stripped + len(v)
        char_pos += len(line) + 1  # +1 for newline

    # Map char positions to tokens
    cell_token_map = []
    for char_pos, cidx in cell_char_positions:
        if cidx >= len(cell_coords):
            break
        # Find which token covers this character
        for t_idx, (start, end) in enumerate(char_offsets):
            if start <= char_pos < end:
                cell_token_map.append((t_idx, cidx))
                break

    return token_ids, cell_token_map, cell_coords


def _compute_char_offsets(tokenizer, token_ids, text):
    """Compute (start, end) character offsets for each token.

    Prefers tokenizer's built-in offset mapping (fast tokenizers).
    Falls back to incremental decode matching for slow tokenizers.
    """
    # Try fast tokenizer offset mapping first
    try:
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        if "offset_mapping" in encoding and encoding["offset_mapping"] is not None:
            return encoding["offset_mapping"]
    except (TypeError, ValueError):
        pass

    # Fallback: incremental decode matching
    offsets = []
    pos = 0
    for tid in token_ids:
        decoded = tokenizer.decode([tid])
        idx = text.find(decoded, pos)
        if idx >= 0:
            offsets.append((idx, idx + len(decoded)))
            pos = idx + len(decoded)
        else:
            offsets.append((pos, pos + len(decoded)))
            pos += len(decoded)
    return offsets


# ── Similarity computation ───────────────────────────────────────────────────

def compute_cell_similarity(
    hidden_states: torch.Tensor,
    cell_to_token_map: Dict[int, List[int]],
    n_cells: int,
) -> torch.Tensor:
    """Compute pairwise cosine similarity between grid cell embeddings.

    Args:
        hidden_states: [seq_len, dim] hidden states from one layer
        cell_to_token_map: cell_idx → [token_indices]
        n_cells: total number of grid cells

    Returns:
        [n_cells, n_cells] cosine similarity matrix
    """
    dim = hidden_states.shape[-1]
    cell_embs = torch.zeros(n_cells, dim, device=hidden_states.device,
                            dtype=hidden_states.dtype)

    for cell_idx in range(n_cells):
        token_indices = cell_to_token_map.get(cell_idx, [])
        if token_indices:
            cell_embs[cell_idx] = hidden_states[token_indices].mean(dim=0)

    # Normalize for cosine similarity
    cell_embs_norm = F.normalize(cell_embs, p=2, dim=-1)
    similarity = cell_embs_norm @ cell_embs_norm.T  # [n_cells, n_cells]

    return similarity.float()


# ── Visual extraction ────────────────────────────────────────────────────────

def extract_visual_similarity(
    model, processor, task, d4_idx, test_idx, cell_size, layer_idx,
    device,
):
    """Extract similarity matrix via visual path.

    Returns (similarity [N,N], cell_coords [(r,c,gid),...], grid_dims [(H,W),...])
    or None if extraction fails.
    """
    render_task_composite, _, _, _, PROMPT_TEXT = _import_visual()

    # Render composite
    composite = render_task_composite(
        task, d4_idx=d4_idx, test_idx=test_idx, cell_size=cell_size,
    )

    # Get cell positions in image coordinates
    cell_positions = compute_grid_cell_positions(
        task, d4_idx, test_idx, cell_size=cell_size,
    )
    grid_dims = compute_grid_dims(task, d4_idx, test_idx)

    # Get vision patch positions
    patch_positions = extract_vision_patch_positions(model, processor, composite.size)

    # Map patches to cells
    cell_to_patches = map_patches_to_cells(patch_positions, cell_positions)

    # Build vision text input
    cfg = model.config
    tok = processor.tokenizer
    vis_start = tok.decode([cfg.vision_start_token_id])
    vis_end = tok.decode([cfg.vision_end_token_id])
    img_pad = tok.decode([cfg.image_token_id])
    chat_text = f"{vis_start}{img_pad}{vis_end}{PROMPT_TEXT}"

    # Process through VLM
    inputs = processor(
        text=[chat_text],
        images=[composite],
        return_tensors="pt",
    ).to(device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx]  # [1, seq_len, dim]
        hidden = hidden.squeeze(0)  # [seq_len, dim]

    # Find vision token range
    input_ids = inputs["input_ids"].squeeze(0)
    vision_start_id = cfg.vision_start_token_id
    vision_end_id = cfg.vision_end_token_id

    vis_start_pos = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
    vis_end_pos = (input_ids == vision_end_id).nonzero(as_tuple=True)[0]

    if len(vis_start_pos) == 0 or len(vis_end_pos) == 0:
        print(f"    WARN: No vision tokens found")
        return None

    vis_start_idx = vis_start_pos[0].item() + 1  # skip start token
    vis_end_idx = vis_end_pos[0].item()           # exclude end token
    vision_hidden = hidden[vis_start_idx:vis_end_idx]  # [n_patches, dim]

    # Map vision patches to cells
    n_cells = len(cell_positions)
    # Remap cell_to_patches: vision patches are 0-indexed within vision_hidden
    cell_to_vis_tokens = {}
    for c_idx, patch_indices in cell_to_patches.items():
        # patch_positions indices correspond to spatial grid positions
        # vision_hidden is the subset — need to map through spatial ordering
        valid = [p for p in patch_indices if p < vision_hidden.shape[0]]
        if valid:
            cell_to_vis_tokens[c_idx] = valid

    similarity = compute_cell_similarity(vision_hidden, cell_to_vis_tokens, n_cells)

    cell_coords = [(r, c, gid) for (_, _, r, c, gid) in cell_positions]

    return similarity.cpu(), cell_coords, grid_dims


# ── Text extraction ──────────────────────────────────────────────────────────

def extract_text_similarity(
    model, tokenizer, task, d4_idx, test_idx, layer_idx, device,
):
    """Extract similarity matrix via text path.

    Returns (similarity [N,N], cell_coords [(r,c,gid),...], grid_dims [(H,W),...])
    or None if extraction fails.
    """
    token_ids_list, cell_token_map, cell_coords = map_tokens_to_cells_text(
        task, d4_idx, test_idx, tokenizer,
    )
    grid_dims = compute_grid_dims(task, d4_idx, test_idx)

    n_cells = len(cell_coords)
    if n_cells == 0:
        return None

    # Forward pass
    input_ids = torch.tensor([token_ids_list], device=device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx].squeeze(0)  # [seq_len, dim]

    # Build cell → token map
    cell_to_tokens: Dict[int, List[int]] = {}
    for tok_idx, cell_idx in cell_token_map:
        cell_to_tokens.setdefault(cell_idx, []).append(tok_idx)

    similarity = compute_cell_similarity(hidden, cell_to_tokens, n_cells)

    return similarity.cpu(), cell_coords, grid_dims


# ── Main ─────────────────────────────────────────────────────────────────────

def precompute_similarities(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # Load model
    print(f"Loading model: {args.model_id}")
    if args.mode == "visual":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().to(device)
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )
        _, _, patch_vision_conv3d, load_arc_tasks_raw, _ = _import_visual()
        patch_vision_conv3d(model)
        tokenizer = processor.tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=True,
        )
        processor = None
        _, _, _, load_arc_tasks_raw, _ = _import_visual()

    for p in model.parameters():
        p.requires_grad = False

    # Detect number of hidden layers (varies by model class)
    if hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
    elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "num_hidden_layers"):
        n_layers = model.config.text_config.num_hidden_layers
    else:
        # Last resort: count layers in the language model
        lm = getattr(model.model, "language_model", model.model)
        n_layers = len(lm.layers)
    # hidden_states[0] = embeddings, hidden_states[n_layers] = last layer output
    layer_idx = args.layer_idx if args.layer_idx >= 0 else n_layers + args.layer_idx + 1
    # e.g., -1 → n_layers (last), -2 → n_layers-1 (second-to-last)
    print(f"Using layer {layer_idx} of {n_layers}")

    # Load ARC tasks
    print(f"Loading ARC tasks from {args.data_dir}")
    all_tasks = load_arc_tasks_raw(args.data_dir)
    n_train = len(all_tasks.get("train", []))
    n_eval = len(all_tasks.get("eval", []))
    print(f"  Train: {n_train}, Eval: {n_eval}")

    # Build work list
    d4_range = range(8) if args.d4_augment else range(1)
    work_items = []
    for split in ("train", "eval"):
        for task in all_tasks.get(split, []):
            for test_idx in range(len(task["test"])):
                for d4_idx in d4_range:
                    work_items.append((task, d4_idx, test_idx, split))

    if args.max_items > 0:
        work_items = work_items[:args.max_items]

    print(f"Total items: {len(work_items)}")

    # Extract similarities
    all_similarities = []
    all_cell_coords = []
    all_grid_dims = []
    all_task_ids = []
    all_d4_indices = []
    all_test_indices = []
    all_splits = []
    n_skipped = 0

    t0 = time.time()
    for idx, (task, d4_idx, test_idx, split) in enumerate(work_items):
        try:
            if args.mode == "visual":
                result = extract_visual_similarity(
                    model, processor, task, d4_idx, test_idx,
                    args.cell_size, layer_idx, device,
                )
            else:
                result = extract_text_similarity(
                    model, tokenizer, task, d4_idx, test_idx,
                    layer_idx, device,
                )
        except Exception as e:
            print(f"    WARN: {task['task_id']} d4={d4_idx} t={test_idx}: {e}")
            n_skipped += 1
            continue

        if result is None:
            n_skipped += 1
            continue

        similarity, cell_coords, grid_dims = result

        all_similarities.append(similarity)
        all_cell_coords.append(cell_coords)
        all_grid_dims.append(grid_dims)
        all_task_ids.append(task["task_id"])
        all_d4_indices.append(d4_idx)
        all_test_indices.append(test_idx)
        all_splits.append(split)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(work_items) - idx - 1) / rate
            print(f"  [{idx + 1}/{len(work_items)}] "
                  f"{rate:.2f} items/s, ETA {eta:.0f}s, "
                  f"skipped={n_skipped}")

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_similarities)} similarity matrices in {elapsed:.1f}s")
    print(f"  Skipped: {n_skipped}")

    if len(all_similarities) == 0:
        print("ERROR: No similarity matrices extracted!")
        return

    # Sanity check: within-grid > cross-grid similarity
    _sanity_check(all_similarities, all_cell_coords)

    # Save
    save_data = {
        "similarities": all_similarities,       # List[Tensor [N_i, N_i]]
        "cell_coords": all_cell_coords,          # List[List[(r,c,gid)]]
        "grid_dims": all_grid_dims,              # List[List[(H,W)]]
        "task_ids": all_task_ids,                # List[str]
        "d4_indices": torch.tensor(all_d4_indices, dtype=torch.long),
        "test_indices": torch.tensor(all_test_indices, dtype=torch.long),
        "splits": all_splits,                    # List[str]
        "model_id": args.model_id,
        "layer_idx": layer_idx,
        "mode": args.mode,
    }
    torch.save(save_data, args.output)
    file_size = os.path.getsize(args.output) / 1e6
    print(f"  Saved to {args.output} ({file_size:.1f} MB)")


def _sanity_check(similarities, cell_coords, n_samples=5):
    """Spot-check: within-grid similarity should exceed cross-grid similarity."""
    print("\n  Sanity check (within-grid vs cross-grid similarity):")
    for i in range(min(n_samples, len(similarities))):
        sim = similarities[i]
        coords = cell_coords[i]
        n = sim.shape[0]
        if n < 4:
            continue

        within_sum, within_count = 0.0, 0
        cross_sum, cross_count = 0.0, 0

        for a in range(n):
            for b in range(a + 1, n):
                val = sim[a, b].item()
                if coords[a][2] == coords[b][2]:  # same grid_id
                    within_sum += val
                    within_count += 1
                else:
                    cross_sum += val
                    cross_count += 1

        within_avg = within_sum / max(within_count, 1)
        cross_avg = cross_sum / max(cross_count, 1)
        print(f"    Sample {i}: within={within_avg:.4f}, cross={cross_avg:.4f}, "
              f"delta={within_avg - cross_avg:+.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Oracle similarity matrix extraction"
    )
    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen3.5-9B-Base",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to ARC-AGI data directory",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for similarity matrices .pt file",
    )
    parser.add_argument(
        "--mode", choices=["visual", "text"], default="visual",
        help="Extraction mode: visual (VLM) or text (causal LM)",
    )
    parser.add_argument(
        "--layer_idx", type=int, default=-1,
        help="Which model layer to extract from (-1 = last, -2 = second-to-last)",
    )
    parser.add_argument(
        "--d4_augment", action="store_true",
        help="Generate all 8 D4 variants per task",
    )
    parser.add_argument(
        "--cell_size", type=int, default=20,
        help="Pixel size per grid cell (visual mode, default: 20)",
    )
    parser.add_argument(
        "--max_items", type=int, default=0,
        help="Max items to process (0=all)",
    )
    args = parser.parse_args()

    precompute_similarities(args)


if __name__ == "__main__":
    main()
