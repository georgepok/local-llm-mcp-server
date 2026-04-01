"""Visual oracle embedding extraction — renders ARC grids as images for VLM.

Renders ARC task grids as colored images using the official ARC palette,
composes demo pairs + test input into a single composite image, and feeds
through Qwen3-VL for fused vision-language embeddings.

The key insight: BPE tokenizers destroy 2D spatial structure. The vision
encoder was literally built to parse it. Feed grids as images, not strings.

Usage:
    python scripts/precompute_visual.py \
        --model_id Qwen/Qwen3-VL-8B-Instruct \
        --data_dir /workspace/fgn-v3/data/arc \
        --output /workspace/latent-oracle/embeddings_visual.pt \
        --d4_augment
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
from PIL import Image, ImageDraw


def patch_vision_conv3d(model):
    """Monkey-patch the vision encoder's Conv3d to disable cuDNN only for that
    one layer. Conv3d in Qwen3.5 vision patch_embed hits a cuDNN engine bug on
    Grace Blackwell (GB10). The rest of the model keeps cuDNN enabled."""
    # Find the patch_embed's Conv3d proj layer
    visual = getattr(model.model, "visual", None)
    if visual is None:
        return
    patch_embed = getattr(visual, "patch_embed", None)
    if patch_embed is None:
        return
    proj = getattr(patch_embed, "proj", None)
    if proj is None or not isinstance(proj, torch.nn.Conv3d):
        return

    original_forward = proj.forward

    def safe_forward(*args, **kwargs):
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            return original_forward(*args, **kwargs)
        finally:
            torch.backends.cudnn.enabled = prev

    proj.forward = safe_forward
    print("Patched vision Conv3d to bypass cuDNN")

# ── ARC official color palette (from ARC-AGI testing_interface CSS) ───────────

ARC_COLORS = {
    0: (0, 0, 0),        # black   #000000
    1: (0, 116, 217),    # blue    #0074D9
    2: (255, 65, 54),    # red     #FF4136
    3: (46, 204, 64),    # green   #2ECC40
    4: (255, 220, 0),    # yellow  #FFDC00
    5: (170, 170, 170),  # grey    #AAAAAA
    6: (240, 18, 190),   # fuchsia #F012BE
    7: (255, 133, 27),   # orange  #FF851B
    8: (127, 219, 255),  # teal    #7FDBFF
    9: (135, 12, 37),    # brown   #870C25
}

GRID_LINE_COLOR = (50, 50, 50)
COMPOSITE_BG = (245, 245, 245)
ARROW_COLOR = (80, 80, 80)


# ── Grid rendering ────────────────────────────────────────────────────────────

def render_grid(grid: List[List[int]], cell_size: int = 20) -> Image.Image:
    """Render a single ARC grid as a PIL image with colored cells."""
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    lw = 1  # grid line width
    img_w = W * cell_size + (W + 1) * lw
    img_h = H * cell_size + (H + 1) * lw

    img = Image.new("RGB", (img_w, img_h), GRID_LINE_COLOR)
    draw = ImageDraw.Draw(img)

    for r in range(H):
        for c in range(W):
            x0 = c * (cell_size + lw) + lw
            y0 = r * (cell_size + lw) + lw
            color = ARC_COLORS.get(grid[r][c], (128, 128, 128))
            draw.rectangle(
                [x0, y0, x0 + cell_size - 1, y0 + cell_size - 1],
                fill=color,
            )

    return img


# ── D4 symmetry (inlined to avoid latent_oracle imports) ─────────────────────

def rot90_grid(grid: List[List[int]]) -> List[List[int]]:
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    return [[grid[H - 1 - j][i] for j in range(H)] for i in range(W)]


def reflect_grid(grid: List[List[int]]) -> List[List[int]]:
    return [row[::-1] for row in grid]


def apply_d4(grid: List[List[int]], d4_idx: int) -> List[List[int]]:
    g = grid
    if d4_idx >= 4:
        g = reflect_grid(g)
    for _ in range(d4_idx % 4):
        g = rot90_grid(g)
    return g


# ── Task composite rendering ─────────────────────────────────────────────────

def render_task_composite(
    task: dict,
    d4_idx: int = 0,
    test_idx: int = 0,
    cell_size: int = 20,
    padding: int = 12,
    arrow_w: int = 30,
) -> Image.Image:
    """Render full ARC task as a single composite image.

    Layout (one row per example):
      [Demo 1 Input] → [Demo 1 Output]
      [Demo 2 Input] → [Demo 2 Output]
      ...
      [Test Input]
    """
    demos = task["train"]
    tests = task["test"]
    test_pair = tests[min(test_idx, len(tests) - 1)]

    # Render all grid pairs
    rows: List[Tuple[Image.Image, Optional[Image.Image]]] = []
    for demo in demos:
        inp = render_grid(apply_d4(demo["input"], d4_idx), cell_size)
        out = render_grid(apply_d4(demo["output"], d4_idx), cell_size)
        rows.append((inp, out))

    test_inp = render_grid(apply_d4(test_pair["input"], d4_idx), cell_size)
    rows.append((test_inp, None))

    # Calculate composite dimensions
    total_w = 0
    total_h = padding

    for in_img, out_img in rows:
        row_w = padding + in_img.width
        if out_img is not None:
            row_w += padding + arrow_w + padding + out_img.width
        row_w += padding
        total_w = max(total_w, row_w)

        row_h = in_img.height
        if out_img is not None:
            row_h = max(row_h, out_img.height)
        total_h += row_h + padding

    # Draw composite
    composite = Image.new("RGB", (total_w, total_h), COMPOSITE_BG)
    draw = ImageDraw.Draw(composite)
    y = padding

    for in_img, out_img in rows:
        row_h = in_img.height
        if out_img is not None:
            row_h = max(row_h, out_img.height)

        # Input grid (vertically centered in row)
        x = padding
        composite.paste(in_img, (x, y + (row_h - in_img.height) // 2))
        x += in_img.width + padding

        if out_img is not None:
            # Arrow
            ay = y + row_h // 2
            draw.line([(x, ay), (x + arrow_w - 8, ay)], fill=ARROW_COLOR, width=2)
            draw.polygon(
                [
                    (x + arrow_w - 8, ay - 5),
                    (x + arrow_w, ay),
                    (x + arrow_w - 8, ay + 5),
                ],
                fill=ARROW_COLOR,
            )
            x += arrow_w + padding

            # Output grid
            composite.paste(out_img, (x, y + (row_h - out_img.height) // 2))

        y += row_h + padding

    return composite


# ── Data loading ──────────────────────────────────────────────────────────────

def load_arc_tasks_raw(data_dir: str):
    """Load raw ARC task dicts from JSON files."""
    result = {}
    for split_name, dir_name in [("train", "training"), ("eval", "evaluation")]:
        split_dir = os.path.join(data_dir, dir_name)
        if not os.path.isdir(split_dir):
            continue
        tasks = []
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(split_dir, fname)) as f:
                task = json.load(f)
            task["task_id"] = fname.replace(".json", "")
            tasks.append(task)
        result[split_name] = tasks
    return result


# ── Main embedding extraction ─────────────────────────────────────────────────

PROMPT_TEXT = "Identify the spatial transformation rule shown in this puzzle."


def precompute_visual(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load VLM + processor (Qwen3.5-9B is a VLM — Qwen3_5ForConditionalGeneration)
    print(f"Loading VLM: {args.model_id}")
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    print(f"Model class: {type(model).__name__}")
    patch_vision_conv3d(model)

    oracle_dim = model.config.text_config.hidden_size
    for p in model.parameters():
        p.requires_grad = False

    print(f"Oracle dim: {oracle_dim}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

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

    print(f"Total items to embed: {len(work_items)}")

    # Build vision text template: <|vision_start|><|image_pad|><|vision_end|> + prompt
    # Base model has no chat template — construct directly from special tokens.
    cfg = model.config
    tok = processor.tokenizer
    vis_start = tok.decode([cfg.vision_start_token_id])
    vis_end = tok.decode([cfg.vision_end_token_id])
    img_pad = tok.decode([cfg.image_token_id])
    chat_text = f"{vis_start}{img_pad}{vis_end}{PROMPT_TEXT}"
    print(f"Vision text template: {chat_text[:80]}...")

    # Process items one at a time (VLM is memory-intensive)
    all_embeddings = []
    all_task_ids = []
    all_d4_indices = []
    all_test_indices = []
    all_splits = []

    t0 = time.time()

    for idx, (task, d4_idx, test_idx, split) in enumerate(work_items):
        # Render composite image
        composite = render_task_composite(
            task, d4_idx=d4_idx, test_idx=test_idx,
            cell_size=args.cell_size,
        )

        # Process through VLM processor
        inputs = processor(
            text=[chat_text],
            images=[composite],
            return_tensors="pt",
        ).to(device)
        inputs.pop("token_type_ids", None)

        # Forward pass — extract hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # [1, seq_len, oracle_dim]

        # Pool embedding
        if args.pooling == "mean":
            if "attention_mask" in inputs:
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden.mean(dim=1)
        else:  # "last" — last token (EOS-style, recommended by Qwen-VL-Embedding)
            if "attention_mask" in inputs:
                # Find last non-pad position
                seq_lens = inputs["attention_mask"].sum(dim=1) - 1  # [B]
                pooled = hidden[0, seq_lens[0]].unsqueeze(0)  # [1, oracle_dim]
            else:
                pooled = hidden[:, -1, :]  # [1, oracle_dim]

        all_embeddings.append(pooled.cpu())
        all_task_ids.append(task["task_id"])
        all_d4_indices.append(d4_idx)
        all_test_indices.append(test_idx)
        all_splits.append(split)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(work_items) - idx - 1) / rate
            print(f"  [{idx + 1}/{len(work_items)}] "
                  f"{rate:.2f} items/s, ETA {eta:.0f}s")

    # Concatenate and save
    embeddings = torch.cat(all_embeddings, dim=0).float()  # [N, oracle_dim]
    d4_tensor = torch.tensor(all_d4_indices, dtype=torch.long)
    test_tensor = torch.tensor(all_test_indices, dtype=torch.long)

    elapsed = time.time() - t0
    print(f"\nDone: {embeddings.shape[0]} embeddings in {elapsed:.1f}s")
    print(f"  Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"  Size: {embeddings.numel() * 4 / 1e6:.1f} MB (float32)")

    save_data = {
        "embeddings": embeddings,       # [N, oracle_dim] float32
        "task_ids": all_task_ids,        # list[str]
        "d4_indices": d4_tensor,         # [N] long
        "test_indices": test_tensor,     # [N] long
        "splits": all_splits,            # list[str]
        "oracle_dim": oracle_dim,        # int
        "model_id": args.model_id,       # str
    }
    torch.save(save_data, args.output)
    file_size = os.path.getsize(args.output) / 1e6
    print(f"  Saved to {args.output} ({file_size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Visual oracle embedding extraction via Qwen3-VL"
    )
    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen3.5-9B-Base",
        help="HuggingFace VLM model ID (default: Qwen3.5-9B-Base)",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to ARC-AGI data directory",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for visual embeddings .pt file",
    )
    parser.add_argument(
        "--d4_augment", action="store_true",
        help="Generate all 8 D4 variants per task",
    )
    parser.add_argument(
        "--cell_size", type=int, default=20,
        help="Pixel size per grid cell (default: 20)",
    )
    parser.add_argument(
        "--pooling", choices=["mean", "last"], default="mean",
        help="Embedding pooling: mean-pool or last-token (default: mean)",
    )
    parser.add_argument(
        "--max_items", type=int, default=0,
        help="Max items to process (0=all, useful for smoke test)",
    )
    args = parser.parse_args()

    precompute_visual(args)


if __name__ == "__main__":
    main()
