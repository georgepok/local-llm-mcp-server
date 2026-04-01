"""ARC-AGI task — cell-as-token representation for geometric reasoning.

Each grid cell becomes one token with additive embeddings:
  h_cell = ColorEmbed(color) + PosEmbed_x(x) + PosEmbed_y(y) + RoleEmbed(role)

Sequence layout:
  [demo_in_1] sep [demo_out_1] sep [demo_in_2] sep [demo_out_2] ... sep [test_in] sep [test_out]

All test output cells predicted simultaneously (not autoregressive).

Augmentation:
  - D4 symmetry: 4 rotations x 2 reflections = 8x
  - Color permutation: randomly remap colors 1-9, keep 0 as background
  - Demo order shuffling: permute demo pair order
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch

# ARC grid constants
MAX_GRID_DIM = 30   # ARC grids are at most 30x30
N_COLORS = 10       # ARC uses colors 0-9
PAD_COLOR = 10      # padding token color index
PAD_COORD = 30      # padding coordinate index

# Role indices
ROLE_INPUT_DEMO = 0
ROLE_OUTPUT_DEMO = 1
ROLE_TEST_INPUT = 2
ROLE_TEST_OUTPUT = 3

# Separator types
SEP_DEMO_IO = 0       # between demo input and output
SEP_BETWEEN_DEMOS = 1 # between demo pairs
SEP_BEFORE_TEST_IN = 2
SEP_BEFORE_TEST_OUT = 3
N_SEP_TYPES = 4


def load_arc_tasks(data_dir: str) -> Dict[str, List[dict]]:
    """Load ARC-AGI tasks from JSON files.

    Args:
        data_dir: path to ARC-AGI data directory (contains training/ and evaluation/)

    Returns:
        {"train": [...], "eval": [...]} where each entry is a task dict with
        "train" (demo pairs) and "test" (test pairs) keys.
    """
    result = {}
    for split in ("training", "evaluation"):
        split_dir = os.path.join(data_dir, split)
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
        key = "train" if split == "training" else "eval"
        result[key] = tasks
    return result


# ─── D4 Symmetry Augmentation ───────────────────────────────────────────────

def rot90_grid(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 90 degrees clockwise."""
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    return [[grid[H - 1 - j][i] for j in range(H)] for i in range(W)]


def reflect_grid(grid: List[List[int]]) -> List[List[int]]:
    """Reflect grid horizontally (left-right)."""
    return [row[::-1] for row in grid]


def apply_d4(grid: List[List[int]], transform_idx: int) -> List[List[int]]:
    """Apply one of 8 D4 symmetry transforms (0-7).

    0-3: 0/90/180/270 rotation
    4-7: reflect + 0/90/180/270 rotation
    """
    g = grid
    n_rot = transform_idx % 4
    do_reflect = transform_idx >= 4

    if do_reflect:
        g = reflect_grid(g)
    for _ in range(n_rot):
        g = rot90_grid(g)
    return g


def apply_d4_to_pair(inp: List[List[int]], out: List[List[int]],
                     transform_idx: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Apply same D4 transform to input and output grid."""
    return apply_d4(inp, transform_idx), apply_d4(out, transform_idx)


# ─── Color Permutation ──────────────────────────────────────────────────────

def random_color_perm() -> Dict[int, int]:
    """Create a random permutation of colors 1-9, keeping 0 fixed."""
    colors = list(range(1, 10))
    shuffled = colors.copy()
    random.shuffle(shuffled)
    perm = {0: 0}
    for orig, new in zip(colors, shuffled):
        perm[orig] = new
    return perm


def apply_color_perm(grid: List[List[int]], perm: Dict[int, int]) -> List[List[int]]:
    """Apply color permutation to grid."""
    return [[perm.get(c, c) for c in row] for row in grid]


# ─── Sequence Building ──────────────────────────────────────────────────────

def grid_to_cells(grid: List[List[int]], role: int
                  ) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Convert grid to flat cell lists (row-major).

    Returns:
        (colors, xs, ys, roles) — each a list of ints, length H*W
    """
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    colors, xs, ys, roles = [], [], [], []
    for y in range(H):
        for x in range(W):
            colors.append(grid[y][x])
            xs.append(x)
            ys.append(y)
            roles.append(role)
    return colors, xs, ys, roles


def build_sequence(task: dict, d4_idx: int = 0,
                   color_perm: Optional[Dict[int, int]] = None,
                   demo_order: Optional[List[int]] = None,
                   test_idx: int = 0,
                   max_seq_len: int = 1024
                   ) -> Optional[Dict[str, torch.Tensor]]:
    """Build cell-as-token sequence from an ARC task.

    Args:
        task: ARC task dict with "train" and "test" keys
        d4_idx: D4 symmetry transform index (0-7)
        color_perm: color permutation dict (None = identity)
        demo_order: permutation of demo indices (None = original order)
        test_idx: which test pair to use
        max_seq_len: maximum sequence length

    Returns:
        Dict with:
          "colors": [N] int tensor — color indices (0-10)
          "xs": [N] int tensor — x coordinates (0-30)
          "ys": [N] int tensor — y coordinates (0-30)
          "roles": [N] int tensor — role indices (0-3)
          "sep_mask": [N] bool tensor — True for separator positions
          "sep_types": [N] int tensor — separator type at each position
          "target_colors": [N_out] int tensor — target colors for test output cells
          "target_mask": [N] bool tensor — True for test output positions
          "context_mask": [N] bool tensor — True for non-test-output positions
          "grid_positions": [N, 2] float tensor — (x, y) for grid distance matrix
          "grid_ids": [N] int tensor — which grid each token belongs to (-1 for sep)
          "test_output_shape": (H, W) — shape of test output grid
        Or None if sequence exceeds max_seq_len.
    """
    demos = task["train"]
    tests = task["test"]
    if test_idx >= len(tests):
        test_idx = 0

    # Determine demo order
    n_demos = len(demos)
    if demo_order is None:
        demo_order = list(range(n_demos))

    # Build full sequence
    all_colors, all_xs, all_ys, all_roles = [], [], [], []
    sep_mask, sep_types = [], []
    grid_ids = []
    grid_counter = 0

    def add_separator(sep_type: int):
        all_colors.append(PAD_COLOR)
        all_xs.append(PAD_COORD)
        all_ys.append(PAD_COORD)
        all_roles.append(0)
        sep_mask.append(True)
        sep_types.append(sep_type)
        grid_ids.append(-1)

    def add_grid(grid: List[List[int]], role: int):
        nonlocal grid_counter
        # Apply augmentations
        g = apply_d4(grid, d4_idx)
        if color_perm is not None:
            g = apply_color_perm(g, color_perm)

        colors, xs, ys, roles = grid_to_cells(g, role)
        n = len(colors)
        all_colors.extend(colors)
        all_xs.extend(xs)
        all_ys.extend(ys)
        all_roles.extend(roles)
        sep_mask.extend([False] * n)
        sep_types.extend([0] * n)
        grid_ids.extend([grid_counter] * n)
        grid_counter += 1
        return g  # return augmented grid

    # Add demo pairs
    for i, demo_i in enumerate(demo_order):
        demo = demos[demo_i]
        if i > 0:
            add_separator(SEP_BETWEEN_DEMOS)
        add_grid(demo["input"], ROLE_INPUT_DEMO)
        add_separator(SEP_DEMO_IO)
        add_grid(demo["output"], ROLE_OUTPUT_DEMO)

    # Add test input
    add_separator(SEP_BEFORE_TEST_IN)
    test_pair = tests[test_idx]
    aug_test_input = add_grid(test_pair["input"], ROLE_TEST_INPUT)

    # Add test output
    add_separator(SEP_BEFORE_TEST_OUT)
    test_output_start = len(all_colors)
    aug_test_output = add_grid(test_pair["output"], ROLE_TEST_OUTPUT)
    test_output_end = len(all_colors)

    # For each test_output cell at (x, y), look up test_input color at (x, y).
    # This gives test_output positions informative content (what the cell looks
    # like in the input) instead of PAD_COLOR, preventing geodesic barriers.
    test_in_H = len(aug_test_input)
    test_in_W = len(aug_test_input[0]) if test_in_H > 0 else 0
    input_colors_for_output = []
    for y_out in range(len(aug_test_output)):
        for x_out in range(len(aug_test_output[0]) if len(aug_test_output) > 0 else 0):
            if y_out < test_in_H and x_out < test_in_W:
                input_colors_for_output.append(aug_test_input[y_out][x_out])
            else:
                input_colors_for_output.append(0)  # background color for out-of-bounds

    N = len(all_colors)
    if N > max_seq_len:
        return None

    # Build tensors
    colors_t = torch.tensor(all_colors, dtype=torch.long)
    xs_t = torch.tensor(all_xs, dtype=torch.long)
    ys_t = torch.tensor(all_ys, dtype=torch.long)
    roles_t = torch.tensor(all_roles, dtype=torch.long)
    sep_mask_t = torch.tensor(sep_mask, dtype=torch.bool)
    sep_types_t = torch.tensor(sep_types, dtype=torch.long)
    grid_ids_t = torch.tensor(grid_ids, dtype=torch.long)

    # Target mask: which positions need prediction
    target_mask_t = torch.zeros(N, dtype=torch.bool)
    target_mask_t[test_output_start:test_output_end] = True

    # Target colors
    target_colors = colors_t[target_mask_t].clone()

    # Context mask: everything that isn't test output
    context_mask_t = ~target_mask_t

    # Grid positions for structural energy (x, y coordinates as floats)
    grid_positions = torch.zeros(N, 2, dtype=torch.float32)
    for i in range(N):
        if not sep_mask[i]:
            grid_positions[i, 0] = float(all_xs[i])
            grid_positions[i, 1] = float(all_ys[i])

    # Test output shape
    test_out_H = len(aug_test_output)
    test_out_W = len(aug_test_output[0]) if test_out_H > 0 else 0

    return {
        "colors": colors_t,
        "xs": xs_t,
        "ys": ys_t,
        "roles": roles_t,
        "sep_mask": sep_mask_t,
        "sep_types": sep_types_t,
        "target_colors": target_colors,
        "target_mask": target_mask_t,
        "context_mask": context_mask_t,
        "grid_positions": grid_positions,
        "grid_ids": grid_ids_t,
        "input_colors_for_output": torch.tensor(input_colors_for_output, dtype=torch.long),
        "test_output_shape": (test_out_H, test_out_W),
    }


def compute_grid_distances(grid_positions: torch.Tensor,
                           grid_ids: torch.Tensor,
                           sep_mask: torch.Tensor) -> torch.Tensor:
    """Compute 2D Euclidean distance matrix within grids.

    Distances between tokens in different grids or at separator positions
    are set to infinity.

    Args:
        grid_positions: [N, 2] (x, y) coordinates
        grid_ids: [N] which grid each token belongs to (-1 for seps)
        sep_mask: [N] True for separator positions

    Returns:
        D_struct: [N, N] normalized distance matrix
    """
    N = grid_positions.shape[0]

    # Pairwise Euclidean distance
    diff = grid_positions.unsqueeze(1) - grid_positions.unsqueeze(0)  # [N, N, 2]
    D = (diff * diff).sum(-1).sqrt()  # [N, N]

    # Find max dimension for normalization
    max_coord = grid_positions.max()
    if max_coord > 0:
        D = D / max_coord

    # Mask: infinity between different grids and at separator positions
    same_grid = grid_ids.unsqueeze(1) == grid_ids.unsqueeze(0)  # [N, N]
    valid = same_grid & (~sep_mask.unsqueeze(1)) & (~sep_mask.unsqueeze(0))
    # Also exclude grid_id == -1 (separators)
    not_sep_grid = (grid_ids >= 0).unsqueeze(1) & (grid_ids >= 0).unsqueeze(0)
    valid = valid & not_sep_grid

    D = D.masked_fill(~valid, float('inf'))
    # Self-distance = 0
    D.fill_diagonal_(0.0)

    return D


# ─── ARC Task Class ─────────────────────────────────────────────────────────

class ARCTask:
    """ARC-AGI task for FluidNet/flat transformer training.

    Unlike other FGN tasks, this does NOT use a tokenizer. Instead it produces
    raw embedding indices (colors, xs, ys, roles) that the model embeds directly.

    generate_batch() returns:
      - input_ids: NOT used (set to zeros placeholder)
      - labels: NOT used (set to -100 placeholder)
      - metadata: Dict with all ARC-specific tensors
    """

    def __init__(self, tokenizer=None, seq_len: int = 1024,
                 data_dir: str = "data/arc",
                 split: str = "train",
                 n_color_perms: int = 10,
                 augment: bool = True,
                 **kwargs):
        """Initialize ARC task.

        Args:
            tokenizer: ignored (kept for interface compatibility)
            seq_len: maximum sequence length
            data_dir: path to ARC-AGI data directory
            split: "train" or "eval"
            n_color_perms: number of color permutations per task per epoch
            augment: whether to apply augmentation
        """
        self.seq_len = seq_len
        self.augment = augment
        self.n_color_perms = n_color_perms

        # Load tasks
        all_tasks = load_arc_tasks(data_dir)
        if split == "train":
            self.tasks = all_tasks.get("train", [])
        else:
            self.tasks = all_tasks.get("eval", [])

        if len(self.tasks) == 0:
            raise ValueError(f"No ARC tasks found in {data_dir}/{split}. "
                             f"Download from https://github.com/fchollet/ARC-AGI")

        print(f"  ARC: loaded {len(self.tasks)} {split} tasks, "
              f"augment={augment}, max_seq_len={seq_len}")

    def _sample_augmented(self, task: dict) -> Optional[Dict[str, torch.Tensor]]:
        """Sample one augmented version of a task."""
        if self.augment:
            d4_idx = random.randint(0, 7)
            color_perm = random_color_perm()
            demo_order = list(range(len(task["train"])))
            random.shuffle(demo_order)
        else:
            d4_idx = 0
            color_perm = None
            demo_order = None

        # Pick a random test pair
        test_idx = random.randint(0, len(task["test"]) - 1)

        return build_sequence(
            task, d4_idx=d4_idx, color_perm=color_perm,
            demo_order=demo_order, test_idx=test_idx,
            max_seq_len=self.seq_len,
        )

    def generate_batch(self, batch_size: int,
                       device: Optional[torch.device] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Generate a batch of augmented ARC sequences.

        Returns:
            (input_ids, labels, metadata)
            - input_ids: [B, max_N] zeros (not used — model uses metadata)
            - labels: [B, max_N] -100 (not used — model uses metadata)
            - metadata: Dict with padded tensors:
                "colors": [B, max_N] color indices
                "xs": [B, max_N] x coordinates
                "ys": [B, max_N] y coordinates
                "roles": [B, max_N] role indices
                "sep_mask": [B, max_N] separator mask
                "target_mask": [B, max_N] test output positions
                "context_mask": [B, max_N] non-test-output positions
                "target_colors": [B, max_out] target color classes
                "lengths": [B] actual sequence length per item
                "n_targets": [B] number of target cells per item
                "grid_distances": [B, max_N, max_N] structural distance matrix
                "test_output_shapes": [(H, W), ...] list of output shapes
        """
        if device is None:
            device = torch.device("cpu")

        samples = []
        attempts = 0
        max_attempts = batch_size * 20

        while len(samples) < batch_size and attempts < max_attempts:
            task = random.choice(self.tasks)
            seq = self._sample_augmented(task)
            attempts += 1
            if seq is not None:
                samples.append(seq)

        if len(samples) == 0:
            raise RuntimeError("Could not generate any valid ARC sequences "
                               f"within max_seq_len={self.seq_len}")

        # Pad to batch_size if needed (repeat last sample)
        while len(samples) < batch_size:
            samples.append(samples[-1])

        # Pad to FIXED seq_len for torch.compile (no recompilation)
        max_N = self.seq_len
        max_out = max(s["target_colors"].shape[0] for s in samples)

        B = batch_size
        colors = torch.full((B, max_N), PAD_COLOR, dtype=torch.long)
        xs = torch.full((B, max_N), PAD_COORD, dtype=torch.long)
        ys = torch.full((B, max_N), PAD_COORD, dtype=torch.long)
        roles = torch.zeros(B, max_N, dtype=torch.long)
        sep_mask = torch.ones(B, max_N, dtype=torch.bool)  # pad = separator-like
        target_mask = torch.zeros(B, max_N, dtype=torch.bool)
        context_mask = torch.zeros(B, max_N, dtype=torch.bool)
        target_colors_pad = torch.full((B, max_out), -100, dtype=torch.long)
        lengths = torch.zeros(B, dtype=torch.long)
        n_targets = torch.zeros(B, dtype=torch.long)
        test_output_shapes = []

        # Build a flat target_labels tensor aligned with sequence positions.
        # target_labels[b, pos] = color if pos is a target position, else -100.
        # This allows vectorized loss computation without per-sample loops.
        target_labels = torch.full((B, max_N), -100, dtype=torch.long)

        for i, s in enumerate(samples):
            N_i = s["colors"].shape[0]
            N_out_i = s["target_colors"].shape[0]

            colors[i, :N_i] = s["colors"]
            xs[i, :N_i] = s["xs"]
            ys[i, :N_i] = s["ys"]
            roles[i, :N_i] = s["roles"]
            sep_mask[i, :N_i] = s["sep_mask"]
            target_mask[i, :N_i] = s["target_mask"]
            context_mask[i, :N_i] = s["context_mask"]
            target_colors_pad[i, :N_out_i] = s["target_colors"]
            lengths[i] = N_i
            n_targets[i] = N_out_i
            test_output_shapes.append(s["test_output_shape"])

            # Place target colors at their sequence positions
            target_positions = s["target_mask"].nonzero(as_tuple=True)[0]
            target_labels[i, target_positions] = s["target_colors"]

        # Placeholder input_ids and labels (model doesn't use these)
        input_ids = torch.zeros(B, max_N, dtype=torch.long, device=device)
        labels = torch.full((B, max_N), -100, dtype=torch.long, device=device)

        # Grid IDs for structural energy (which grid each token belongs to)
        grid_ids = torch.full((B, max_N), -1, dtype=torch.long)
        for i, s in enumerate(samples):
            N_i = s["colors"].shape[0]
            grid_ids[i, :N_i] = s["grid_ids"]

        # Input colors at target positions (test_input colors at test_output positions)
        # This replaces PAD_COLOR masking — test_output positions get the corresponding
        # test_input color at the same (x,y) position, making them informationally rich.
        target_input_colors = torch.full((B, max_N), PAD_COLOR, dtype=torch.long)
        for i, s in enumerate(samples):
            target_positions = s["target_mask"].nonzero(as_tuple=True)[0]
            target_input_colors[i, target_positions] = s["input_colors_for_output"]

        metadata = {
            "colors": colors.to(device),
            "xs": xs.to(device),
            "ys": ys.to(device),
            "roles": roles.to(device),
            "sep_mask": sep_mask.to(device),
            "sep_types": torch.zeros(B, max_N, dtype=torch.long, device=device),
            "target_mask": target_mask.to(device),
            "context_mask": context_mask.to(device),
            "target_colors": target_colors_pad.to(device),
            "target_labels": target_labels.to(device),
            "lengths": lengths.to(device),
            "n_targets": n_targets.to(device),
            "grid_ids": grid_ids.to(device),
            "target_input_colors": target_input_colors.to(device),
            "test_output_shapes": test_output_shapes,
        }

        return input_ids, labels, metadata


# ─── TTT Utilities ─────────────────────────────────────────────────────────

def pad_single_to_batch(seq: Dict[str, torch.Tensor], max_seq_len: int,
                        device: torch.device) -> Dict[str, torch.Tensor]:
    """Pad a single build_sequence() output to [1, max_seq_len] batch format.

    Returns metadata dict compatible with model forward().
    """
    N = seq["colors"].shape[0]

    colors = torch.full((1, max_seq_len), PAD_COLOR, dtype=torch.long)
    xs = torch.full((1, max_seq_len), PAD_COORD, dtype=torch.long)
    ys = torch.full((1, max_seq_len), PAD_COORD, dtype=torch.long)
    roles = torch.zeros(1, max_seq_len, dtype=torch.long)
    sep_mask = torch.ones(1, max_seq_len, dtype=torch.bool)
    target_mask = torch.zeros(1, max_seq_len, dtype=torch.bool)
    context_mask = torch.zeros(1, max_seq_len, dtype=torch.bool)
    target_labels = torch.full((1, max_seq_len), -100, dtype=torch.long)
    grid_ids = torch.full((1, max_seq_len), -1, dtype=torch.long)
    target_input_colors = torch.full((1, max_seq_len), PAD_COLOR, dtype=torch.long)

    colors[0, :N] = seq["colors"]
    xs[0, :N] = seq["xs"]
    ys[0, :N] = seq["ys"]
    roles[0, :N] = seq["roles"]
    sep_mask[0, :N] = seq["sep_mask"]
    target_mask[0, :N] = seq["target_mask"]
    context_mask[0, :N] = seq["context_mask"]
    grid_ids[0, :N] = seq["grid_ids"]

    # Place target colors at their sequence positions
    target_positions = seq["target_mask"].nonzero(as_tuple=True)[0]
    target_labels[0, target_positions] = seq["target_colors"]

    # Input colors at target positions
    target_input_colors[0, target_positions] = seq["input_colors_for_output"]

    return {
        "colors": colors.to(device),
        "xs": xs.to(device),
        "ys": ys.to(device),
        "roles": roles.to(device),
        "sep_mask": sep_mask.to(device),
        "sep_types": torch.zeros(1, max_seq_len, dtype=torch.long, device=device),
        "target_mask": target_mask.to(device),
        "context_mask": context_mask.to(device),
        "target_labels": target_labels.to(device),
        "lengths": torch.tensor([N], dtype=torch.long, device=device),
        "n_targets": torch.tensor([len(target_positions)], dtype=torch.long, device=device),
        "grid_ids": grid_ids.to(device),
        "target_input_colors": target_input_colors.to(device),
    }


def invert_d4(grid: List[List[int]], transform_idx: int) -> List[List[int]]:
    """Invert a D4 transform to recover original orientation.

    D4 group inverses:
      rot0 → rot0, rot90 → rot270, rot180 → rot180, rot270 → rot90
      reflect+rotN → reflect+rotN (reflections are self-inverse after rotation)
    """
    do_reflect = transform_idx >= 4
    n_rot = transform_idx % 4

    g = grid
    if do_reflect:
        # Undo rotation first (inverse of N rotations = 4-N rotations)
        for _ in range((4 - n_rot) % 4):
            g = rot90_grid(g)
        g = reflect_grid(g)
    else:
        for _ in range((4 - n_rot) % 4):
            g = rot90_grid(g)
    return g


def invert_color_perm(perm: Dict[int, int]) -> Dict[int, int]:
    """Invert a color permutation mapping."""
    return {v: k for k, v in perm.items()}


def reconstruct_grid(preds: torch.Tensor, shape: Tuple[int, int]) -> List[List[int]]:
    """Convert flat prediction tensor back to 2D grid (row-major).

    Args:
        preds: [H*W] tensor of predicted color indices
        shape: (H, W) grid dimensions
    """
    H, W = shape
    grid = []
    for y in range(H):
        row = []
        for x in range(W):
            row.append(preds[y * W + x].item())
        grid.append(row)
    return grid


if __name__ == "__main__":
    print("Testing ARC task...")

    # Create a small synthetic task for testing
    test_task = {
        "train": [
            {"input": [[0, 1, 2], [3, 4, 5]], "output": [[5, 4, 3], [2, 1, 0]]},
            {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
        ],
        "test": [
            {"input": [[2, 3], [4, 5]], "output": [[5, 4], [3, 2]]},
        ],
        "task_id": "test_task",
    }

    # Test sequence building
    seq = build_sequence(test_task, d4_idx=0, max_seq_len=1024)
    assert seq is not None
    N = seq["colors"].shape[0]
    print(f"  Sequence length: {N}")
    print(f"  Target cells: {seq['target_colors'].shape[0]}")
    print(f"  Test output shape: {seq['test_output_shape']}")

    # Test D4 augmentation
    grid = [[1, 2], [3, 4]]
    for i in range(8):
        g = apply_d4(grid, i)
        print(f"  D4[{i}]: {g}")

    # Test color permutation
    perm = random_color_perm()
    print(f"  Color perm: {perm}")

    # Test grid distances
    D = compute_grid_distances(seq["grid_positions"], seq["grid_ids"], seq["sep_mask"])
    finite_count = (D < float('inf')).sum()
    print(f"  Grid distances: {D.shape}, finite entries: {finite_count}")

    print("ARC task OK")
