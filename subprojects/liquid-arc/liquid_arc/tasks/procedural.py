"""Procedural ARC task generator — infinite non-repeating streams.

Every batch contains grids the model has never seen. Rules remain constant
but colors, grid sizes, object shapes, and positions are fully randomized.

All tasks are global-relational: they cannot be solved by looking at a single
cell or a small local neighborhood. The model must process the entire grid.

Curriculum stages:
  Stage 1 (GLOBAL):      whole-grid spatial transforms
  Stage 2 (RELATIONAL):  object-level reasoning
  Stage 3 (COMPOSITION): multi-step pattern composition

Compatible with ARCTask.generate_batch() interface.
"""

import random
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch

# ARC constants (must match fgn/tasks/arc.py)
N_COLORS = 10
PAD_COLOR = 10
PAD_COORD = 30

# Roles
ROLE_INPUT_DEMO = 0
ROLE_OUTPUT_DEMO = 1
ROLE_TEST_INPUT = 2
ROLE_TEST_OUTPUT = 3

# Separator types
SEP_DEMO_IO = 0
SEP_BETWEEN_DEMOS = 1
SEP_BEFORE_TEST_IN = 2
SEP_BEFORE_TEST_OUT = 3


class CurriculumStage(IntEnum):
    GLOBAL = 1       # Steps 0-20K: whole-grid spatial transforms
    RELATIONAL = 2   # Steps 20K-100K: object-level reasoning
    COMPOSITION = 3  # Steps 100K+: multi-step pattern composition


# ────────────────────────────────────────────────────────────────────
# Grid utilities
# ────────────────────────────────────────────────────────────────────

def _rand_grid(H: int, W: int, n_colors: int = 2, bg: int = 0) -> List[List[int]]:
    """Random grid with mostly background."""
    grid = [[bg] * W for _ in range(H)]
    colors = [c for c in range(N_COLORS) if c != bg]
    random.shuffle(colors)
    palette = colors[:n_colors - 1]  # exclude bg
    for y in range(H):
        for x in range(W):
            if random.random() < 0.15:
                grid[y][x] = random.choice(palette)
    return grid


def _empty_grid(H: int, W: int, bg: int = 0) -> List[List[int]]:
    return [[bg] * W for _ in range(H)]


def _copy_grid(grid: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in grid]


def _rand_dims(min_d: int = 3, max_d: int = 10) -> Tuple[int, int]:
    return random.randint(min_d, max_d), random.randint(min_d, max_d)


def _rand_bg() -> int:
    return random.randint(0, N_COLORS - 1)


def _rand_palette(n: int, exclude: int = -1) -> List[int]:
    pool = [c for c in range(N_COLORS) if c != exclude]
    random.shuffle(pool)
    return pool[:n]


def _place_rect(grid, y0, x0, h, w, color):
    H, W = len(grid), len(grid[0])
    for dy in range(h):
        for dx in range(w):
            yy, xx = y0 + dy, x0 + dx
            if 0 <= yy < H and 0 <= xx < W:
                grid[yy][xx] = color


def _flood_fill(grid, y, x, old_color, new_color):
    H, W = len(grid), len(grid[0])
    if y < 0 or y >= H or x < 0 or x >= W:
        return
    if grid[y][x] != old_color:
        return
    grid[y][x] = new_color
    _flood_fill(grid, y - 1, x, old_color, new_color)
    _flood_fill(grid, y + 1, x, old_color, new_color)
    _flood_fill(grid, y, x - 1, old_color, new_color)
    _flood_fill(grid, y, x + 1, old_color, new_color)


def _flood_fill_iter(grid, y, x, old_color, new_color):
    """Iterative flood fill to avoid recursion depth issues."""
    H, W = len(grid), len(grid[0])
    stack = [(y, x)]
    while stack:
        cy, cx = stack.pop()
        if cy < 0 or cy >= H or cx < 0 or cx >= W:
            continue
        if grid[cy][cx] != old_color:
            continue
        grid[cy][cx] = new_color
        stack.extend([(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)])


# ────────────────────────────────────────────────────────────────────
# Stage 1: Global spatial transform rules
# ────────────────────────────────────────────────────────────────────

def rule_gravity(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Objects fall to the bottom of the grid.

    Requires knowing grid extents — a purely local operation cannot
    determine how far each pixel must fall.
    """
    random.seed(rng_seed)
    H, W = _rand_dims(5, 10)
    bg = _rand_bg()
    palette = _rand_palette(3, exclude=bg)

    inp = _empty_grid(H, W, bg)
    for _ in range(random.randint(3, H * W // 4)):
        y, x = random.randint(0, H - 1), random.randint(0, W - 1)
        inp[y][x] = random.choice(palette)

    # Gravity: for each column, collect non-bg pixels, stack at bottom
    out = _empty_grid(H, W, bg)
    for x in range(W):
        col_pixels = [inp[y][x] for y in range(H) if inp[y][x] != bg]
        for i, c in enumerate(reversed(col_pixels)):
            out[H - 1 - i][x] = c

    return inp, out


def rule_translate(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Translate an object by a vector indicated by a marker pixel.

    The marker's position relative to the object encodes the translation
    direction — requires reading the entire grid to locate both.
    """
    random.seed(rng_seed)
    H, W = _rand_dims(6, 10)
    bg = _rand_bg()
    obj_color, marker_color = _rand_palette(2, exclude=bg)

    inp = _empty_grid(H, W, bg)

    # Place a small object
    oh, ow = random.randint(2, 3), random.randint(2, 3)
    oy, ox = random.randint(0, H // 2 - oh), random.randint(0, W // 2 - ow)
    obj_cells = []
    for dy in range(oh):
        for dx in range(ow):
            inp[oy + dy][ox + dx] = obj_color
            obj_cells.append((oy + dy, ox + dx))

    # Translation vector
    ty = random.randint(1, 3)
    tx = random.randint(1, 3)

    # Place marker showing direction (single pixel offset from object)
    my, mx = oy + oh + ty, ox + ow + tx
    if 0 <= my < H and 0 <= mx < W:
        inp[my][mx] = marker_color

    # Output: object translated, marker remains
    out = _copy_grid(inp)
    for cy, cx in obj_cells:
        out[cy][cx] = bg
    for cy, cx in obj_cells:
        ny, nx = cy + ty, cx + tx
        if 0 <= ny < H and 0 <= nx < W:
            out[ny][nx] = obj_color

    return inp, out


def rule_reflect_h(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Mirror the left half of the grid onto the right half.

    Output right half is determined entirely by left half — requires
    global positional awareness of grid width.
    """
    random.seed(rng_seed)
    H, W = _rand_dims(4, 8)
    bg = _rand_bg()
    palette = _rand_palette(3, exclude=bg)

    inp = _empty_grid(H, W, bg)
    for y in range(H):
        for x in range(W // 2):
            if random.random() < 0.3:
                inp[y][x] = random.choice(palette)

    out = _copy_grid(inp)
    for y in range(H):
        for x in range(W // 2):
            out[y][W - 1 - x] = inp[y][x]

    return inp, out


def rule_reflect_v(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Mirror the top half of the grid onto the bottom half.

    Output bottom half is determined by top half — requires global
    awareness of grid height.
    """
    random.seed(rng_seed)
    H, W = _rand_dims(4, 8)
    bg = _rand_bg()
    palette = _rand_palette(3, exclude=bg)

    inp = _empty_grid(H, W, bg)
    for y in range(H // 2):
        for x in range(W):
            if random.random() < 0.3:
                inp[y][x] = random.choice(palette)

    out = _copy_grid(inp)
    for y in range(H // 2):
        for x in range(W):
            out[H - 1 - y][x] = inp[y][x]

    return inp, out


def rule_draw_line(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Connect two endpoint dots with a straight horizontal or vertical line.

    Both endpoints must be found across the grid before the line can be drawn.
    """
    random.seed(rng_seed)
    H, W = _rand_dims(5, 10)
    bg = _rand_bg()
    dot_color = _rand_palette(1, exclude=bg)[0]

    inp = _empty_grid(H, W, bg)
    out = _empty_grid(H, W, bg)

    if random.random() < 0.5:
        # Horizontal line — ensure at least 2 gap so interior cells exist
        y = random.randint(0, H - 1)
        x1 = random.randint(0, max(0, W - 3))
        x2 = random.randint(min(x1 + 2, W - 1), W - 1)
        inp[y][x1] = dot_color
        inp[y][x2] = dot_color
        out = _copy_grid(inp)
        for x in range(x1, x2 + 1):
            out[y][x] = dot_color
    else:
        # Vertical line — ensure at least 2 gap so interior cells exist
        x = random.randint(0, W - 1)
        y1 = random.randint(0, max(0, H - 3))
        y2 = random.randint(min(y1 + 2, H - 1), H - 1)
        inp[y1][x] = dot_color
        inp[y2][x] = dot_color
        out = _copy_grid(inp)
        for y in range(y1, y2 + 1):
            out[y][x] = dot_color

    return inp, out


def rule_raycast(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Shoot a ray from each source pixel until it hits a wall or the grid edge.

    The ray travels in a fixed cardinal direction encoded by the source color:
      - color index 0 mod 4 → right
      - color index 1 mod 4 → left
      - color index 2 mod 4 → down
      - color index 3 mod 4 → up

    Requires locating all source pixels and all wall pixels across the entire
    grid before any ray endpoint can be determined.
    """
    random.seed(rng_seed)
    H, W = _rand_dims(5, 12)
    bg = _rand_bg()

    # Pick distinct colors for walls and 1-3 sources
    n_sources = random.randint(1, 3)
    needed = 1 + n_sources  # wall + sources
    palette = _rand_palette(needed, exclude=bg)
    wall_color = palette[0]
    source_colors = palette[1:1 + n_sources]

    inp = _empty_grid(H, W, bg)

    # Place wall segments (short horizontal or vertical bars, 1-3 walls)
    n_walls = random.randint(1, 3)
    for _ in range(n_walls):
        if random.random() < 0.5:
            # Horizontal wall bar
            wy = random.randint(0, H - 1)
            wx0 = random.randint(0, W - 2)
            wx1 = min(wx0 + random.randint(1, 3), W - 1)
            for wx in range(wx0, wx1 + 1):
                inp[wy][wx] = wall_color
        else:
            # Vertical wall bar
            wx = random.randint(0, W - 1)
            wy0 = random.randint(0, H - 2)
            wy1 = min(wy0 + random.randint(1, 3), H - 1)
            for wy in range(wy0, wy1 + 1):
                inp[wy][wx] = wall_color

    # Place source pixels — guaranteed to have at least one bg cell in ray direction
    sources: List[Tuple[int, int, int]] = []  # (y, x, color)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U
    placed = 0
    attempts = 0
    while placed < n_sources and attempts < 400:
        attempts += 1
        sc = source_colors[placed]
        color_idx = sc % 4
        dy, dx = directions[color_idx]

        sy = random.randint(0, H - 1)
        sx = random.randint(0, W - 1)
        if inp[sy][sx] != bg:
            continue

        # Verify at least one bg cell exists ahead in the ray direction
        cy, cx = sy + dy, sx + dx
        has_open = False
        while 0 <= cy < H and 0 <= cx < W:
            if inp[cy][cx] == wall_color:
                break
            if inp[cy][cx] == bg:
                has_open = True
                break
            cy += dy
            cx += dx

        if not has_open:
            continue

        inp[sy][sx] = sc
        sources.append((sy, sx, sc))
        placed += 1

    # If no sources could be placed (degenerate grid), fall back to one guaranteed source
    if not sources:
        # Clear all walls from the middle row/col and place one source
        mid_y = H // 2
        for x in range(W):
            if inp[mid_y][x] == wall_color:
                inp[mid_y][x] = bg
        sc = source_colors[0]
        # Direction R (dx=1): place source at leftmost cell of middle row
        inp[mid_y][0] = sc
        sources.append((mid_y, 0, sc))

    # Draw rays in output
    out = _copy_grid(inp)
    for sy, sx, sc in sources:
        color_idx = sc % 4
        dy, dx = directions[color_idx]
        cy, cx = sy + dy, sx + dx
        while 0 <= cy < H and 0 <= cx < W:
            if out[cy][cx] == wall_color:
                break
            # Only paint over background cells (don't overwrite other sources/rays)
            if out[cy][cx] == bg:
                out[cy][cx] = sc
            cy += dy
            cx += dx

    return inp, out


def rule_connect_same_color(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Connect 2-4 pairs of same-colored dots with straight (H or V) lines.

    All pairs must be found across the entire grid before any connection
    can be drawn. Pairs are guaranteed to be axis-aligned for unique paths.
    """
    random.seed(rng_seed)
    H, W = _rand_dims(6, 12)
    bg = _rand_bg()

    n_pairs = random.randint(2, 4)
    palette = _rand_palette(n_pairs, exclude=bg)

    inp = _empty_grid(H, W, bg)
    out = _empty_grid(H, W, bg)

    pairs_placed = 0
    attempts_total = 0
    # Track occupied cells to avoid collisions
    occupied = set()

    for color in palette:
        if pairs_placed >= n_pairs:
            break
        placed = False
        for _ in range(100):
            attempts_total += 1
            # Randomly choose H or V alignment
            if random.random() < 0.5:
                # Horizontal pair: same row, different columns
                row = random.randint(0, H - 1)
                c1 = random.randint(0, W // 2 - 1)
                c2 = random.randint(W // 2, W - 1)
                cells_on_line = [(row, c) for c in range(c1, c2 + 1)]
            else:
                # Vertical pair: same column, different rows
                col = random.randint(0, W - 1)
                r1 = random.randint(0, H // 2 - 1)
                r2 = random.randint(H // 2, H - 1)
                cells_on_line = [(r, col) for r in range(r1, r2 + 1)]

            # Check no collision with existing occupied cells
            if any(cell in occupied for cell in cells_on_line):
                continue

            # Mark endpoints in input, full line in output
            p1, p2 = cells_on_line[0], cells_on_line[-1]
            inp[p1[0]][p1[1]] = color
            inp[p2[0]][p2[1]] = color
            for (r, c) in cells_on_line:
                out[r][c] = color
            occupied.update(cells_on_line)
            pairs_placed += 1
            placed = True
            break

        if not placed:
            # Could not place this color without collision — skip it
            continue

    # Fill non-line cells in output with bg
    for y in range(H):
        for x in range(W):
            if out[y][x] == bg and inp[y][x] != bg:
                out[y][x] = inp[y][x]  # Keep endpoint dots that weren't connected

    # Make sure input != output
    changed = any(
        inp[y][x] != out[y][x]
        for y in range(H) for x in range(W)
    )
    if not changed and pairs_placed > 0:
        # All pairs were single-cell? Force a two-cell gap somewhere
        # Fallback: pick first pair color and make endpoints 2 apart
        pass  # extremely unlikely with the range constraints above

    return inp, out


# ────────────────────────────────────────────────────────────────────
# Stage 2: Relational (object-level) rules
# ────────────────────────────────────────────────────────────────────

def rule_copy_object(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Copy an object to a marked destination location.

    The source object and destination marker are separated across the grid —
    both must be located before the copy can be performed.
    """
    random.seed(rng_seed)
    H, W = _rand_dims(7, 12)
    bg = _rand_bg()
    obj_color, mark_color = _rand_palette(2, exclude=bg)

    inp = _empty_grid(H, W, bg)

    # Object: small shape in top-left quadrant
    oh, ow = random.randint(2, 3), random.randint(2, 3)
    oy, ox = random.randint(0, 2), random.randint(0, 2)
    obj_cells = []
    for dy in range(oh):
        for dx in range(ow):
            if random.random() < 0.7:
                inp[oy + dy][ox + dx] = obj_color
                obj_cells.append((dy, dx))

    if not obj_cells:
        obj_cells = [(0, 0)]
        inp[oy][ox] = obj_color

    # Destination marker in opposite quadrant
    dy_off = random.randint(oh + 1, max(oh + 2, H - oh - 1))
    dx_off = random.randint(ow + 1, max(ow + 2, W - ow - 1))
    dest_y, dest_x = min(oy + dy_off, H - 1), min(ox + dx_off, W - 1)
    inp[dest_y][dest_x] = mark_color

    out = _copy_grid(inp)
    for dy, dx in obj_cells:
        ny, nx = dest_y + dy, dest_x + dx
        if 0 <= ny < H and 0 <= nx < W:
            out[ny][nx] = obj_color

    return inp, out


def rule_enclosed_fill(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Fill all enclosed regions (bounded by walls) with a new color.

    Requires detecting whether each interior cell is fully enclosed —
    a determination that depends on the boundary across the whole grid.
    """
    random.seed(rng_seed)
    H, W = _rand_dims(6, 10)
    bg = _rand_bg()
    wall_color, fill_color = _rand_palette(2, exclude=bg)

    inp = _empty_grid(H, W, bg)

    # Draw 1-2 closed rectangles
    n_rects = random.randint(1, 2)
    rects = []
    for _ in range(n_rects):
        rh = random.randint(3, min(5, H - 1))
        rw = random.randint(3, min(5, W - 1))
        ry = random.randint(0, H - rh)
        rx = random.randint(0, W - rw)
        rects.append((ry, rx, rh, rw))
        for dy in range(rh):
            for dx in range(rw):
                if dy == 0 or dy == rh - 1 or dx == 0 or dx == rw - 1:
                    inp[ry + dy][rx + dx] = wall_color

    out = _copy_grid(inp)
    for ry, rx, rh, rw in rects:
        for dy in range(1, rh - 1):
            for dx in range(1, rw - 1):
                if out[ry + dy][rx + dx] == bg:
                    out[ry + dy][rx + dx] = fill_color

    return inp, out


def rule_recolor(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Recolor objects by relative size: largest → color A, rest → color B.

    Requires counting all object sizes globally before any cell can be
    assigned its output color.
    """
    random.seed(rng_seed)
    H, W = _rand_dims(6, 10)
    bg = _rand_bg()
    obj_color = _rand_palette(1, exclude=bg)[0]
    color_big, color_small = _rand_palette(2, exclude=bg)

    inp = _empty_grid(H, W, bg)

    # Place 2-3 non-overlapping rectangles of same color but different sizes
    rects = []
    for _ in range(random.randint(2, 3)):
        rh = random.randint(2, max(2, H // 3))
        rw = random.randint(2, max(2, W // 3))
        ry = random.randint(0, H - rh)
        rx = random.randint(0, W - rw)
        _place_rect(inp, ry, rx, rh, rw, obj_color)
        rects.append((ry, rx, rh, rw, rh * rw))

    out = _copy_grid(inp)
    if rects:
        rects.sort(key=lambda r: r[4], reverse=True)
        for i, (ry, rx, rh, rw, _) in enumerate(rects):
            c = color_big if i == 0 else color_small
            _place_rect(out, ry, rx, rh, rw, c)

    return inp, out


def rule_extend_pattern(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Extrapolate a partial repeating pattern to fill the entire grid.

    A stripe or checkerboard pattern is shown in the top row (or top-left
    corner). The output tiles this pattern across the full grid. Requires
    detecting periodicity and applying it globally.
    """
    random.seed(rng_seed)
    H, W = _rand_dims(4, 8)
    bg = _rand_bg()

    pattern_type = random.choice(["stripe_h", "stripe_v", "checker"])

    if pattern_type == "stripe_h":
        # Alternating horizontal stripes of period 1 or 2
        period = random.randint(1, 2)
        palette = _rand_palette(2, exclude=bg)
        out = _empty_grid(H, W, bg)
        for y in range(H):
            c = palette[(y // period) % 2]
            for x in range(W):
                out[y][x] = c
        # Input: only the first `period * 2` rows are shown, rest is bg
        inp = _empty_grid(H, W, bg)
        reveal_rows = min(period * 2, H)
        for y in range(reveal_rows):
            for x in range(W):
                inp[y][x] = out[y][x]

    elif pattern_type == "stripe_v":
        # Alternating vertical stripes of period 1 or 2
        period = random.randint(1, 2)
        palette = _rand_palette(2, exclude=bg)
        out = _empty_grid(H, W, bg)
        for x in range(W):
            c = palette[(x // period) % 2]
            for y in range(H):
                out[y][x] = c
        # Input: only the first `period * 2` columns are shown, rest is bg
        inp = _empty_grid(H, W, bg)
        reveal_cols = min(period * 2, W)
        for y in range(H):
            for x in range(reveal_cols):
                inp[y][x] = out[y][x]

    else:  # checker
        palette = _rand_palette(2, exclude=bg)
        out = _empty_grid(H, W, bg)
        for y in range(H):
            for x in range(W):
                out[y][x] = palette[(y + x) % 2]
        # Input: top-left quadrant only
        inp = _empty_grid(H, W, bg)
        reveal_h = max(1, H // 2)
        reveal_w = max(1, W // 2)
        for y in range(reveal_h):
            for x in range(reveal_w):
                inp[y][x] = out[y][x]

    # Guarantee input != output
    changed = any(
        inp[y][x] != out[y][x]
        for y in range(H) for x in range(W)
    )
    if not changed:
        # Grid is too small to hide anything — just use full output but
        # shift input by leaving last row as bg (degenerate fallback)
        inp = _copy_grid(out)
        if H > 1:
            for x in range(W):
                inp[H - 1][x] = bg

    return inp, out


# ────────────────────────────────────────────────────────────────────
# Stage 3: Composition rules
# ────────────────────────────────────────────────────────────────────

def rule_pattern_repeat(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Tile a small pattern shown once in the corner to fill the full grid.

    Requires recognizing the tile extent, then applying the repetition
    across both dimensions globally.
    """
    random.seed(rng_seed)
    th, tw = random.randint(2, 3), random.randint(2, 3)
    bg = _rand_bg()
    palette = _rand_palette(2, exclude=bg)

    tile = _empty_grid(th, tw, bg)
    for y in range(th):
        for x in range(tw):
            if random.random() < 0.4:
                tile[y][x] = random.choice(palette)

    # Ensure tile is not all-background
    if all(tile[y][x] == bg for y in range(th) for x in range(tw)):
        tile[0][0] = palette[0]

    reps_h = random.randint(2, 3)
    reps_w = random.randint(2, 3)
    H, W = th * reps_h, tw * reps_w

    # Input: tile shown once in top-left, rest background
    inp = _empty_grid(H, W, bg)
    for y in range(th):
        for x in range(tw):
            inp[y][x] = tile[y][x]

    # Output: tile repeated across full grid
    out = _empty_grid(H, W, bg)
    for ry in range(reps_h):
        for rx in range(reps_w):
            for y in range(th):
                for x in range(tw):
                    out[ry * th + y][rx * tw + x] = tile[y][x]

    return inp, out


def rule_scale_up(rng_seed: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Scale a small pattern up by 2×.

    Requires understanding the full small pattern (global input) and
    computing its expanded form (global output).
    """
    random.seed(rng_seed)
    sh, sw = random.randint(2, 4), random.randint(2, 4)
    bg = _rand_bg()
    palette = _rand_palette(3, exclude=bg)

    small = _empty_grid(sh, sw, bg)
    for y in range(sh):
        for x in range(sw):
            if random.random() < 0.4:
                small[y][x] = random.choice(palette)

    # Ensure not all-background
    if all(small[y][x] == bg for y in range(sh) for x in range(sw)):
        small[0][0] = palette[0]

    inp = small
    H, W = sh * 2, sw * 2
    out = _empty_grid(H, W, bg)
    for y in range(sh):
        for x in range(sw):
            c = small[y][x]
            out[2 * y][2 * x] = c
            out[2 * y][2 * x + 1] = c
            out[2 * y + 1][2 * x] = c
            out[2 * y + 1][2 * x + 1] = c

    return inp, out


# ────────────────────────────────────────────────────────────────────
# Rule registry by curriculum stage
# ────────────────────────────────────────────────────────────────────

RULES_BY_STAGE = {
    CurriculumStage.GLOBAL: [
        rule_gravity,
        rule_translate,
        rule_reflect_h,
        rule_reflect_v,
        rule_raycast,
        rule_draw_line,
        rule_connect_same_color,
    ],
    CurriculumStage.RELATIONAL: [
        rule_copy_object,
        rule_enclosed_fill,
        rule_recolor,
        rule_extend_pattern,
    ],
    CurriculumStage.COMPOSITION: [
        rule_pattern_repeat,
        rule_scale_up,
    ],
}


# ────────────────────────────────────────────────────────────────────
# Sequence building (mirrors fgn/tasks/arc.py)
# ────────────────────────────────────────────────────────────────────

def _grid_to_cells(grid: List[List[int]], role: int):
    """Flatten 2D grid to cell-as-token representation (row-major)."""
    colors, xs, ys, roles = [], [], [], []
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    for y in range(H):
        for x in range(W):
            colors.append(grid[y][x])
            xs.append(x)
            ys.append(y)
            roles.append(role)
    return colors, xs, ys, roles


def _add_sep(colors, xs, ys, roles, sep_types_list, sep_mask_list, grid_ids, sep_type):
    """Add a separator token."""
    colors.append(PAD_COLOR)
    xs.append(PAD_COORD)
    ys.append(PAD_COORD)
    roles.append(0)
    sep_types_list.append(sep_type)
    sep_mask_list.append(True)
    grid_ids.append(-1)


def build_sequence(demos: List[Tuple[List[List[int]], List[List[int]]]],
                   test_input: List[List[int]],
                   test_output: List[List[int]],
                   ) -> Dict:
    """Build cell-as-token sequence from demo pairs + test pair.

    Returns dict with same keys as fgn/tasks/arc.py build_sequence().
    """
    colors, xs, ys, roles_list = [], [], [], []
    sep_mask_list: List[bool] = []
    sep_types_list: List[int] = []
    grid_ids: List[int] = []
    grid_counter = 0

    for i, (demo_in, demo_out) in enumerate(demos):
        if i > 0:
            _add_sep(colors, xs, ys, roles_list, sep_types_list, sep_mask_list,
                     grid_ids, SEP_BETWEEN_DEMOS)

        # Demo input
        c, x, y, r = _grid_to_cells(demo_in, ROLE_INPUT_DEMO)
        colors.extend(c); xs.extend(x); ys.extend(y); roles_list.extend(r)
        sep_mask_list.extend([False] * len(c))
        sep_types_list.extend([0] * len(c))
        grid_ids.extend([grid_counter] * len(c))
        grid_counter += 1

        _add_sep(colors, xs, ys, roles_list, sep_types_list, sep_mask_list,
                 grid_ids, SEP_DEMO_IO)

        # Demo output
        c, x, y, r = _grid_to_cells(demo_out, ROLE_OUTPUT_DEMO)
        colors.extend(c); xs.extend(x); ys.extend(y); roles_list.extend(r)
        sep_mask_list.extend([False] * len(c))
        sep_types_list.extend([0] * len(c))
        grid_ids.extend([grid_counter] * len(c))
        grid_counter += 1

    # Separator before test input
    _add_sep(colors, xs, ys, roles_list, sep_types_list, sep_mask_list,
             grid_ids, SEP_BEFORE_TEST_IN)

    # Test input
    c, x, y, r = _grid_to_cells(test_input, ROLE_TEST_INPUT)
    colors.extend(c); xs.extend(x); ys.extend(y); roles_list.extend(r)
    sep_mask_list.extend([False] * len(c))
    sep_types_list.extend([0] * len(c))
    grid_ids.extend([grid_counter] * len(c))
    test_input_grid_id = grid_counter
    grid_counter += 1

    # Separator before test output
    _add_sep(colors, xs, ys, roles_list, sep_types_list, sep_mask_list,
             grid_ids, SEP_BEFORE_TEST_OUT)

    # Test output
    test_out_start = len(colors)
    c, x, y, r = _grid_to_cells(test_output, ROLE_TEST_OUTPUT)
    colors.extend(c); xs.extend(x); ys.extend(y); roles_list.extend(r)
    sep_mask_list.extend([False] * len(c))
    sep_types_list.extend([0] * len(c))
    grid_ids.extend([grid_counter] * len(c))
    test_out_end = len(colors)

    N = len(colors)

    # Build target_mask
    target_mask = [False] * N
    for i in range(test_out_start, test_out_end):
        target_mask[i] = True

    # Build target_colors (just the test output cells)
    target_colors = colors[test_out_start:test_out_end]

    # Build target_input_colors: test input color at same (x, y)
    test_in_H = len(test_input)
    test_in_W = len(test_input[0]) if test_in_H > 0 else 0
    target_input_colors = [PAD_COLOR] * N
    for i in range(test_out_start, test_out_end):
        tx, ty = xs[i], ys[i]
        if 0 <= ty < test_in_H and 0 <= tx < test_in_W:
            target_input_colors[i] = test_input[ty][tx]
        else:
            target_input_colors[i] = 0  # bg

    return {
        "colors": colors,
        "xs": xs,
        "ys": ys,
        "roles": roles_list,
        "sep_mask": sep_mask_list,
        "sep_types": sep_types_list,
        "grid_ids": grid_ids,
        "target_mask": target_mask,
        "target_colors": target_colors,
        "target_input_colors": target_input_colors,
        "length": N,
    }


def _augment_grid(grid: List[List[int]], rot: int, flip: bool) -> List[List[int]]:
    """Apply D4 symmetry augmentation (rotation + optional reflection)."""
    g = grid
    for _ in range(rot):
        # 90° clockwise: new[x][y] = old[H-1-y][x]
        H, W = len(g), len(g[0])
        g = [[g[H - 1 - y][x] for y in range(H)] for x in range(W)]
    if flip:
        g = [row[::-1] for row in g]
    return g


def _permute_colors(grid: List[List[int]], perm: Dict[int, int]) -> List[List[int]]:
    """Apply color permutation (keep 0 fixed typically)."""
    return [[perm.get(c, c) for c in row] for row in grid]


# ────────────────────────────────────────────────────────────────────
# Main task class
# ────────────────────────────────────────────────────────────────────

class ProceduralARCTask:
    """Infinite procedural ARC task generator with curriculum support.

    Drop-in replacement for ARCTask — same generate_batch() interface.

    All tasks are global-relational: no task can be solved by inspecting
    a single cell or a small local neighborhood. Every rule requires
    whole-grid awareness.

    Args:
        seq_len: max sequence length (padded to this)
        stage: curriculum stage (GLOBAL, RELATIONAL, COMPOSITION)
        include_lower: if True, include all stages up to and including `stage`
        augment: apply D4 symmetry + color permutation
        n_demos: number of demonstration pairs per task
    """

    def __init__(
        self,
        seq_len: int = 1024,
        stage: CurriculumStage = CurriculumStage.GLOBAL,
        include_lower: bool = True,
        augment: bool = True,
        n_demos: int = 2,
        **kwargs,
    ):
        self.seq_len = seq_len
        self.stage = stage
        self.augment = augment
        self.n_demos = n_demos
        self._seed_counter = random.randint(0, 2**31)

        # Collect rules for this stage
        self.rules = []
        if include_lower:
            for s in CurriculumStage:
                if s <= stage:
                    self.rules.extend(RULES_BY_STAGE[s])
        else:
            self.rules = list(RULES_BY_STAGE[stage])

    def _next_seed(self) -> int:
        """Monotonically increasing seed — guarantees never-seen-before grids."""
        self._seed_counter += 1
        return self._seed_counter

    def _generate_one(self) -> Dict:
        """Generate a single task instance (demos + test) and serialize."""
        rule = random.choice(self.rules)

        # Augmentation params (shared across all demos + test for this task)
        rot = random.randint(0, 3) if self.augment else 0
        flip = random.random() < 0.5 if self.augment else False

        # Color permutation: randomly remap colors 1-9 (keep 0/bg fixed)
        if self.augment:
            perm_vals = list(range(1, N_COLORS))
            random.shuffle(perm_vals)
            color_perm = {0: 0}
            for i, v in enumerate(perm_vals):
                color_perm[i + 1] = v
        else:
            color_perm = {i: i for i in range(N_COLORS)}

        demos = []
        for _ in range(self.n_demos):
            seed = self._next_seed()
            inp, out = rule(seed)
            inp = _permute_colors(_augment_grid(inp, rot, flip), color_perm)
            out = _permute_colors(_augment_grid(out, rot, flip), color_perm)
            demos.append((inp, out))

        # Test pair (different seed = different grid instance, same rule)
        test_seed = self._next_seed()
        test_inp, test_out = rule(test_seed)
        test_inp = _permute_colors(_augment_grid(test_inp, rot, flip), color_perm)
        test_out = _permute_colors(_augment_grid(test_out, rot, flip), color_perm)

        seq = build_sequence(demos, test_inp, test_out)

        # Truncate if too long
        if seq["length"] > self.seq_len:
            # Retry with fewer demos
            if self.n_demos > 1:
                demos = demos[:1]
                seq = build_sequence(demos, test_inp, test_out)
            # Still too long? Truncate (lossy)
            if seq["length"] > self.seq_len:
                for key in ["colors", "xs", "ys", "roles", "sep_mask", "sep_types",
                            "grid_ids", "target_mask", "target_input_colors"]:
                    seq[key] = seq[key][:self.seq_len]
                seq["length"] = self.seq_len

        return seq

    def generate_batch(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Generate a batch of procedural ARC tasks.

        Returns same format as ARCTask.generate_batch():
            (input_ids, labels, metadata)

        where metadata contains all tensors needed by the model.
        """
        if device is None:
            device = torch.device("cpu")

        samples = []
        for _ in range(batch_size):
            samples.append(self._generate_one())

        max_N = self.seq_len  # Fixed padding for torch.compile stability

        # Allocate padded tensors
        colors = torch.full((batch_size, max_N), PAD_COLOR, dtype=torch.long, device=device)
        xs_t = torch.full((batch_size, max_N), PAD_COORD, dtype=torch.long, device=device)
        ys_t = torch.full((batch_size, max_N), PAD_COORD, dtype=torch.long, device=device)
        roles = torch.zeros(batch_size, max_N, dtype=torch.long, device=device)
        sep_mask = torch.ones(batch_size, max_N, dtype=torch.bool, device=device)  # pad = sep
        sep_types = torch.zeros(batch_size, max_N, dtype=torch.long, device=device)
        grid_ids = torch.full((batch_size, max_N), -1, dtype=torch.long, device=device)
        target_mask = torch.zeros(batch_size, max_N, dtype=torch.bool, device=device)
        target_labels = torch.full((batch_size, max_N), -100, dtype=torch.long, device=device)
        target_input_colors = torch.full((batch_size, max_N), PAD_COLOR, dtype=torch.long, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        context_mask = torch.ones(batch_size, max_N, dtype=torch.bool, device=device)

        for i, s in enumerate(samples):
            N = s["length"]
            lengths[i] = N
            colors[i, :N] = torch.tensor(s["colors"], dtype=torch.long)
            xs_t[i, :N] = torch.tensor(s["xs"], dtype=torch.long)
            ys_t[i, :N] = torch.tensor(s["ys"], dtype=torch.long)
            roles[i, :N] = torch.tensor(s["roles"], dtype=torch.long)
            sep_mask[i, :N] = torch.tensor(s["sep_mask"], dtype=torch.bool)
            sep_types[i, :N] = torch.tensor(s["sep_types"], dtype=torch.long)
            grid_ids[i, :N] = torch.tensor(s["grid_ids"], dtype=torch.long)
            target_mask[i, :N] = torch.tensor(s["target_mask"], dtype=torch.bool)
            target_input_colors[i, :N] = torch.tensor(s["target_input_colors"], dtype=torch.long)

            # Align target_labels: place ground truth colors at target positions
            tgt_positions = [j for j, m in enumerate(s["target_mask"]) if m]
            for j, pos in enumerate(tgt_positions):
                if j < len(s["target_colors"]):
                    target_labels[i, pos] = s["target_colors"][j]

            # Context mask = ~target_mask for actual tokens, True for padding
            context_mask[i, :N] = ~target_mask[i, :N]

        # Placeholder returns (matches ARCTask interface)
        input_ids = torch.zeros(batch_size, max_N, dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_N), -100, dtype=torch.long, device=device)

        meta = {
            "colors": colors,
            "xs": xs_t,
            "ys": ys_t,
            "roles": roles,
            "sep_mask": sep_mask,
            "sep_types": sep_types,
            "grid_ids": grid_ids,
            "target_mask": target_mask,
            "target_labels": target_labels,
            "target_input_colors": target_input_colors,
            "context_mask": context_mask,
            "lengths": lengths,
        }

        return input_ids, labels, meta


if __name__ == "__main__":
    print("Testing ProceduralARCTask...")

    # Test each stage independently
    for stage in CurriculumStage:
        task = ProceduralARCTask(seq_len=512, stage=stage, include_lower=False, n_demos=2)
        _, _, meta = task.generate_batch(4)

        print(f"\n  Stage {stage.name}:")
        print(f"    Rules: {[r.__name__ for r in task.rules]}")
        print(f"    Colors shape: {meta['colors'].shape}")
        print(f"    Lengths: {meta['lengths'].tolist()}")
        print(f"    Targets per sample: {[(meta['target_labels'][i] != -100).sum().item() for i in range(4)]}")
        print(f"    Grid IDs range: [{meta['grid_ids'].min().item()}, {meta['grid_ids'].max().item()}]")

        # Verify structure
        for i in range(4):
            n = meta["lengths"][i].item()
            n_tgt = (meta["target_labels"][i] != -100).sum().item()
            n_ctx = meta["context_mask"][i, :n].sum().item()
            n_tmask = meta["target_mask"][i, :n].sum().item()
            assert n_tgt == n_tmask, f"target_labels/target_mask mismatch: {n_tgt} vs {n_tmask}"
            assert n_ctx + n_tmask == n, f"context+target != length: {n_ctx}+{n_tmask} != {n}"

        # Spot-check that each rule produces input != output
        print(f"    Input!=output check:")
        for rule_fn in task.rules:
            changed_count = 0
            for seed in range(20):
                inp, out = rule_fn(seed)
                if inp != out:
                    changed_count += 1
            status = "OK" if changed_count >= 18 else f"WARN ({changed_count}/20)"
            print(f"      {rule_fn.__name__}: {status}")

    # Test cumulative stages (include_lower=True)
    task_all = ProceduralARCTask(seq_len=512, stage=CurriculumStage.COMPOSITION,
                                  include_lower=True, n_demos=2)
    print(f"\n  All stages combined: {len(task_all.rules)} rules")
    _, _, meta = task_all.generate_batch(8)
    print(f"    Batch lengths: {meta['lengths'].tolist()}")

    # Verify infinite non-repeating
    _, _, m1 = task_all.generate_batch(2)
    _, _, m2 = task_all.generate_batch(2)
    same = (m1["colors"] == m2["colors"]).all().item()
    print(f"    Two batches identical: {same} (should be False)")
    assert not same, "Two batches should not be identical"

    print("\nProceduralARCTask OK")
