"""ARC grid → text serialization for oracle embedding extraction.

Converts ARC task dicts into structured text prompts that a foundation model
can encode. Includes D4 dihedral augmentation (4 rotations x 2 flips = 8 variants).
"""

from typing import List


def serialize_grid(grid: List[List[int]]) -> str:
    """Serialize a grid to text. Rows by newlines, cells by spaces.

    Example: [[0, 8, 0], [8, 8, 8]] → '0 8 0\\n8 8 8'
    """
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def rot90_grid(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 90 degrees clockwise."""
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    return [[grid[H - 1 - j][i] for j in range(H)] for i in range(W)]


def reflect_grid(grid: List[List[int]]) -> List[List[int]]:
    """Reflect grid horizontally (left-right)."""
    return [row[::-1] for row in grid]


def apply_d4(grid: List[List[int]], d4_idx: int) -> List[List[int]]:
    """Apply one of 8 D4 symmetry transforms (0-7).

    0-3: 0/90/180/270 rotation
    4-7: reflect + 0/90/180/270 rotation
    """
    g = grid
    n_rot = d4_idx % 4
    do_reflect = d4_idx >= 4

    if do_reflect:
        g = reflect_grid(g)
    for _ in range(n_rot):
        g = rot90_grid(g)
    return g


def serialize_task(task: dict, d4_idx: int = 0, test_idx: int = 0) -> str:
    """Serialize a full ARC task to a structured text prompt.

    Includes all demo pairs with D4 augmentation applied. Test output is excluded
    so the oracle must infer the transformation rule from demos + test input.

    Args:
        task: ARC task dict with "train" (demo pairs) and "test" keys
        d4_idx: D4 symmetry transform index (0-7)
        test_idx: which test pair to use

    Returns:
        Structured text prompt for the oracle model.
    """
    demos = task["train"]
    tests = task["test"]
    if test_idx >= len(tests):
        test_idx = 0

    parts = ["Task: Apply the spatial transformation rule."]

    for i, demo in enumerate(demos):
        inp_grid = apply_d4(demo["input"], d4_idx)
        out_grid = apply_d4(demo["output"], d4_idx)
        parts.append(f"Demo {i + 1}:")
        parts.append("Input:")
        parts.append(serialize_grid(inp_grid))
        parts.append("Output:")
        parts.append(serialize_grid(out_grid))

    test_pair = tests[test_idx]
    test_input = apply_d4(test_pair["input"], d4_idx)
    parts.append("Test:")
    parts.append("Input:")
    parts.append(serialize_grid(test_input))

    return "\n".join(parts)


if __name__ == "__main__":
    # Smoke test
    task = {
        "train": [
            {"input": [[0, 0], [0, 8]], "output": [[8, 0], [0, 0]]},
        ],
        "test": [
            {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
        ],
    }

    text = serialize_task(task, d4_idx=0, test_idx=0)
    print(text)
    print(f"\nLength: {len(text)} chars")

    # D4 variant
    text_r90 = serialize_task(task, d4_idx=1, test_idx=0)
    assert text != text_r90, "D4 rotation should change output"
    print("\nD4 rotation 90:")
    print(text_r90)

    print("\nserialize.py OK")
