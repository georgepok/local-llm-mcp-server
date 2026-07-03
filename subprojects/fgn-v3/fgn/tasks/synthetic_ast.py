"""Phase 1 — Synthetic AST editor task on fixed-shape binary expression trees.

Tests whether the K=2 coupled-substrate specialisation observed in Phase 0
(NumPy MLP toy, ast_editor_toy.py — see liquid-arc/research/self_org_sim/)
transfers to LiquidARC's continuous ODE substrate when the head structure
is (pointer, op, payload) emitted as an AR token stream.

Setup
-----
Tree: depth-3 complete binary expression tree (15 nodes, fixed shape).
    pre-order layout: 1 root op, 2 inner ops at depth 1, 4 inner ops at depth 2,
    8 leaf vars at depth 3.
    Inner ops at positions: 0, 1, 2, 3, 4, 5, 6  (7 positions)
    Leaves at positions:    7, 8, 9, 10, 11, 12, 13, 14  (8 positions)

Vocabulary (37 tokens):
    0:  PAD            (also EOS — unused after fixed length)
    1:  BOS
    2:  SEP_TGT
    3:  SEP_EDIT
    4-7:  VAR_a..VAR_d (4 vars)
    8-10: OP_+/OP_-/OP_* (3 ops)
    11-25: PTR_0..PTR_14 (15 pointer values)
    26-29: EOP_NOP/EOP_RV/EOP_RO/EOP_SL (4 edit-op values)
    30-36: PAY_0..PAY_6 (7 payload values — covers var idx, op idx,
           and SWAP-leaf offset 0..6)

Edit ops (4)
------------
    NOP            — identity
    REPLACE_VAR(p, payload)   — replace leaf at p with var index `payload`
    REPLACE_OP(p, payload)    — replace inner op at p with op index `payload`
    SWAP_LEAVES(p, payload)   — swap leaf at p with leaf at LEAF_POS-offset
                                (p+payload+1) within LEAF_POS. Payload encodes
                                "how many leaves over to swap with" in [0..6].
                                Requires positional reasoning, not lookup.

Canonical fix policy: find first mismatch p (leftmost). If p is a leaf AND
the swap-pair partner (a leaf q where current[q]=target[p] AND
current[p]=target[q]) exists, emit SWAP_LEAVES; otherwise REPLACE_VAR.
For inner-op mismatch, emit REPLACE_OP. Pad with NOP. Deterministic.

Sequence layout (length 4 + 2*15 + 4*3 = 46):
    BOS src(15) SEP_TGT tgt(15) SEP_EDIT [PTR1 OP1 PAY1 ... PTR4 OP4 PAY4]

Labels: -100 everywhere except positions SCRIPT_START..SCRIPT_START+11 which
carry the GT edit tokens. Convention matches SBF: labels[pos] is the GT
output at pos, supervised by logits[pos] via the LM head.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

PAD, BOS, SEP_TGT, SEP_EDIT = 0, 1, 2, 3

VAR_BASE = 4    # VAR_a..VAR_d
N_VARS = 4
OP_BASE = 8     # OP_+, OP_-, OP_*
N_OPS_TREE = 3

# Tree structure: depth-3 complete binary tree (15 nodes)
N_NODES = 15
INNER_POS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
LEAF_POS: Tuple[int, ...] = (7, 8, 9, 10, 11, 12, 13, 14)
N_LEAVES = len(LEAF_POS)
N_INNER = len(INNER_POS)
LEAF_INDEX = {p: i for i, p in enumerate(LEAF_POS)}
IS_LEAF = {p: (p in LEAF_POS) for p in range(N_NODES)}

# Edit-op INDEX values (NOT vocab tokens — these are emitted by the op head
# directly as integer labels in [0..N_EDIT_OPS-1]).
EOP_NOP, EOP_RV, EOP_RO, EOP_SL = 0, 1, 2, 3
N_EDIT_OPS = 4

PAY_RANGE = max(N_VARS, N_OPS_TREE, N_LEAVES - 1)  # = 7

# Single SLOT placeholder token in the input sequence — one per edit slot.
SLOT_TOKEN = 11
VOCAB_SIZE = SLOT_TOKEN + 1   # = 12 (compact — heads emit indices, not vocab)

K_CORRUPT = 2
K_FIX = 3

# Layout: BOS src(15) SEP_TGT tgt(15) SEP_EDIT [SLOT × K_FIX]
SEQ_LEN = 1 + N_NODES + 1 + N_NODES + 1 + K_FIX   # = 41
SCRIPT_START = 1 + N_NODES + 1 + N_NODES + 1      # = 33


# ---------------------------------------------------------------------------
# Tree generation + edit application
# ---------------------------------------------------------------------------

def gen_tree(rng: random.Random) -> List[int]:
    tokens = [0] * N_NODES
    for p in INNER_POS:
        tokens[p] = OP_BASE + rng.randrange(N_OPS_TREE)
    for p in LEAF_POS:
        tokens[p] = VAR_BASE + rng.randrange(N_VARS)
    return tokens


def apply_edit(tree: List[int], p: int, op: int, payload: int) -> List[int]:
    """Apply one edit. Out-of-range or wrong-position triples → NOP.
    Args use raw indices (p in [0..N_NODES), op in [0..N_EDIT_OPS),
    payload in [0..PAY_RANGE))."""
    out = list(tree)
    if not (0 <= p < N_NODES) or not (0 <= op < N_EDIT_OPS):
        return out
    if op == EOP_NOP:
        return out
    if op == EOP_RV and IS_LEAF.get(p, False) and 0 <= payload < N_VARS:
        out[p] = VAR_BASE + payload
    elif op == EOP_RO and (p in INNER_POS) and 0 <= payload < N_OPS_TREE:
        out[p] = OP_BASE + payload
    elif op == EOP_SL and IS_LEAF.get(p, False) and 0 <= payload < N_LEAVES - 1:
        i = LEAF_INDEX[p]
        j = (i + payload + 1) % N_LEAVES
        q = LEAF_POS[j]
        out[p], out[q] = out[q], out[p]
    return out


def gen_corruption(target: List[int], k_corrupt: int,
                   rng: random.Random) -> List[int]:
    """Apply k_corrupt random non-trivial edits to target → source tree."""
    state = list(target)
    for _ in range(k_corrupt):
        for _try in range(10):
            roll = rng.random()
            if roll < 0.4:
                p = rng.choice(LEAF_POS)
                old = state[p] - VAR_BASE
                payload = rng.choice([v for v in range(N_VARS) if v != old])
                edit = (p, EOP_RV, payload)
            elif roll < 0.7:
                p = rng.choice(INNER_POS)
                old = state[p] - OP_BASE
                payload = rng.choice([o for o in range(N_OPS_TREE) if o != old])
                edit = (p, EOP_RO, payload)
            else:
                p = rng.choice(LEAF_POS)
                payload = rng.randrange(N_LEAVES - 1)
                edit = (p, EOP_SL, payload)
            new_state = apply_edit(state, *edit)
            if new_state != state:
                state = new_state
                break
        else:
            state = new_state
    return state


def canonical_fix(state: List[int], target: List[int],
                  max_steps: int) -> Tuple[List[Tuple[int, int, int]], bool]:
    """Greedy leftmost-mismatch policy. Prefers SWAP_LEAVES for leaf-pair
    mismatches, REPLACE_VAR otherwise; REPLACE_OP for inner mismatches."""
    state = list(state)
    script: List[Tuple[int, int, int]] = []
    for _ in range(max_steps):
        diff = [i for i in range(N_NODES) if state[i] != target[i]]
        if not diff:
            script.append((0, EOP_NOP, 0))
            continue
        p = diff[0]
        if IS_LEAF[p]:
            # Try SWAP_LEAVES with another leaf
            swap_q = None
            for q in diff[1:]:
                if (IS_LEAF.get(q, False)
                        and state[q] == target[p]
                        and state[p] == target[q]):
                    swap_q = q
                    break
            if swap_q is not None:
                i = LEAF_INDEX[p]
                j = LEAF_INDEX[swap_q]
                # Solve (i + payload + 1) % N_LEAVES == j  for payload
                payload = (j - i - 1) % N_LEAVES
                if 0 <= payload < N_LEAVES - 1:
                    edit = (p, EOP_SL, payload)
                else:
                    edit = (p, EOP_RV, target[p] - VAR_BASE)
            else:
                edit = (p, EOP_RV, target[p] - VAR_BASE)
        else:
            edit = (p, EOP_RO, target[p] - OP_BASE)
        state = apply_edit(state, *edit)
        script.append(edit)
    return script, (state == target)


# ---------------------------------------------------------------------------
# Sequence assembly
# ---------------------------------------------------------------------------

def _build_sequence(src: List[int], tgt: List[int],
                    script: List[Tuple[int, int, int]]
                    ) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Returns (input_ids, gt_ptr, gt_op, gt_pay).
    input_ids carries SLOT_TOKEN at the K_FIX edit positions; the model has
    no AR access to GT — three independent heads must emit (p, op, pay) per
    slot from one substrate hidden state.
    """
    assert len(src) == N_NODES and len(tgt) == N_NODES
    assert len(script) == K_FIX
    input_ids: List[int] = [BOS]
    input_ids += src
    input_ids.append(SEP_TGT)
    input_ids += tgt
    input_ids.append(SEP_EDIT)
    input_ids.extend([SLOT_TOKEN] * K_FIX)
    assert len(input_ids) == SEQ_LEN, f"got {len(input_ids)} expected {SEQ_LEN}"
    gt_ptr = [edit[0] for edit in script]
    gt_op = [edit[1] for edit in script]
    gt_pay = [edit[2] for edit in script]
    return input_ids, gt_ptr, gt_op, gt_pay


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class SyntheticASTTask:
    """Phase 1 AST editor task. Yields (input_ids, labels, meta) batches."""

    def __init__(self, seed: int = 0,
                 k_corrupt: int = K_CORRUPT, k_fix: int = K_FIX,
                 max_resamples: int = 8):
        assert k_fix == K_FIX, "K_FIX is fixed at 4 in vocab layout"
        assert k_corrupt == K_CORRUPT
        self.k_corrupt = k_corrupt
        self.k_fix = k_fix
        self.max_resamples = max_resamples
        self.seq_len = SEQ_LEN
        self.vocab_size = VOCAB_SIZE
        self.pad_token_id = PAD
        self._rng = random.Random(seed)

    def _sample_one(self):
        for _ in range(self.max_resamples):
            tgt = gen_tree(self._rng)
            src = gen_corruption(tgt, self.k_corrupt, self._rng)
            script, recovered = canonical_fix(src, tgt, self.k_fix)
            if recovered:
                return src, tgt, script
        return src, tgt, script

    def generate_batch(self, batch_size: int,
                       device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cpu")
        ids_list: List[List[int]] = []
        gt_ptr_list: List[List[int]] = []
        gt_op_list: List[List[int]] = []
        gt_pay_list: List[List[int]] = []
        op_counts = [0, 0, 0, 0]
        for _ in range(batch_size):
            src, tgt, script = self._sample_one()
            for _, op, _ in script:
                op_counts[op] += 1
            input_ids, gt_ptr, gt_op, gt_pay = _build_sequence(src, tgt, script)
            ids_list.append(input_ids)
            gt_ptr_list.append(gt_ptr)
            gt_op_list.append(gt_op)
            gt_pay_list.append(gt_pay)
        return (
            torch.tensor(ids_list, dtype=torch.long, device=device),
            torch.tensor(gt_ptr_list, dtype=torch.long, device=device),
            torch.tensor(gt_op_list, dtype=torch.long, device=device),
            torch.tensor(gt_pay_list, dtype=torch.long, device=device),
            {"task": "AST_PHASE1", "task_name": "synthetic_ast",
             "batch_size": batch_size,
             "op_counts": op_counts},
        )

    # ------------------------------------------------------------------
    # Eval — three independent heads emit (pointer, op, payload) per slot
    # in a single forward pass. No AR teacher-forcing easy path.
    # ------------------------------------------------------------------

    @staticmethod
    def extract_state_target(input_ids: torch.Tensor):
        B = input_ids.shape[0]
        srcs, tgts = [], []
        for b in range(B):
            row = input_ids[b].tolist()
            srcs.append(row[1:1 + N_NODES])
            tgts.append(row[1 + N_NODES + 1: 1 + N_NODES + 1 + N_NODES])
        return srcs, tgts

    @classmethod
    def exact_match(cls, input_ids: torch.Tensor,
                    ptr_logits: torch.Tensor,
                    op_logits: torch.Tensor,
                    pay_logits: torch.Tensor,
                    gt_ptr: torch.Tensor,
                    gt_op: torch.Tensor) -> Tuple[float, float, float]:
        """ptr/op/pay logits: [B, K_FIX, *]. Apply argmax-decoded edits to
        source, check final == target. Also returns per-head argmax accuracy
        on PTR and OP (vs GT)."""
        B, K = ptr_logits.shape[:2]
        srcs, tgts = cls.extract_state_target(input_ids)
        ptr_pred = ptr_logits.argmax(dim=-1)  # [B, K]
        op_pred = op_logits.argmax(dim=-1)
        pay_pred = pay_logits.argmax(dim=-1)
        n_match = 0
        ptr_correct = (ptr_pred == gt_ptr).float().mean().item()
        op_correct = (op_pred == gt_op).float().mean().item()
        for b in range(B):
            cur = list(srcs[b])
            for k in range(K):
                cur = apply_edit(cur, int(ptr_pred[b, k].item()),
                                 int(op_pred[b, k].item()),
                                 int(pay_pred[b, k].item()))
            if cur == tgts[b]:
                n_match += 1
        return n_match / max(B, 1), float(ptr_correct), float(op_correct)


if __name__ == "__main__":
    task = SyntheticASTTask(seed=0)
    ids, gt_p, gt_o, gt_y, meta = task.generate_batch(8)
    print(f"input_ids: {tuple(ids.shape)}  gt_ptr: {tuple(gt_p.shape)}  "
          f"gt_op: {tuple(gt_o.shape)}  gt_pay: {tuple(gt_y.shape)}")
    print(f"vocab_size = {VOCAB_SIZE}, seq_len = {SEQ_LEN}, "
          f"script_start = {SCRIPT_START}, K_FIX = {K_FIX}")
    print(f"op_counts (NOP/RV/RO/SL across {8 * K_FIX} slots): {meta['op_counts']}")
    # GT-fed roundtrip: feed perfect predictions, expect EM=100%
    B = 8
    ptr_logits = torch.full((B, K_FIX, N_NODES), -1e9)
    op_logits = torch.full((B, K_FIX, N_EDIT_OPS), -1e9)
    pay_logits = torch.full((B, K_FIX, PAY_RANGE), -1e9)
    for b in range(B):
        for k in range(K_FIX):
            ptr_logits[b, k, int(gt_p[b, k])] = 0.0
            op_logits[b, k, int(gt_o[b, k])] = 0.0
            pay_logits[b, k, int(gt_y[b, k])] = 0.0
    em, ptr_acc, op_acc = SyntheticASTTask.exact_match(
        ids, ptr_logits, op_logits, pay_logits, gt_p, gt_o)
    print(f"GT-fed sanity: EM = {em * 100:.2f}% (expect 100%), "
          f"ptr_acc={ptr_acc*100:.2f}, op_acc={op_acc*100:.2f}")
    assert em == 1.0
    print("Task smoke test: PASS")
