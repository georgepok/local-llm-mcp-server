# Phase 1 — Correction: Task 4 Priority

**From:** Claude Desktop  
**Date:** 2026-03-10  

---

## Correction to Task 4

The `fluid_geometry.py` and `nano_v3_reasoning_parser.py` mounted in the vllm container are **leftover artifacts** from previous FGN research work. They're not required for current serving and may or may not be relevant to the neuroplastic project.

**Revised priority:** Task 4 is now **low priority / optional**. Don't spend significant time analyzing these files. If it's a quick read, great — document what they do and note them as available infrastructure. If it requires investigation, skip it and move on.

**Revised task order:**
1. **Task 1: Blueprint prompt** — highest priority, everything depends on this
2. **Task 2: Evaluation harness** — needed before any modifications
3. **Task 3: Checkpoint/rollback investigation** — needed for Phase 2
4. **Task 4: Fluid geometry files** — optional, read if time permits

The decision on whether to use the existing geometric infrastructure as a modification pathway vs. going straight to weight-level modification is a research design choice that Claude Desktop will make after seeing Phase 1 results. The agent should focus on the core deliverables.

**One additional note on the checkpoint/rollback investigation (Task 3):** Since the fluid_geometry logits processor is a leftover and may be removed, check whether vllm serves cleanly without the `--logits-processors` flag. If so, a cleaner deployment without the leftover artifacts might be the right baseline to work from. But don't restart vllm without confirming with George first — just document the finding.

---

*This supersedes the priority order in phase1_self_model_construction.md §8.*
