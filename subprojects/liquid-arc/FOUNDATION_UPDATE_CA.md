# FOUNDATION DOC UPDATE — Cellular Automata Results (March 2026)

## Add to Rules Section (after the auxiliary modules rule):

**Finding: The phase transition is TASK-CONTINGENT, not architectural.** Tested on cellular automata (local neighbor rules): CV reaches 3.0 and stays flat, model learns to 46% xform WITHOUT transitioning. The same architecture at CV 3.0 on ARC achieves only 15% xform (useless). The transition fires only when the task demands routing complexity that exceeds what CV ~3.0 provides. Local tasks (CA) are satisfiable with near-uniform routing; global tasks (ARC) are not.

## Update Open Question #3 (now answered):

3. **Does the transition transfer to non-ARC spatial tasks?** ANSWERED: No — tested on 2D cellular automata. The transition is task-contingent. CA rules are local (neighbor-dependent) and satisfiable with CV ~3.0 routing. The model achieves 46% xform on CA without transitioning, while ARC is at 15% at the same CV. The transition fires specifically when the task demands GLOBAL spatial routing that uniform/local attention cannot provide. This is a stronger finding than universal transfer — it means the transition is an adaptive response to computational demand, not an intrinsic training artifact. Analogous to biological critical periods that fire only in response to structured sensory input.

```
Cellular Automata Results:
  CV:        3.0 (stable from step 250, no acceleration)
  Loss:      3.23 → 1.32 (gradual, no collapse)
  Eval xform: 46% by step 500 (without transition)
  Tau σ:     0.073 at step 3000 (low — positions not strongly differentiated)
  |κ|:       0.0004 (flat metric works for local rules)
  
  Compare ARC at same training point:
  CV:        3.0-3.3 (plateau, eventually accelerates to 5.5+)
  Loss:      2.30 (flat, waiting for transition)
  Eval xform: 15-22% (useless pre-transition)
  Tau σ:     0.12-0.19 (strongly diversifying)
```

## Version History Addition:

- **2026-03-17 v4**: Cellular automata domain transfer experiment
  - Phase transition does NOT fire on CA (local rules don't demand non-trivial routing)
  - Model achieves 46% xform on CA without transitioning (at CV 3.0)
  - Key finding: transition is task-contingent, not architectural
  - Analogous to biological critical periods (triggered by sensory demand, not intrinsic program)
  - Clean ARC reproduction pending to verify codebase reproducibility
