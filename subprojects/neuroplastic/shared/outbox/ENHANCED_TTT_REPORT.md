# Enhanced TTT Evaluation Report

## Comparison

| Config | Steps | Params | Verified | V.Base | V.TTT | Delta | Solved | New Solves |
|--------|-------|--------|----------|--------|-------|-------|--------|-----------|
| A | 100 | geo | 102 | 64.1% | 74.9% | +10.8% | 2 | 2 |
| C | 500 | geo | 198 | 51.6% | 54.4% | +2.8% | 2 | 2 |
| D | 100 | geo+ffn | 116 | 61.0% | 72.4% | +11.4% | 2 | 2 |
| F | 500 | geo+ffn | 216 | 50.1% | 53.6% | +3.5% | 2 | 2 |

## Per-Config Details

### Config A: 100 steps, geo
- Evaluated: 279, Skipped: 121
- Gate: verified=102, partial=48, fallback=129
- Overall: base=47.1%, ttt=40.0%, verified=48.8%
- Solved: base=1, ttt=2, verified=2
- Time: 1193s (4.3s/task)

### Config C: 500 steps, geo
- Evaluated: 279, Skipped: 121
- Gate: verified=198, partial=54, fallback=27
- Overall: base=47.1%, ttt=38.7%, verified=47.9%
- Solved: base=1, ttt=2, verified=2
- Time: 3851s (13.8s/task)

### Config D: 100 steps, geo+ffn
- Evaluated: 279, Skipped: 121
- Gate: verified=116, partial=48, fallback=115
- Overall: base=47.1%, ttt=42.3%, verified=49.9%
- Solved: base=1, ttt=2, verified=2
- Time: 1137s (4.1s/task)

### Config F: 500 steps, geo+ffn
- Evaluated: 279, Skipped: 121
- Gate: verified=216, partial=53, fallback=10
- Overall: base=47.1%, ttt=41.1%, verified=48.8%
- Solved: base=1, ttt=2, verified=2
- Time: 3450s (12.4s/task)
