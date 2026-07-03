# GRAPH ENGINE MCP — Live Test Validation

## Test Results: 9/10 correct

| # | Test | Result | Confidence | Notes |
|---|---|---|---|---|
| 1 | 5-hop root cause (bridge→shortage) | CORRECT | 1.0 | Full path traced |
| 2 | 8-hop root cause (earthquake→evacuation) | CORRECT | 1.0 | The test Qwen3-4B alone fails |
| 3 | Cross-chain connection (pesticide→sushi) | CORRECT | 0.99 | Correctly returns false |
| 4 | Cross-chain root cause (sushi→drought) | CORRECT | 0.5 | Right chain selected from 2 options |
| 5 | Scoped logic: junior→crypto≠linear_algebra | CORRECT | 0.91 | Scope constraint respected |
| 6 | Scoped logic: senior→crypto=linear_algebra | **WRONG** | 0.91 | Returns false, should be true |
| 7 | Isomorphic graphs (same topology) | CORRECT | cos=1.0 | Perfect detection |
| 8 | Non-isomorphic (different size) | CORRECT | cos=0.70 | Correctly distinguished |
| 9 | Non-isomorphic (same size, diff topology) | CORRECT | cos=0.78 | Chain vs diamond distinguished |
| 10 | Multi-chain root cause (3 chains, 9 nodes) | CORRECT | 0.33 | Right chain despite role-based clustering |

## Key Findings

### Root cause tracing: STRONG (4/4 correct)
- 5-hop and 8-hop chains traced with confidence 1.0
- Multi-chain scenarios correctly select the right chain
- The reachability fallback (networkx shortest_path filter) is load-bearing for multi-root scenarios — the head gives equal probability to same-type roots, but the filter selects only the one with a valid path to the target

### Scoped logic: PARTIAL (1/2 correct)
- "valid=false" case works (junior scope, crypto doesn't imply linear_algebra)
- "valid=true" case FAILS (senior scope, crypto SHOULD imply linear_algebra)
- The head appears biased toward "false" — possibly trained on skewed data, or scope masking doesn't fully remove out-of-scope paths from the node representations (nodes are always embedded, only edges are masked)

### Graph comparison: STRONG (3/3 correct)
- Identical topology: cosine 1.0 (perfect match)
- Different size graphs: cosine 0.70 (correctly non-isomorphic)
- Same size, different topology: cosine 0.78 (correctly distinguished)
- The 52-d signature approach works better in live testing than the 58% training accuracy suggested — possibly because real graphs have more distinct type patterns than the synthetic training data

### Connection check: CORRECT on this test (1/1)
- pesticide→sushi correctly returned false (prob 0.012)
- The 61% training accuracy was for harder cases with ambiguous connectivity — the clear disconnected case works

## Geometric Diagnostics — Validated

6-node chain:
```
CV = 3.09 (above threshold), D²/4τ = 21.1 (near target 18)
tau: root=0.51 (fast), intermediates≈1.0 (standard)
centrality: interior > endpoints (correct topology)
```

9-node multi-chain:
```
CV = 2.82, D²/4τ = 39.7
Clusters: [roots] [intermediates] [terminals] — cross-chain structural role detection
tau: perfectly symmetric across parallel chains
```

The MetricNet discovers structural roles from categorical embeddings. The TauNet assigns role-appropriate integration speed. The heat kernel respects the edge mask for within-chain routing.

## Bug: implication_check "valid=true" case

The head returns `valid: false` with confidence 0.91 even when the scoped edge IS active (senior_engineer scope, crypto→linear_algebra edge scoped to senior_engineer). 

Possible causes:
1. Training data skew (more "false" than "true" implication examples)
2. The scope masking gates EDGES but not NODES — alt_pathway is still embedded and visible to the head even when its incoming edge is masked. The head may detect alt_pathway's presence and infer "alternative exists" regardless of scope
3. The head may have learned a shortcut: "if the conclusion node has degree > 1, return false" — which works for most training examples but fails when one path is out of scope

Fix: either add scope-aware node filtering (mask nodes whose only incoming edges are out-of-scope) or add more "valid=true" training examples to balance the head.

## Overall Assessment

The graph MCP engine works as designed for its proven capabilities (root cause tracing, graph comparison) and partially for its known-weak capabilities (scoped logic, connection check). The geometric diagnostics provide rich, meaningful signal about graph structure. The system operates at criticality (D²/4τ ≈ 21, CV ≈ 3) as designed.

The 8-hop root cause trace is the headline capability — this is something Qwen3-4B alone CANNOT do (it stops at 3 hops), and the heat kernel provides through transitive diffusion over 16 ODE steps.
