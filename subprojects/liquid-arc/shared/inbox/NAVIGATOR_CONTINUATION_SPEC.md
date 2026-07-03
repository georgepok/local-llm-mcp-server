# GEOMETRIC NAVIGATOR — Continuation Spec

## What Phase 1 Proved

| Finding | Status | Evidence |
|---|---|---|
| h_state accumulates meaningful geometry | PROVEN | CV=2.97, 6 clusters, 52 nodes after 20 interactions |
| Cross-domain pattern transfer via signatures | PROVEN | Cosine ≥ 0.994 across supply-chain → ecology |
| LLM structure extraction works | PROVEN | Zero parse failures, semantically correct graphs |
| Navigator improves self-contained reasoning | DISPROVEN | 18/30 vs plain 20/30 — navigator is noise on simple problems |
| Simple networkx hint massively helps LLM | PROVEN | 29/30 — minimal structural hint is optimal for single problems |

## What Phase 1 Missed

The navigator was tested on self-contained problems where the answer is in the current message. This is the wrong test. The navigator's unique capabilities are:

1. **Persistent structural memory** — h_state carries structure across interactions that leave the LLM's context window
2. **Cross-domain pattern recognition** — metric signatures detect structural isomorphisms the LLM can't see
3. **Accumulated graph intelligence** — 100 interactions build a metric landscape no single query can reconstruct

None of these were tested in a scenario that requires them. The Phase 1 test suite was a maze test on a straight road.

## Phase 2 Design: The Long Session

### Core Test Structure

A 60-interaction evaluation session that REQUIRES accumulated state to answer the final questions. The plain LLM cannot hold 60 interactions in context. The navigator's h_state is the only component with cross-session structural memory.

```
Interactions 1-20:    Domain A (supply chain disruptions)
                      Build structural knowledge: factories, ports, routes, dependencies
                      Navigator accumulates: 40-60 nodes, cascade patterns, bottleneck patterns

Interactions 21-40:   Domain B (hospital operations)  
                      Build structural knowledge: departments, staff, equipment, protocols
                      Navigator accumulates: 80-120 nodes, new cascade patterns, scope patterns

Interactions 41-50:   Domain C (cybersecurity incidents)
                      Build structural knowledge: systems, vulnerabilities, attack paths
                      Navigator accumulates: 120-160 nodes, attack cascade patterns

Interactions 51-60:   CROSS-DOMAIN QUERIES (the actual test)
                      Questions that require connecting A, B, and/or C
                      Questions that require recalling specific structures from early interactions
                      Questions that require pattern recognition across domains
```

### Interaction Format

Each interaction is a short scenario (2-4 sentences) that introduces entities and relationships. The navigator extracts structure, merges into h_state, optionally provides hints. The LLM responds. This mimics real usage: a consultant working on multiple domains over days/weeks, building structural understanding.

```python
# Example interaction from Domain A (supply chain):
interaction_05 = {
    "user": "The Shanghai port congestion caused a 3-week delay for electronic "
            "components. This backed up the Shenzhen assembly plant, which couldn't "
            "fulfill orders for the Munich distribution center.",
    "expected_extraction": {
        "nodes": [
            {"id": "shanghai_congestion", "type": "event", "role": "root"},
            {"id": "component_delay", "type": "consequence", "role": "intermediate"},
            {"id": "shenzhen_backup", "type": "consequence", "role": "intermediate"},
            {"id": "munich_unfulfilled", "type": "consequence", "role": "terminal"}
        ],
        "edges": [
            {"src": "shanghai_congestion", "dst": "component_delay", "type": "causes"},
            {"src": "component_delay", "dst": "shenzhen_backup", "type": "causes"},
            {"src": "shenzhen_backup", "dst": "munich_unfulfilled", "type": "causes"}
        ]
    }
}
```

### The 10 Cross-Domain Queries (interactions 51-60)

These are designed so that the answer REQUIRES information the navigator accumulated but that is NOT in the current message.

**Type 1: Long-range recall (3 queries)**

The answer depends on a specific structural detail from interactions 1-20 that is no longer in any LLM context window.

```python
query_51 = {
    "user": "We just discovered that the Munich distribution center also supplies "
            "backup generators to City General Hospital. If Munich has fulfillment "
            "problems, what's the furthest upstream cause we've seen?",
    
    # Answer requires: recalling that Munich's problems traced to Shanghai port
    # congestion from interaction 5. This was 46 interactions ago — well outside
    # any LLM context window. Only h_state carries this.
    
    "expected_answer": "shanghai_congestion",
    "requires": "h_state recall of interaction 5 chain",
    "plain_llm_can_solve": False  # info not in current context
}
```

**Type 2: Cross-domain structural analogy (3 queries)**

The answer requires recognizing that a structure in domain C matches a pattern from domain A or B.

```python
query_54 = {
    "user": "The ransomware entered through an unpatched VPN, propagated to the "
            "domain controller, encrypted the file servers, which knocked out the "
            "billing system, shutting down patient intake. Does this remind you of "
            "any pattern we've seen before?",
    
    # Answer requires: pattern library matching this 5-node cascade to supply-chain
    # cascades from domain A. The metric signature should match at cosine > 0.9.
    # Plain LLM has no access to domain A patterns (they're 35+ interactions ago).
    
    "expected_answer": "cascade_failure pattern from supply chain domain",
    "requires": "pattern library cross-domain match",
    "plain_llm_can_solve": False  # domain A not in current context
}
```

**Type 3: Accumulated topology query (2 queries)**

The answer requires the navigator's global graph view — how many clusters, what are the hub nodes, where are the bottlenecks across all three domains.

```python
query_57 = {
    "user": "Across everything we've discussed — supply chains, hospital operations, "
            "and cybersecurity — what are the single points of failure? Entities that "
            "if they go down, cascade into the most downstream consequences?",
    
    # Answer requires: metric centrality analysis across the full 120+ node graph.
    # The navigator's per-node centrality in metric space identifies hub nodes.
    # Plain LLM can't compute centrality — it would need all 60 interactions in context.
    
    "expected_answer": "top-3 nodes by metric centrality across all domains",
    "requires": "full h_state graph + centrality computation",
    "plain_llm_can_solve": False  # requires global graph analysis
}
```

**Type 4: Scope transfer (2 queries)**

A scoped-logic pattern from domain B applied to a novel scenario using domain C entities.

```python
query_59 = {
    "user": "In the hospital, junior nurses couldn't prescribe controlled substances "
            "without a supervisor's cosign. We're now setting up cybersecurity "
            "authorization tiers. Should a tier-1 analyst be able to authorize "
            "a production firewall change without a senior architect's approval, "
            "given what we learned about scope dependencies?",
    
    # Answer requires: recognizing the scope pattern from hospital domain B
    # (junior scope → restricted action → requires senior approval) and
    # applying it to cybersecurity domain C. Navigator's pattern library
    # carries the scope_dependency signature.
    
    "expected_answer": "No — same scope-restricted pattern as hospital; tier-1 needs senior approval",
    "requires": "scope pattern transfer from domain B to domain C",
    "plain_llm_can_solve": False  # hospital scope pattern not in current context
}
```

### Dataset Generation

Generate the full 60-interaction dataset procedurally:

```python
def generate_session():
    """Generate a 60-interaction evaluation session.
    
    Domain A: 20 supply-chain interactions
    Domain B: 20 hospital operations interactions
    Domain C: 10 cybersecurity interactions
    Queries:  10 cross-domain queries
    
    Each interaction: 2-4 sentences introducing 2-5 entities and 1-4 relationships.
    Cross-domain queries designed to require h_state for correct answer.
    """
    
    # Domain A: supply chain
    # Build a complex supply chain network over 20 interactions:
    # Ports → shipping routes → factories → assembly plants → distribution centers
    # Include: 3 cascade failures, 2 bottleneck patterns, 1 scope dependency (customs)
    
    # Domain B: hospital operations  
    # Build hospital network over 20 interactions:
    # Departments → staff roles → equipment → protocols → patient pathways
    # Include: 2 cascade failures (equipment → department → patient impact),
    # 3 scope dependencies (junior/senior, department-specific protocols),
    # 1 bottleneck (single MRI machine)
    
    # Domain C: cybersecurity
    # Build network topology over 10 interactions:
    # Systems → vulnerabilities → attack paths → lateral movement → impact
    # Include: 2 cascade attacks, 1 scope issue (access tiers)
    
    # Cross-domain queries: 10 queries requiring cross-domain reasoning
    # Type 1 (recall): 3 queries requiring info from interactions 1-20
    # Type 2 (analogy): 3 queries requiring pattern match across domains
    # Type 3 (topology): 2 queries requiring global graph analysis
    # Type 4 (scope transfer): 2 queries requiring scope pattern transfer
    
    return interactions, queries
```

**Dataset size:** 60 interactions + 10 queries = 70 total items. Each interaction is 2-4 sentences. Each query is 1-3 sentences plus expected answer and required capability.

Generate 3 variants of the full session (different entities, same structural patterns) for statistical robustness. Total: 210 items across 3 variants.

### Evaluation Conditions

**Condition A: Plain LLM**

The LLM receives ONLY the current query plus the last 5 interactions as context (simulating realistic context window limitations). Interactions 1-45 are NOT in context.

```python
def condition_A(query, recent_5_interactions):
    prompt = f"""Recent context:
{format_interactions(recent_5_interactions)}

Current question:
{query["user"]}

Answer the question based on what you know."""
    return llm.generate(prompt)
```

**Condition B: LLM + Navigator**

The LLM receives the current query, the navigator's structural hint (from accumulated h_state), and the last 5 interactions. The navigator has processed all 50 prior interactions.

```python
def condition_B(query, recent_5_interactions, navigator):
    # Navigator processes the query
    nav_result = navigator.process_interaction(query["user"], [])
    
    hint_text = format_hint(nav_result["structural_hint"])
    context_nodes = nav_result["context_nodes"]
    
    # Retrieve text segments associated with relevant nodes
    relevant_context = retrieve_text_for_nodes(context_nodes)
    
    prompt = f"""Recent context:
{format_interactions(recent_5_interactions)}

Structural analysis from accumulated knowledge:
{hint_text}

Relevant historical context (from prior sessions):
{relevant_context}

Current question:
{query["user"]}

Answer the question using both the recent context and the structural analysis."""
    return llm.generate(prompt)
```

**Condition C: LLM + Full History (oracle upper bound)**

The LLM receives ALL 50 prior interactions plus the current query. This is the oracle — it has perfect memory but relies on the LLM's native attention to find relevant information in a massive context.

```python
def condition_C(query, all_50_interactions):
    prompt = f"""Complete interaction history:
{format_interactions(all_50_interactions)}

Current question:
{query["user"]}

Answer the question based on everything discussed above."""
    return llm.generate(prompt)
```

Note: Condition C may exceed the LLM's context window (50 interactions × ~100 tokens = ~5000 tokens, within Nemotron's limits but potentially degraded by lost-in-the-middle effects). This is realistic — even with full history, the LLM may not find the relevant needle.

**Condition D: LLM + Navigator (no pattern library)**

Same as B but with pattern library disabled. Tests whether the ODE geometry alone (without cross-domain pattern matching) provides value.

**Condition E: LLM + Networkx (with full graph history)**

The networkx tool has access to the full accumulated graph (same as navigator's h_state) but without geometric analysis — just BFS/DFS/shortest path. Tests whether graph algorithms alone provide the same value as geometric analysis.

### Scoring

Use LLM-as-judge with a DIFFERENT model than the answer model (to avoid self-evaluation bias). If Nemotron generates answers, use Qwen3-4B as judge (or vice versa). If both are unavailable for cross-evaluation, use the same model but with a structured rubric.

```python
JUDGE_PROMPT = """You are evaluating whether an AI answer correctly addresses a question.

Question: {question}
Expected answer: {expected}
Required capability: {requires}
AI's answer: {answer}

Score:
- CORRECT: answer contains the key information from expected answer
- PARTIAL: answer is on the right track but misses key details
- WRONG: answer is incorrect or doesn't address the question
- REFUSED: answer says it doesn't have enough information

Output ONLY one of: CORRECT, PARTIAL, WRONG, REFUSED"""
```

### Success Criteria

| # | Criterion | Gate |
|---|---|---|
| 1 | Navigator (B) > Plain LLM (A) on recall queries (type 1) | B correct on ≥ 2/3 where A is WRONG or REFUSED |
| 2 | Navigator (B) > Plain LLM (A) on analogy queries (type 2) | B correct on ≥ 2/3 where A is WRONG or REFUSED |
| 3 | Navigator (B) > Plain LLM (A) on topology queries (type 3) | B correct on ≥ 1/2 where A is WRONG or REFUSED |
| 4 | Navigator (B) ≥ Oracle (C) on at least 3/10 queries | Navigator's targeted retrieval beats brute-force context |
| 5 | Navigator (B) > Networkx-only (E) on analogy queries | Geometric signatures add value beyond graph algorithms |
| 6 | Navigator with patterns (B) > without patterns (D) on analogy queries | Pattern library specifically contributes |
| 7 | Zero regressions: B never WRONG where A is CORRECT | Navigator doesn't hurt on any query |
| 8 | Across 3 session variants: results consistent (±1 per condition) | Not a single-session fluke |

The critical gates are 1-3: does the navigator help when the answer ISN'T in the current context? If the plain LLM can't answer (because the information is 46 interactions ago), but the navigator can (because h_state carries it), that's the unique value proposition validated.

Gate 4 tests something interesting: can targeted geometric retrieval (navigator selects 5-10 relevant nodes from 120+) beat brute-force full-history (LLM sees everything but has to find the needle)? If yes, the navigator provides more efficient context composition than raw context stuffing.

## Phase 2 Fixes From Phase 1

### Fix 1: Dual retrieval mode

Phase 1 showed the metric clusters by type/role, not by causal chain. Add graph-adjacency retrieval alongside metric retrieval:

```python
class GeometricState:
    def query_relevant(self, query_nodes, k=10, mode="metric"):
        """
        mode="metric":  return nodes metrically close (same type/role/topology)
        mode="graph":   return nodes reachable via edges (same causal chain)
        mode="both":    union of both, deduplicated, sorted by combined score
        """
        if mode == "metric":
            return self._query_metric(query_nodes, k)
        elif mode == "graph":
            return self._query_graph_adjacent(query_nodes, k)
        elif mode == "both":
            metric_results = self._query_metric(query_nodes, k)
            graph_results = self._query_graph_adjacent(query_nodes, k)
            return self._merge_results(metric_results, graph_results, k)
```

For the cross-domain queries, use `mode="both"`: metric retrieval finds structurally similar nodes across domains, graph retrieval finds causally connected nodes within the same domain. The union provides the most relevant context.

### Fix 2: Minimal hints for simple queries

Phase 1 showed the navigator's verbose hints are noise on simple problems. Apply a confidence-based hint verbosity policy:

```python
class HintGenerator:
    def generate_hint(self, analysis, h_state, pattern_match=None):
        confidence = analysis.get("confidence", 0)
        
        if confidence > 0.95:
            # High confidence: minimal hint (like networkx)
            # Just the answer, no context padding
            return self._minimal_hint(analysis)
        elif confidence > 0.7:
            # Medium confidence: answer + brief supporting evidence
            return self._standard_hint(analysis, pattern_match)
        else:
            # Low confidence: full context (metric-nearest, patterns, caveats)
            return self._full_hint(analysis, h_state, pattern_match)
```

This ensures simple problems get simple hints (matching networkx's 29/30 performance) while complex/uncertain problems get the full geometric context.

### Fix 3: Text segment retrieval for context composition

The navigator returns relevant node IDs. But the LLM needs TEXT associated with those nodes for context composition. Add a text segment store:

```python
class GeometricState:
    def __init__(self, ...):
        # ... existing fields ...
        self.text_segments = {}  # node_id → list of text snippets that mention this node
    
    def merge_fragment(self, graph_fragment, source_text=None):
        # ... existing merge logic ...
        
        # Store the source text associated with each node
        if source_text:
            for node in graph_fragment["nodes"]:
                nid = node["id"]
                if nid not in self.text_segments:
                    self.text_segments[nid] = []
                self.text_segments[nid].append({
                    "text": source_text,
                    "timestamp": time.time(),
                    "interaction_index": self._interaction_count
                })
    
    def retrieve_text_for_nodes(self, node_ids, max_segments=5):
        """Get text snippets associated with given nodes.
        
        Returns the most recent segments, deduplicated.
        """
        segments = []
        for nid in node_ids:
            if nid in self.text_segments:
                segments.extend(self.text_segments[nid])
        
        # Deduplicate and sort by recency
        seen = set()
        unique = []
        for seg in sorted(segments, key=lambda s: s["timestamp"], reverse=True):
            if seg["text"] not in seen:
                seen.add(seg["text"])
                unique.append(seg)
        
        return unique[:max_segments]
```

This bridges the gap: navigator selects relevant nodes geometrically → retrieves associated text segments → provides them to the LLM as historical context. The LLM reads text. The navigator provides structure. Clean separation.

## Implementation Plan

### Week 1: Dataset generation + fixes

1. Generate 3 variants of the 60-interaction session (script: `gen_nav_session.py`)
2. Implement dual retrieval mode (metric + graph adjacency)
3. Implement confidence-based hint verbosity
4. Implement text segment storage and retrieval
5. Test fixes on Phase 1 Experiment 4 problems (regression check)

### Week 2: Run Phase 2 experiments

1. Run all 3 session variants × 5 conditions (A, B, C, D, E) = 15 runs
2. Score with LLM-as-judge
3. Compile results per query type and condition
4. Evaluate 8 success criteria

### Week 3: Analysis and iteration

1. Analyze failures: where does the navigator fail that it shouldn't?
2. Check h_state health after 50 interactions: CV, D²/4τ, cluster quality
3. Analyze pattern library: how many patterns accumulated? Do they match correctly?
4. If criteria met: document as validated. If not: diagnose specific failure mode.

## Compute Requirements

- 60 interactions × 5 conditions × 3 variants = 900 LLM calls for answers
- 10 queries × 5 conditions × 3 variants = 150 judge calls
- Navigator processing: 50 merge + ODE calls per session variant = 150 total
- Estimated wall time: 4-6 hours per variant (Nemotron on Spark), ~15 hours total
- Memory: navigator h_state at 160 nodes ≈ 200MB GPU, well within Spark capacity

## What This Tests That Phase 1 Didn't

| Phase 1 test | Phase 2 test |
|---|---|
| Self-contained problems | Information spread across 50+ interactions |
| Answer in current message | Answer requires recall from interaction #5 |
| Single domain | Three domains, cross-domain queries |
| No context limitation | Plain LLM sees only last 5 interactions |
| Navigator vs no-tools | Navigator vs oracle (full history) vs graph-only |
| Pattern library on synthetic pairs | Pattern library on naturally accumulated structures |

Phase 2 tests the navigator in its intended operating regime: long-running, multi-domain, accumulated structural intelligence where the relevant context ISN'T in the current message. If it works here, it validates the unique value proposition. If it fails here, the architecture doesn't provide what the theory predicts.
