# GEOMETRIC NAVIGATOR — Experiment Spec

## Core Principle

The navigator NEVER processes text. It operates exclusively on typed graphs — the domain where LiquidARC is proven (100% root cause, 10× on LeWM dynamics, phase transitions on ARC). The LLM handles all text ↔ structure translation. Communication is always symbolic.

```
Text world (LLM)              Structure world (Navigator)
──────────────                 ──────────────────────────
user text ──→ LLM extracts ──→ typed graph fragment
                                    │
                                    ▼
                               ODE + MetricNet + heat kernel
                               Graph engine (root cause, scope, analogy)
                               h_state accumulation
                               Pattern library matching
                                    │
                                    ▼
LLM renders ←── structural hint ←── {node_ids, chains, framing_type}
response text
```

## What Already Exists (reuse directly)

| Component | Location | Status |
|---|---|---|
| ContinuousDynamics (low-rank metric) | `liquid_arc/dynamics.py` | Proven (LeWM 10×) |
| Criticality scaffolding | `liquid_arc/sustained_criticality.py` | Proven (14× on LeWM, 2-3× on ARC) |
| GraphNodeEmbedding | `liquid_arc/graph_embed.py` | Proven (CV=3+, phase transitions) |
| Graph engine (analyze, compare, diagnostics) | `liquid_arc/graph_engine_inference.py` | Proven (9/10 live tests) |
| MCP server | `liquid_arc/mcp_graph_serve.py` | Running on Spark port 8420 |
| Online correction (correct_answer) | `liquid_arc/mcp_graph_serve.py` | Tested (with collapse safeguard) |
| Edge mask (causal, scope-aware) | `liquid_arc/graph_mask.py` | Proven |
| Structural features | `liquid_arc/graph_features.py` | Proven |
| Checkpoint | `output_graph_engine_final/checkpoints/step_500.pt` | Deployed |

## What's New (build these)

### 1. Persistent h_state Manager

Accumulates graph structure across interactions. NOT text. NOT embeddings. Graph nodes and their metric positions.

```python
class GeometricState:
    """Persistent geometric state across interactions.
    
    Stores: accumulated graph structure + metric landscape.
    Never stores: text, token embeddings, raw strings.
    Persists: to disk as JSON, loaded on restart.
    """
    def __init__(self, state_path, dynamics, max_nodes=512):
        self.state_path = state_path  # e.g., /workspace/liquid-arc/navigator_state.json
        self.dynamics = dynamics       # frozen ContinuousDynamics
        self.max_nodes = max_nodes
        
        # The accumulated graph
        self.nodes = {}       # id → {type, role, first_seen, last_seen, mention_count}
        self.edges = []       # [{src, dst, type, scope?, weight}]
        self.embeddings = {}  # id → tensor [d_model] — ODE-processed position
        
        # Metric landscape (computed from embeddings via MetricNet)
        self.clusters = []    # [{cluster_id, members, centroid}]
        self.signatures = []  # [{signature_52d, label, source_interaction, count}]
        
        # Load from disk if exists
        self._load()
    
    def merge_fragment(self, graph_fragment):
        """Merge a new graph fragment (from LLM extraction) into persistent state.
        
        Args:
            graph_fragment: {
                "nodes": [{"id": str, "type": str, "role": str}, ...],
                "edges": [{"src": str, "dst": str, "type": str}, ...]
            }
        
        New nodes get added. Existing nodes get mention_count incremented.
        New edges get added. Existing edges get weight incremented.
        After merge, re-run ODE on the updated graph to refresh embeddings.
        """
        for node in graph_fragment["nodes"]:
            nid = node["id"]
            if nid in self.nodes:
                self.nodes[nid]["mention_count"] += 1
                self.nodes[nid]["last_seen"] = time.time()
            else:
                self.nodes[nid] = {
                    "type": node.get("type", "entity"),
                    "role": node.get("role", "intermediate"),
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                    "mention_count": 1
                }
        
        for edge in graph_fragment["edges"]:
            existing = self._find_edge(edge["src"], edge["dst"], edge["type"])
            if existing:
                existing["weight"] += 1
            else:
                self.edges.append({
                    "src": edge["src"], "dst": edge["dst"],
                    "type": edge.get("type", "related_to"),
                    "scope": edge.get("scope", None),
                    "weight": 1
                })
        
        # Prune if over max_nodes (remove least-recently-seen)
        if len(self.nodes) > self.max_nodes:
            self._prune_oldest(self.max_nodes - len(self.nodes))
        
        # Re-run ODE on full graph to update metric positions
        self._recompute_geometry()
        self._save()
    
    def query_relevant(self, query_nodes, k=10):
        """Find the k nodes most metrically close to query_nodes.
        
        Args:
            query_nodes: list of node IDs (extracted from current query)
            k: how many relevant nodes to return
            
        Returns:
            list of {id, type, metric_distance, cluster_id}
            sorted by metric distance (closest first)
        """
        if not query_nodes or not self.embeddings:
            return []
        
        # Get query embeddings
        q_embs = [self.embeddings[n] for n in query_nodes if n in self.embeddings]
        if not q_embs:
            return []
        q_mean = torch.stack(q_embs).mean(dim=0)
        
        # Compute metric distances to all nodes
        distances = {}
        for nid, emb in self.embeddings.items():
            if nid not in query_nodes:
                # D² in metric space (using current MetricNet g)
                d_sq = self._metric_distance(q_mean, emb)
                distances[nid] = d_sq
        
        # Return k closest
        sorted_nodes = sorted(distances.items(), key=lambda x: x[1])[:k]
        return [
            {"id": nid, "type": self.nodes[nid]["type"], 
             "metric_distance": dist, "cluster_id": self._get_cluster(nid)}
            for nid, dist in sorted_nodes
        ]
    
    def get_signature(self):
        """Compute current metric signature of the full accumulated graph.
        
        Returns 52-d vector: 20 geometric dims + 32 structural feature dims.
        Used for pattern library matching.
        """
        return compute_graph_signature(self.embeddings, self.dynamics)
    
    def _recompute_geometry(self):
        """Run ODE on the full accumulated graph. Updates embeddings and clusters."""
        graph_json = self._to_graph_json()
        h = embed_graph(graph_json, self.dynamics.graph_embed)
        mask = build_edge_mask(graph_json)
        h_out = ode_integrate(self.dynamics, h, mask, n_steps=16)
        
        # Store per-node ODE output as metric position
        for i, nid in enumerate(self.nodes.keys()):
            self.embeddings[nid] = h_out[0, i, :].detach()
        
        # Recompute clusters via single-link on pairwise D²
        self.clusters = compute_metric_clusters(self.embeddings, self.dynamics)
    
    def _save(self):
        """Persist to disk. Embeddings stored as lists for JSON serialization."""
        state = {
            "nodes": self.nodes,
            "edges": self.edges,
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
            "clusters": self.clusters,
            "signatures": self.signatures,
        }
        with open(self.state_path, 'w') as f:
            json.dump(state, f)
    
    def _load(self):
        """Load from disk if exists."""
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r') as f:
                state = json.load(f)
            self.nodes = state["nodes"]
            self.edges = state["edges"]
            self.embeddings = {k: torch.tensor(v) for k, v in state["embeddings"].items()}
            self.clusters = state["clusters"]
            self.signatures = state.get("signatures", [])
```

### 2. Structure Extraction Prompt

The LLM extracts graph fragments from text. This is a PROMPT, not a neural module. LLMs are excellent at structured extraction.

```python
EXTRACT_GRAPH_PROMPT = """Extract entities and relationships from the following text as a JSON graph.

Rules:
- Each entity becomes a node with: id (snake_case), type (one of: event, consequence, state, cause, role, credential, requirement, prerequisite, concept, entity), role (one of: root, intermediate, terminal, scope)
- Each relationship becomes an edge with: src, dst, type (one of: causes, requires, precedes, enables, depends_on, related_to, blocks, is_a)
- If a relationship only applies within a specific scope/context, add "scope": "<scope_node_id>"
- Extract ONLY what's explicitly stated or clearly implied. Don't infer.
- Keep ids short and descriptive.

Text:
{text}

Respond with ONLY valid JSON, no explanation:
{{"nodes": [...], "edges": [...]}}"""
```

This runs through the LLM (Qwen3-4B, Nemotron, or any available model). The navigator receives the structured output. No text touches the ODE.

### 3. Structural Hint Generator

Translates navigator's geometric analysis into a structured hint the LLM can use.

```python
class HintGenerator:
    """Convert geometric analysis into structural hints for LLM.
    
    Output is a structured dict, NOT text. The LLM's prompt template
    renders it into natural language.
    """
    def generate_hint(self, analysis, h_state, pattern_match=None):
        """
        Args:
            analysis: output from graph engine (root_cause, connection, etc.)
            h_state: current GeometricState
            pattern_match: optional match from pattern library
        
        Returns:
            hint dict consumed by LLM prompt template
        """
        hint = {
            "analysis_type": analysis.get("type"),
            "confidence": analysis.get("confidence"),
        }
        
        if analysis.get("type") == "root_cause":
            hint["chain"] = analysis["path"]
            hint["root"] = analysis["root_cause"]
            hint["hops"] = analysis["hops"]
            
        if analysis.get("type") == "connection_check":
            hint["connected"] = analysis["connected"]
            hint["probability"] = analysis.get("connected_head_prob")
            
        if analysis.get("type") == "implication_check":
            hint["valid"] = analysis["valid"]
            hint["scope_filtered_edges"] = analysis.get("n_edges_after_scope_filter")
        
        # Add relevant context from h_state
        relevant = h_state.query_relevant(
            [analysis.get("target", analysis.get("src"))], k=5
        )
        if relevant:
            hint["related_context"] = [
                {"id": r["id"], "type": r["type"], "distance": r["metric_distance"]}
                for r in relevant
            ]
        
        # Add pattern match if available
        if pattern_match:
            hint["known_pattern"] = {
                "label": pattern_match["label"],
                "similarity": pattern_match["similarity"],
                "prior_occurrences": pattern_match["count"]
            }
        
        return hint


HINT_TEMPLATE = """The following structural analysis is available for your response:

Analysis type: {analysis_type}
{chain_section}
{connection_section}
{implication_section}
Confidence: {confidence}

{context_section}
{pattern_section}

Use this structural information to inform your answer. The analysis is based on explicit graph relationships extracted from the conversation."""
```

### 4. Navigator Orchestrator

The main loop that ties everything together.

```python
class GeometricNavigator:
    """Orchestrates geometric reasoning alongside LLM interaction.
    
    The navigator NEVER processes text. It operates on typed graphs.
    The LLM translates text ↔ structure in both directions.
    """
    def __init__(self, graph_engine, h_state, llm_client, pattern_library_path):
        self.engine = graph_engine           # existing GraphEngine
        self.state = h_state                 # GeometricState (persistent)
        self.llm = llm_client               # LLM for text ↔ structure
        self.hints = HintGenerator()
        self.pattern_library = PatternLibrary(pattern_library_path)
    
    def process_interaction(self, user_text, conversation_history):
        """Main entry point for each user interaction.
        
        Args:
            user_text: raw user message (string)
            conversation_history: list of prior messages (for LLM context)
        
        Returns:
            {
                "structural_hint": dict (for LLM to use in response generation),
                "context_nodes": list of relevant node IDs (for context composition),
                "diagnostics": dict (CV, D²/4τ, clusters — for monitoring)
            }
        """
        # Step 1: LLM extracts graph fragment from user text
        graph_fragment = self._extract_structure(user_text, conversation_history)
        
        if not graph_fragment or not graph_fragment.get("nodes"):
            # No structure detected — pass through to LLM without geometric analysis
            return {"structural_hint": None, "context_nodes": [], "diagnostics": {}}
        
        # Step 2: Merge into persistent state
        self.state.merge_fragment(graph_fragment)
        
        # Step 3: Determine query type and run appropriate analysis
        query = self._infer_query(graph_fragment, user_text)
        
        analysis = None
        if query:
            # Run graph engine on the RELEVANT SUBGRAPH (not full h_state)
            subgraph = self._extract_relevant_subgraph(query)
            analysis = self.engine.analyze_graph(
                json.dumps(subgraph), json.dumps(query)
            )
            analysis = json.loads(analysis) if isinstance(analysis, str) else analysis
        
        # Step 4: Check pattern library
        current_sig = self.state.get_signature()
        pattern_match = self.pattern_library.find_nearest(current_sig, threshold=0.85)
        
        # Step 5: Get relevant context nodes (for LLM context composition)
        query_nodes = [n["id"] for n in graph_fragment["nodes"]]
        context_nodes = self.state.query_relevant(query_nodes, k=10)
        
        # Step 6: Generate structural hint
        hint = None
        if analysis:
            hint = self.hints.generate_hint(analysis, self.state, pattern_match)
        
        # Step 7: Store pattern if novel
        if analysis and not pattern_match:
            self.pattern_library.store(current_sig, {
                "label": self._auto_label(analysis),
                "source_query": query,
                "timestamp": time.time()
            })
        
        # Step 8: Diagnostics
        diagnostics = json.loads(
            self.engine.get_graph_diagnostics(json.dumps(self._state_to_graph()))
        ) if len(self.state.nodes) > 2 else {}
        
        return {
            "structural_hint": hint,
            "context_nodes": context_nodes,
            "diagnostics": diagnostics
        }
    
    def _extract_structure(self, text, history):
        """Use LLM to extract graph fragment from text."""
        prompt = EXTRACT_GRAPH_PROMPT.format(text=text)
        response = self.llm.generate(prompt, max_tokens=500)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None
    
    def _infer_query(self, fragment, user_text):
        """Determine what type of graph query the user is asking.
        
        Uses simple heuristics on the fragment structure, NOT text processing.
        """
        terminals = [n for n in fragment["nodes"] if n.get("role") == "terminal"]
        roots = [n for n in fragment["nodes"] if n.get("role") == "root"]
        scopes = [n for n in fragment["nodes"] if n.get("role") == "scope"]
        
        # If there are terminals and causal edges → likely root_cause query
        causal_edges = [e for e in fragment["edges"] if e["type"] in ("causes", "precedes")]
        if terminals and causal_edges:
            return {"type": "root_cause", "target": terminals[0]["id"]}
        
        # If there are scope nodes → likely implication_check
        if scopes and len(fragment["nodes"]) > 3:
            prereqs = [n for n in fragment["nodes"] if n["type"] == "prerequisite"]
            reqs = [n for n in fragment["nodes"] if n["type"] == "requirement"]
            if prereqs and reqs:
                return {
                    "type": "implication_check",
                    "premise": reqs[0]["id"],
                    "conclusion": prereqs[0]["id"],
                    "context_scope": scopes[0]["id"]
                }
        
        # If disconnected subgraphs → likely connection_check
        if self._has_disconnected_components(fragment):
            components = self._get_components(fragment)
            if len(components) >= 2:
                src = components[0][0]["id"]
                dst = components[1][0]["id"]
                return {"type": "connection_check", "src": src, "dst": dst}
        
        # Default: no specific query, just accumulate structure
        return None
    
    def _extract_relevant_subgraph(self, query):
        """Extract the subgraph around the query target from h_state.
        
        Uses metric distance: include nodes within 2× the median metric distance
        of the query target. This gives the ODE the relevant local context.
        """
        target_id = query.get("target", query.get("src", query.get("premise")))
        if not target_id or target_id not in self.state.embeddings:
            return self._state_to_graph()  # fallback: full graph
        
        relevant = self.state.query_relevant([target_id], k=20)
        relevant_ids = set([target_id] + [r["id"] for r in relevant])
        
        # Also include query-specific nodes
        for key in ["target", "src", "dst", "premise", "conclusion", "context_scope"]:
            if key in query and query[key] in self.state.nodes:
                relevant_ids.add(query[key])
        
        # Build subgraph
        nodes = [
            {"id": nid, "type": self.state.nodes[nid]["type"],
             "role": self.state.nodes[nid]["role"]}
            for nid in relevant_ids if nid in self.state.nodes
        ]
        edges = [
            e for e in self.state.edges
            if e["src"] in relevant_ids and e["dst"] in relevant_ids
        ]
        
        return {"nodes": nodes, "edges": edges}
```

### 5. Pattern Library

Stores metric signatures of discovered structural patterns. Enables cross-session transfer.

```python
class PatternLibrary:
    """Persistent library of metric signatures for structural pattern matching.
    
    Each pattern: 52-d signature + label + metadata.
    Lookup: cosine similarity against query signature.
    """
    def __init__(self, library_path):
        self.library_path = library_path
        self.patterns = []
        self._load()
    
    def find_nearest(self, signature, threshold=0.85):
        """Find the pattern most similar to the query signature."""
        if not self.patterns:
            return None
        
        sig_tensor = torch.tensor(signature)
        best_match = None
        best_sim = -1
        
        for pattern in self.patterns:
            pat_tensor = torch.tensor(pattern["signature"])
            sim = F.cosine_similarity(sig_tensor.unsqueeze(0), pat_tensor.unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_match = pattern
        
        if best_sim >= threshold:
            return {**best_match, "similarity": best_sim}
        return None
    
    def store(self, signature, metadata):
        """Store a new pattern."""
        self.patterns.append({
            "signature": signature if isinstance(signature, list) else signature.tolist(),
            "label": metadata.get("label", "unnamed"),
            "source_query": metadata.get("source_query"),
            "timestamp": metadata.get("timestamp", time.time()),
            "count": 1
        })
        self._save()
    
    def _save(self):
        with open(self.library_path, 'w') as f:
            json.dump(self.patterns, f)
    
    def _load(self):
        if os.path.exists(self.library_path):
            with open(self.library_path, 'r') as f:
                self.patterns = json.load(f)

```

### 6. MCP Integration (extend existing server)

Add navigator tools to the existing `mcp_graph_serve.py`:

```python
@tool
def navigate(user_text: str, conversation_history: str = "[]") -> str:
    """Process a user interaction through the geometric navigator.
    
    Extracts structure from text (via LLM), accumulates in persistent state,
    runs geometric analysis, returns structural hint for response generation.
    
    Args:
        user_text: the user's message
        conversation_history: JSON list of prior messages (optional)
    
    Returns:
        JSON with structural_hint, context_nodes, diagnostics
    """
    history = json.loads(conversation_history) if conversation_history else []
    result = navigator.process_interaction(user_text, history)
    return json.dumps(result, default=str)

@tool
def get_navigator_state() -> str:
    """Return the current geometric state: nodes, clusters, patterns, diagnostics."""
    return json.dumps({
        "n_nodes": len(navigator.state.nodes),
        "n_edges": len(navigator.state.edges),
        "n_patterns": len(navigator.pattern_library.patterns),
        "clusters": navigator.state.clusters,
        "recent_nodes": sorted(
            navigator.state.nodes.items(),
            key=lambda x: x[1]["last_seen"], reverse=True
        )[:10]
    }, default=str)

@tool
def reset_navigator_state() -> str:
    """Clear the persistent geometric state. Use with caution."""
    navigator.state.nodes = {}
    navigator.state.edges = []
    navigator.state.embeddings = {}
    navigator.state.clusters = []
    navigator.state._save()
    return json.dumps({"reset": True})
```

## Experiment Design

### Experiment 1: Structure Extraction Validation

Verify the LLM reliably extracts graph fragments from natural text.

**Test:** 30 text passages of varying complexity (simple causal statements, multi-chain narratives, scoped logic descriptions). Manually annotate the expected graph. Measure extraction accuracy.

**Metric:** Node F1, edge F1, type accuracy.

**Pass criterion:** Node F1 > 0.85, edge F1 > 0.75. If the LLM can't reliably extract structure, the whole pipeline fails at step 1.

**LLM:** Use Nemotron-30B (on Spark via vLLM) or Qwen3-4B. Test both.

### Experiment 2: h_state Accumulation

Verify that persistent state accumulates meaningful structure across interactions.

**Test:** Feed 20 interactions from a single domain (e.g., supply chain disruptions). Each interaction introduces 2-5 new entities and 1-3 new relationships. After 20 interactions, the h_state should contain a rich graph with meaningful metric structure.

**Metrics:**
- Graph size: should grow to 30-60 nodes, 40-80 edges
- Metric clusters: should form domain-meaningful groups (causes, effects, entities)
- CV: should be > 2.0 (geometric structure present)
- Pattern library: should contain 3-5 discovered patterns

**Pass criterion:** CV > 2.0 after 20 interactions. At least 3 metric clusters formed. State persists correctly across simulated restarts.

### Experiment 3: Context Relevance

Verify that metric-based context selection outperforms recency-based selection.

**Test:** After accumulating 20 interactions (from Exp 2), present a query that's semantically related to interaction #5 but not to interactions #18-20. 

**Compare:**
- Recency-based: returns nodes from interactions #18-20 (most recent)
- Metric-based: returns nodes metrically close to query (should include nodes from interaction #5)

**Metric:** Does the metric-based selection include the actually-relevant nodes that recency misses?

**Pass criterion:** Metric-based selection retrieves at least 2 relevant nodes that recency-based misses on 5 out of 10 test queries.

### Experiment 4: End-to-End Reasoning Improvement

The main test: does the navigator improve LLM reasoning on structural tasks?

**Test suite:** 30 problems across 3 categories:
- 10 multi-hop causal reasoning (5-8 hops, multi-chain, shared vocabulary)
- 10 scoped logic (nested scopes, multiple dependency paths)  
- 10 cross-session structural transfer (present pattern in session 1, test recognition in session 2)

**Conditions:**
- A) Plain LLM (no navigator)
- B) LLM + navigator (full pipeline: extraction → h_state → analysis → hint)
- C) LLM + hand-written graph tools (networkx BFS/DFS, no geometric analysis)

**Metric:** Accuracy on each problem (correct/incorrect, scored by LLM-as-judge to avoid keyword fragility).

**Pass criteria:**
- B > A on at least 6 of 30 problems (20% improvement)
- B > C on at least 3 of 30 problems (navigator adds value beyond algorithmic tools)
- B has 0 regressions vs A (no problems where navigator hurts)
- Cross-session transfer: B solves at least 5/10 transfer problems, A solves < 3/10

### Experiment 5: Pattern Library Cross-Domain Transfer

The civilization hypothesis: patterns discovered in one domain transfer to another.

**Test:** 
- Phase 1: Navigator processes 20 supply-chain interactions. Accumulates patterns (cascade failures, bottleneck detection, etc.)
- Phase 2: Present an ECOLOGY problem with isomorphic structure (predator removal → cascading ecosystem collapse). Navigator has never seen ecology.

**Metric:** Does the pattern library match the ecology problem to a supply-chain pattern? Does the structural hint improve the LLM's reasoning about the ecology problem?

**Pass criterion:** Pattern library matches with cosine > 0.80. LLM with hint produces more structurally accurate response than without.

## Implementation Phases

### Phase 1: Structure extraction + h_state (Week 1)

1. Implement GeometricState (merge, query_relevant, save/load, recompute_geometry)
2. Write EXTRACT_GRAPH_PROMPT, test on Nemotron/Qwen3
3. Run Experiment 1 (extraction validation)
4. Run Experiment 2 (accumulation validation)

Deliverable: working h_state that accumulates graph structure across interactions.

### Phase 2: Navigator orchestrator + hints (Week 2)

1. Implement HintGenerator and GeometricNavigator
2. Implement _infer_query (structural query detection)
3. Implement _extract_relevant_subgraph (metric-based subgraph selection)
4. Run Experiment 3 (context relevance)
5. Add MCP tools (navigate, get_navigator_state)

Deliverable: working navigator that accepts text, extracts structure, runs analysis, returns hints.

### Phase 3: Full evaluation + pattern library (Week 3)

1. Implement PatternLibrary (store, find_nearest, persist)
2. Build the 30-problem test suite
3. Run Experiment 4 (end-to-end reasoning improvement)
4. Run Experiment 5 (cross-domain transfer)

Deliverable: complete results on all 5 experiments.

## Architecture Notes for Implementation

### LLM for structure extraction

The extraction LLM doesn't need to be the same as the generation LLM. Options:

- **Nemotron-30B** (on Spark via vLLM): best extraction quality, already running
- **Qwen3-4B** (lighter): faster, may suffice for simple extraction
- **The calling LLM itself** (Claude/GPT): if navigator is used via MCP from Claude Desktop, Claude can do the extraction — but this adds latency

For experiments: use Nemotron-30B. For deployment: allow configurable extraction model.

### Subgraph sizing

The graph engine works best on graphs of 3-20 nodes (training distribution). The h_state may accumulate 100+ nodes. The _extract_relevant_subgraph method must select a manageable subgraph around the query. Use metric distance as the selection criterion — include nodes within 2× median metric distance of the query target, capped at 20 nodes.

### h_state recomputation cost

Every merge_fragment triggers a full ODE recomputation on the accumulated graph. At N=100 nodes, this is 16 ODE steps over [1, 100, d_model] — about 50-100ms on Spark GPU. Acceptable for interactive use (human typing is slower). At N=500, might need optimization (incremental update instead of full recompute).

### What the navigator does NOT do

- Process text through the ODE (NEVER — the ODE sees only typed graphs)
- Modify the LLM's internal computation (no hooks, no biases, no attention perturbation)
- Generate text (the LLM generates all text; navigator provides structural hints)
- Replace the LLM on any task (navigator is a structural reasoning assistant, not a language model)
- Make autonomous decisions (navigator provides analysis; the LLM or user decides what to do)

## Success Criteria Summary

| # | Criterion | Experiment | Target |
|---|---|---|---|
| 1 | LLM extracts structure reliably | Exp 1 | Node F1 > 0.85 |
| 2 | h_state accumulates meaningful geometry | Exp 2 | CV > 2.0 after 20 interactions |
| 3 | Metric selection beats recency | Exp 3 | Finds relevant nodes recency misses |
| 4 | Navigator improves LLM reasoning | Exp 4 | ≥6/30 improvements, 0 regressions |
| 5 | Patterns transfer cross-domain | Exp 5 | Cosine > 0.80 on isomorphic structure |

## Why This Should Work

Every component operates in its PROVEN domain:

- ODE + low-rank metric: proven on LeWM dynamics (10×) and ARC (2-3×)
- Graph engine: proven on typed graphs (100% root cause, 9/10 live)
- Criticality: proven to prevent collapse and enable scaling
- Pattern library (cosine on signatures): proven on graph comparison (3/3 correct)
- LLM as text interface: LLMs are excellent at structured extraction

What's new is the ORCHESTRATION — connecting these proven components into a pipeline where the LLM handles text and the navigator handles structure. No component is asked to do something it hasn't been validated on. The risk is in the integration, not in the components.
