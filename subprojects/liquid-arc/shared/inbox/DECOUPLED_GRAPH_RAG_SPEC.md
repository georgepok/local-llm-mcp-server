# DECOUPLED GRAPH RAG — Graph DB + ODE Subgraph Engine

## Design Principle

Separate storage/traversal from geometric computation. Each component operates at its natural scale.

```
Graph DB (unlimited):     stores nodes, edges, types, scopes, text links
                          handles: BFS, scope filter, edge traversal, community detection
                          capacity: millions of nodes
                          latency: <10ms for traversal queries

ODE Engine (≤200 nodes):  processes subgraphs extracted from graph DB
                          handles: metric centrality, structural analysis, signature computation
                          capacity: 200 nodes per invocation
                          latency: 50-600ms depending on subgraph size

Pattern Library (unlimited): stores 52-d metric signatures
                             handles: cross-domain structural matching
                             capacity: millions of signatures
                             latency: <10ms for cosine search
```

The current architecture processes the ENTIRE h_state through the ODE on every merge. The decoupled architecture runs the ODE ONLY when geometric computation is specifically needed, on a subgraph extracted from the graph DB.

## What Changes

```
CURRENT (monolithic h_state):
  New chunk → extract → merge into h_state → ODE over ALL nodes → update ALL positions
  Query → ODE over ALL nodes → metric retrieval → answer
  Bottleneck: ODE runs on full graph at both ingestion AND query time

DECOUPLED:
  New chunk → extract → insert into graph DB → done (no ODE at ingestion)
  Query → graph DB handles traversal/scope → IF geometric needed → extract subgraph → ODE → answer
  ODE only runs on demand, on ≤200 node subgraphs
```

Ingestion becomes O(1) per chunk (graph DB insert). Geometric computation is deferred to query time and scoped to the relevant subgraph.

## Architecture

### Component 1: Graph DB

Use NetworkX as the in-process graph store. It's already a dependency, handles all needed operations, and avoids external service complexity. Swap to Neo4j later if scale demands it.

```python
class KnowledgeGraphDB:
    """Persistent typed knowledge graph with scope-aware edges.
    
    Backed by NetworkX DiGraph. Serializes to JSON for persistence.
    All graph-algorithmic operations happen here — no ODE.
    """
    def __init__(self, db_path):
        self.G = nx.DiGraph()
        self.text_segments = {}  # node_id → [{"text": str, "timestamp": float, "chunk_id": str}]
        self.db_path = db_path
        self._load()
    
    # === WRITE OPERATIONS ===
    
    def add_fragment(self, fragment, source_text=None, chunk_id=None):
        """Add extracted graph fragment. O(V+E) — no ODE."""
        for node in fragment["nodes"]:
            nid = node["id"]
            if self.G.has_node(nid):
                self.G.nodes[nid]["mention_count"] += 1
                self.G.nodes[nid]["last_seen"] = time.time()
            else:
                self.G.add_node(nid,
                    type=node.get("type", "entity"),
                    role=node.get("role", "intermediate"),
                    first_seen=time.time(),
                    last_seen=time.time(),
                    mention_count=1)
            
            if source_text:
                self.text_segments.setdefault(nid, []).append({
                    "text": source_text,
                    "timestamp": time.time(),
                    "chunk_id": chunk_id
                })
        
        for edge in fragment["edges"]:
            self.G.add_edge(edge["src"], edge["dst"],
                type=edge.get("type", "related_to"),
                scope=edge.get("scope", None),
                weight=self.G[edge["src"]][edge["dst"]].get("weight", 0) + 1
                    if self.G.has_edge(edge["src"], edge["dst"]) else 1)
        
        self._save()
    
    # === GRAPH-ALGORITHMIC QUERIES (no ODE) ===
    
    def trace_causal_chain(self, target, max_hops=10):
        """BFS backward from target through causal edges. Deterministic."""
        causal_types = {"causes", "precedes", "enables"}
        path = [target]
        current = target
        for _ in range(max_hops):
            predecessors = [
                src for src, _, data in self.G.in_edges(current, data=True)
                if data.get("type") in causal_types
            ]
            if not predecessors:
                break
            current = predecessors[0]  # follow primary cause
            path.append(current)
        path.reverse()
        return {"root": path[0], "path": path, "hops": len(path) - 1}
    
    def scope_filter(self, query_scope):
        """Return subgraph with only edges valid for given scope. Deterministic."""
        filtered = self.G.copy()
        edges_to_remove = [
            (u, v) for u, v, data in filtered.edges(data=True)
            if data.get("scope") is not None and data["scope"] != query_scope
        ]
        filtered.remove_edges_from(edges_to_remove)
        return filtered
    
    def get_reachable(self, source, scope=None, max_hops=5):
        """All nodes reachable from source within max_hops, optionally scope-filtered."""
        G = self.scope_filter(scope) if scope else self.G
        reachable = set()
        frontier = {source}
        for _ in range(max_hops):
            next_frontier = set()
            for node in frontier:
                for successor in G.successors(node):
                    if successor not in reachable:
                        next_frontier.add(successor)
                        reachable.add(successor)
            frontier = next_frontier
            if not frontier:
                break
        return reachable
    
    def find_communities(self, min_size=3):
        """Louvain community detection. For subgraph selection."""
        undirected = self.G.to_undirected()
        communities = nx.community.louvain_communities(undirected)
        return [c for c in communities if len(c) >= min_size]
    
    def get_neighbors(self, node_ids, hops=2):
        """Get k-hop neighborhood of given nodes."""
        neighborhood = set(node_ids)
        frontier = set(node_ids)
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.G.predecessors(node))
                next_frontier.update(self.G.successors(node))
            frontier = next_frontier - neighborhood
            neighborhood.update(frontier)
        return neighborhood
    
    def extract_subgraph(self, node_ids, max_nodes=200):
        """Extract a subgraph for ODE processing. Cap at max_nodes."""
        node_set = set(node_ids)
        if len(node_set) > max_nodes:
            # Priority: query nodes first, then by mention_count
            scored = [(n, self.G.nodes[n].get("mention_count", 0)) for n in node_set]
            scored.sort(key=lambda x: x[1], reverse=True)
            node_set = set(n for n, _ in scored[:max_nodes])
        
        subgraph_nodes = [
            {"id": n, "type": self.G.nodes[n]["type"], "role": self.G.nodes[n]["role"]}
            for n in node_set if n in self.G.nodes
        ]
        subgraph_edges = [
            {"src": u, "dst": v, "type": data.get("type", "related_to"),
             "scope": data.get("scope")}
            for u, v, data in self.G.edges(data=True)
            if u in node_set and v in node_set
        ]
        return {"nodes": subgraph_nodes, "edges": subgraph_edges}
    
    def retrieve_text(self, node_ids, max_segments=10):
        """Get text chunks linked to given nodes."""
        segments = []
        for nid in node_ids:
            segments.extend(self.text_segments.get(nid, []))
        segments.sort(key=lambda s: s["timestamp"], reverse=True)
        seen = set()
        unique = []
        for seg in segments:
            if seg["text"] not in seen:
                seen.add(seg["text"])
                unique.append(seg)
        return unique[:max_segments]
    
    # === STATS ===
    
    def stats(self):
        return {
            "n_nodes": self.G.number_of_nodes(),
            "n_edges": self.G.number_of_edges(),
            "n_communities": len(self.find_communities()),
            "n_text_segments": sum(len(v) for v in self.text_segments.values()),
            "node_types": dict(Counter(
                data.get("type", "unknown") for _, data in self.G.nodes(data=True)
            ))
        }


### Component 2: ODE Subgraph Engine

Existing GraphEngine, unchanged. Called ONLY when geometric computation is needed.

```python
class SubgraphODEEngine:
    """Runs ODE on extracted subgraphs. ≤200 nodes per invocation.
    
    Wraps the existing GraphEngine. The only change: input comes from
    KnowledgeGraphDB.extract_subgraph() instead of direct user input.
    """
    def __init__(self, checkpoint_path, device="cpu"):
        self.engine = GraphEngine(checkpoint_path, device=device)
    
    def compute_centrality(self, subgraph_json):
        """Run ODE, return per-node metric centrality."""
        diagnostics = self.engine.get_graph_diagnostics(json.dumps(subgraph_json))
        return json.loads(diagnostics)
    
    def compute_signature(self, subgraph_json):
        """Run ODE, return 52-d metric signature for pattern matching."""
        diagnostics = self.engine.get_graph_diagnostics(json.dumps(subgraph_json))
        d = json.loads(diagnostics)
        # Extract the signature components
        return {
            "cv": d["cv_g"],
            "criticality": d["criticality_ratio"],
            "tau_mean": d["tau_mean"],
            "clusters": d["metric_clusters"],
            "centrality": d["per_node_centrality_metric_space"],
            "full_signature": self._extract_52d(d)
        }
    
    def analyze(self, subgraph_json, query_json):
        """Run full geometric analysis (root cause, connection, etc.)."""
        return self.engine.analyze_graph(json.dumps(subgraph_json), json.dumps(query_json))
```

### Component 3: Pattern Library (unchanged)

Existing PatternLibrary. Stores 52-d signatures. Cosine matching. No capacity limit.

### Component 4: Decoupled Orchestrator

The new orchestrator routes queries to the appropriate component.

```python
class DecoupledGraphRAG:
    """Routes queries to graph DB or ODE engine based on what's needed.
    
    Design: graph DB handles everything it can (fast, unlimited scale).
    ODE engine handles ONLY what graph DB can't (geometric computation).
    """
    def __init__(self, graph_db, ode_engine, pattern_library, vector_db, extractor):
        self.db = graph_db
        self.ode = ode_engine
        self.patterns = pattern_library
        self.vector_db = vector_db
        self.extractor = extractor
    
    # === INGESTION (no ODE) ===
    
    def ingest(self, doc_text, metadata=None):
        """Ingest a document. Graph DB only — no ODE at ingestion time."""
        chunks = chunk_text(doc_text)
        for chunk in chunks:
            # Vector DB (standard RAG)
            self.vector_db.add(chunk.text, chunk.embedding, metadata)
            
            # Graph DB (extract + resolve + insert)
            fragment = self.extractor.extract(chunk.text)
            if fragment and fragment.get("nodes"):
                fragment = self.entity_resolver.resolve(fragment, self.db)
                self.db.add_fragment(fragment, source_text=chunk.text, chunk_id=chunk.id)
        
        # Optionally: compute signature for this document's subgraph
        # and store in pattern library (runs ODE once per document, not per chunk)
        doc_nodes = [n["id"] for n in fragment["nodes"]] if fragment else []
        if len(doc_nodes) >= 3:
            neighborhood = self.db.get_neighbors(doc_nodes, hops=2)
            subgraph = self.db.extract_subgraph(neighborhood, max_nodes=50)
            sig = self.ode.compute_signature(subgraph)
            self.patterns.store(sig["full_signature"], {
                "label": f"doc_{metadata.get('title', 'unknown')}",
                "source": metadata
            })
    
    # === QUERY ===
    
    def query(self, query_text, scope=None):
        """Route query to appropriate components."""
        
        # Step 1: Vector retrieval (always)
        vector_chunks = self.vector_db.query(query_text, k=10)
        
        # Step 2: Extract entities from query
        fragment = self.extractor.extract(query_text)
        query_nodes = [n["id"] for n in fragment.get("nodes", [])] if fragment else []
        
        # Step 3: Route based on query type
        route = self._route(query_text, query_nodes, scope)
        
        graph_result = {}
        graph_chunks = []
        structural_hint = None
        
        if "causal" in route:
            # Graph DB handles causal tracing (BFS — no ODE needed)
            for qn in query_nodes:
                if qn in self.db.G:
                    chain = self.db.trace_causal_chain(qn)
                    graph_result["causal_chain"] = chain
                    graph_chunks.extend(
                        self.db.retrieve_text(chain["path"], max_segments=5)
                    )
                    structural_hint = f"Root cause: {chain['root']}, {chain['hops']} hops"
                    break
        
        if "scope" in route and scope:
            # Graph DB handles scope filtering (deterministic — no ODE needed)
            filtered = self.db.scope_filter(scope)
            reachable = set()
            for qn in query_nodes:
                if qn in filtered:
                    reachable.update(nx.descendants(filtered, qn))
            graph_result["scope_filtered_nodes"] = list(reachable)
            graph_chunks.extend(
                self.db.retrieve_text(list(reachable), max_segments=5)
            )
        
        if "topology" in route:
            # ODE needed — extract subgraph, compute metric centrality
            if query_nodes:
                neighborhood = self.db.get_neighbors(query_nodes, hops=3)
            else:
                # Global topology query — use largest community
                communities = self.db.find_communities()
                neighborhood = max(communities, key=len) if communities else set()
            
            subgraph = self.db.extract_subgraph(neighborhood, max_nodes=200)
            ode_result = self.ode.compute_centrality(json.dumps(subgraph))
            graph_result["topology"] = ode_result
            
            # Get top SPOFs
            centrality = ode_result.get("per_node_centrality_metric_space", {})
            top_spofs = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            spof_nodes = [n for n, _ in top_spofs]
            graph_chunks.extend(
                self.db.retrieve_text(spof_nodes, max_segments=5)
            )
            structural_hint = f"Top SPOFs: {', '.join(spof_nodes)}"
        
        if "pattern" in route:
            # Pattern library — compute signature of query fragment, match
            if fragment and len(fragment.get("nodes", [])) >= 3:
                query_neighborhood = self.db.get_neighbors(query_nodes, hops=2)
                subgraph = self.db.extract_subgraph(query_neighborhood, max_nodes=50)
                sig = self.ode.compute_signature(json.dumps(subgraph))
                match = self.patterns.find_nearest(sig["full_signature"], threshold=0.85)
                if match:
                    graph_result["pattern_match"] = match
                    structural_hint = (structural_hint or "") + \
                        f"\nMatches pattern: {match['label']} (cosine {match['similarity']:.3f})"
        
        # Step 4: Merge and return
        all_chunks = self._merge_dedup(vector_chunks, graph_chunks)
        
        return {
            "chunks": all_chunks,
            "structural_hint": structural_hint,
            "graph_result": graph_result,
            "route": route,
            "stats": {
                "vector_chunks": len(vector_chunks),
                "graph_chunks": len(graph_chunks),
                "ode_invoked": "topology" in route or "pattern" in route,
                "db_nodes": self.db.G.number_of_nodes()
            }
        }
    
    def _route(self, query_text, query_nodes, scope):
        """Determine which retrieval modes to activate."""
        modes = []
        q = query_text.lower()
        
        if any(t in q for t in ["cause", "why", "root", "led to", "resulted"]):
            modes.append("causal")
        if scope or any(t in q for t in ["as a", "for role", "my department", "authorized"]):
            modes.append("scope")
        if any(t in q for t in ["critical", "risk", "single point", "hub", "bottleneck", "important"]):
            modes.append("topology")
        if any(t in q for t in ["similar", "like before", "pattern", "seen this", "reminds"]):
            modes.append("pattern")
        
        # Default: if no structural signal detected, causal + scope (cheap, graph-only)
        if not modes and query_nodes:
            modes = ["causal"]
        
        return modes
```

## Experiment Design

### Experiment 1: Scale stress test

Ingest 1000 documents (synthetic enterprise corpus) into the decoupled system. Verify:
- Graph DB handles 1000 documents without ODE bottleneck
- Node count grows beyond previous 200 ceiling
- Query latency stays under 1 second for all query types

```python
def test_scale():
    system = DecoupledGraphRAG(...)
    
    # Ingest 1000 documents
    for i, doc in enumerate(generate_enterprise_docs(1000)):
        t0 = time.time()
        system.ingest(doc.text, {"title": doc.title, "domain": doc.domain})
        ingest_time = time.time() - t0
        assert ingest_time < 2.0, f"Ingestion too slow at doc {i}: {ingest_time}s"
    
    stats = system.db.stats()
    print(f"Graph: {stats['n_nodes']} nodes, {stats['n_edges']} edges")
    assert stats['n_nodes'] > 500, "Entity resolution too aggressive"
    
    # Run queries at scale
    for query in generate_test_queries(50):
        t0 = time.time()
        result = system.query(query.text, scope=query.scope)
        query_time = time.time() - t0
        assert query_time < 2.0, f"Query too slow: {query_time}s"
        print(f"Query: {query.type}, time: {query_time:.3f}s, ODE invoked: {result['stats']['ode_invoked']}")
```

**Pass criteria:**
- 1000 documents ingested at <2s per document (no ODE at ingestion)
- Graph grows to 500+ nodes
- All query types complete in <2 seconds
- ODE invoked ONLY for topology and pattern queries (not causal, not scope)

### Experiment 2: Causal chain at scale

Verify that BFS causal tracing works correctly on a large graph without ODE.

```
Generate: 50 causal chains of length 3-8, interleaved across 500 documents
Test: 50 root-cause queries, compare answers to known ground truth
Conditions:
  A: Decoupled (graph DB BFS, no ODE)
  B: Monolithic (full ODE over all nodes, current architecture)
  C: ODE on subgraph (extract neighborhood, run ODE, use root_cause head)
```

**Pass criteria:**
- A matches B on ≥95% of queries (BFS gives same answer as ODE for causal tracing)
- A is ≥10× faster than B (BFS vs full ODE)
- C adds no accuracy over A for pure causal queries

This validates that causal tracing doesn't need the ODE — the graph DB handles it.

### Experiment 3: Scope filtering at scale

Verify that deterministic scope filtering works on a large graph.

```
Generate: 5 scopes × 100 scoped edges across 500 documents
Test: 30 scoped queries, measure precision@5
Conditions:
  A: Decoupled (graph DB scope filter)
  B: Monolithic (h_state with scope-aware mask)
```

**Pass criteria:**
- A precision@5 = 1.000 (deterministic filtering, same as benchmark 3)
- A latency < 50ms (graph DB filter, no ODE)
- B equivalent precision (scope filtering doesn't benefit from ODE)

### Experiment 4: Topology with subgraph extraction

The one query type that NEEDS the ODE. Verify that subgraph extraction → ODE centrality gives meaningful results on a large graph.

```
Generate: 1000-node graph from 500 documents
Plant: 5 known structural hubs (high downstream reach)
Test: 10 topology queries asking for SPOFs
Conditions:
  A: Decoupled (community → subgraph ≤200 nodes → ODE centrality)
  B: NetworkX centrality on full graph (betweenness, no ODE)
  C: NetworkX degree centrality (simplest baseline)
```

**Pass criteria:**
- A identifies ≥3/5 planted hubs in top-10 results
- A outperforms C (metric centrality better than degree count)
- A latency < 2 seconds (community detection + ODE)
- A handles 1000-node graph that monolithic ODE cannot (>60 second timeout)

### Experiment 5: Pattern matching at scale

Verify pattern library works with 100+ stored signatures.

```
Phase 1: Ingest 500 documents across 5 domains. Store signatures per-document subgraph.
Phase 2: Present 10 novel scenarios with known structural matches to stored patterns.
Test: does cosine matching find the correct pattern?
```

**Pass criteria:**
- Pattern library contains 50+ signatures after 500 documents
- Correct pattern matched at cosine >0.85 for ≥7/10 novel scenarios
- Pattern query latency <100ms (cosine search, no ODE needed at query time)

### Experiment 6: End-to-end comparison

The definitive test: decoupled architecture vs monolithic vs agentic baseline.

```
Full pipeline: 200 documents ingested → 30 mixed queries
Conditions:
  A: Plain LLM + vector RAG (no graph)
  B: Monolithic h_state (current architecture, caps at ~200 nodes)
  C: Decoupled (graph DB + on-demand ODE)
  D: LLM agent + NetworkX tools (agentic baseline)

Score: LLM-as-judge + structural scorer (dual evaluation from Phase 2)
```

**Pass criteria:**
- C ≥ B on all query types (decoupled is at least as good as monolithic)
- C handles document volumes that B cannot (>200 node equivalent)
- C scope precision = 1.000 (deterministic, same as benchmark 3)
- C outperforms A on causal + topology + pattern queries (graph layer adds value)
- C competitive with D on causal (BFS ≈ agent BFS) but faster on topology and pattern

## MCP Server Extension

Add graph DB tools alongside existing geometric tools:

```python
# New tools (graph DB — fast, unlimited scale)
@tool
def graph_query(query_text: str, scope: str = "") -> str:
    """Full pipeline: route query → graph DB + optional ODE → combined results."""
    return json.dumps(system.query(query_text, scope=scope or None))

@tool
def graph_stats() -> str:
    """Current knowledge graph statistics."""
    return json.dumps(system.db.stats())

@tool
def graph_ingest(text: str, title: str = "") -> str:
    """Ingest a document into the knowledge graph."""
    system.ingest(text, {"title": title})
    return json.dumps({"ingested": True, "stats": system.db.stats()})

@tool
def graph_causal_chain(target_node: str) -> str:
    """Trace causal chain to root cause. Graph DB only — no ODE."""
    return json.dumps(system.db.trace_causal_chain(target_node))

@tool  
def graph_scope_query(query_text: str, scope: str) -> str:
    """Scope-filtered retrieval. Deterministic — precision 1.000."""
    # ... scope filter + text retrieval

# Existing tools (ODE — invoked only when geometric computation needed)
# analyze_graph, compare_graphs, get_graph_diagnostics, correct_answer
# These now receive SUBGRAPHS from graph_query, not full h_state
```

## Implementation Plan

### Week 1: Graph DB + ingestion
1. Implement KnowledgeGraphDB (NetworkX-backed, all graph-algorithmic methods)
2. Implement DecoupledGraphRAG.ingest() (no ODE at ingestion)
3. Run Experiment 1 (scale stress test — 1000 documents)
4. Run Experiment 2 (causal chain at scale — BFS vs ODE)

### Week 2: Query routing + ODE on demand
1. Implement DecoupledGraphRAG.query() with full routing
2. Implement subgraph extraction for ODE (neighborhood + cap at 200)
3. Run Experiment 3 (scope at scale)
4. Run Experiment 4 (topology with subgraph extraction)

### Week 3: Pattern library + full evaluation
1. Wire pattern library to per-document signature computation
2. Run Experiment 5 (pattern matching at scale)
3. Run Experiment 6 (end-to-end comparison, all conditions)
4. MCP server extension with new tools

## What This Proves If It Works

The decoupled architecture separates three scaling regimes:

```
Graph storage/traversal:  O(V+E) — scales to millions of nodes
Pattern matching:          O(K×52) — scales to millions of signatures  
Geometric computation:     O(N²×16) — bounded at N≤200 per invocation, invoked on demand
```

The ODE is no longer a bottleneck because it's no longer on the critical path for most operations. It's a specialized accelerator invoked only for topology ranking and signature computation — the two operations where it provides unique value. Everything else runs on the graph DB at graph-algorithm speed.

If Experiment 6 shows C ≥ B (decoupled at least as good as monolithic), the monolithic h_state can be retired. The graph DB becomes the primary state store, and the ODE becomes a callable service for geometric intelligence on demand.
