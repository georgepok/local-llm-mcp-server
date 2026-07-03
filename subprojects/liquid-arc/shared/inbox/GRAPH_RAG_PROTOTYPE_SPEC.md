# GRAPH RAG PROTOTYPE — Spec

## Motivation: Phase 2 Validated the Core Capabilities

| Capability | Evidence | RAG Application |
|---|---|---|
| Recall from accumulated state | 1.7-2.3× lift over plain LLM | Retrieve context from months-old documents |
| Topology-aware SPOF detection | 3.3× lift, beats oracle | "What are the critical dependencies?" |
| Cross-domain pattern matching | Cosine 0.994 | Structural precedent finding |
| Scope-filtered retrieval | 0.83 matches oracle | Role/department-aware queries |
| Noise resistance | Advantage grows with realistic prose | Enterprise document handling |
| Zero regressions | Confirmed across all conditions | Risk-free RAG enhancement |

## Architecture

```
Documents → LLM extraction → Knowledge Graph → LiquidARC → Metric Landscape
                                                                ↑
Query → Vector RAG (existing) → top-k chunks ──────────────────┤
      → Graph RAG (new)     → structurally relevant chunks ────┤
                                                                ↓
                                        Combined context → LLM → Answer
```

LiquidARC is an ADDITIVE layer alongside existing vector RAG. It never replaces vector retrieval — it supplements with structural retrieval that vector similarity misses.

## Components

### 1. Document Ingestion Pipeline

```python
class GraphRAGIngester:
    """Process documents into the knowledge graph incrementally."""
    
    def ingest_document(self, doc_text, doc_metadata=None):
        """
        1. Chunk document (standard RAG chunking — 500 token chunks with overlap)
        2. For each chunk: LLM extracts entities + relations
        3. Entity resolution: merge duplicates (fuzzy match on node IDs)
        4. Merge into h_state (navigator.state.merge_fragment)
        5. Link chunks to entities (text_segments store)
        6. Store chunks in vector DB as usual (existing RAG pipeline)
        """
        chunks = self.chunker.chunk(doc_text)
        
        for chunk in chunks:
            # Standard RAG: embed and store in vector DB
            self.vector_db.add(chunk.text, chunk.embedding, doc_metadata)
            
            # Graph RAG: extract structure and merge
            fragment = self.extractor.extract(chunk.text)
            fragment = self.entity_resolver.resolve(fragment, self.navigator.state)
            self.navigator.state.merge_fragment(fragment, source_text=chunk.text)
        
        return {"chunks_processed": len(chunks), 
                "nodes_added": len(self.navigator.state.nodes),
                "edges_added": len(self.navigator.state.edges)}
```

### 2. Entity Resolution

Critical for RAG: "Shanghai Port", "shanghai_port", "the Shanghai facility", "SH port" must all resolve to ONE node.

```python
class EntityResolver:
    """Merge duplicate entities across extractions."""
    
    def resolve(self, fragment, existing_state):
        """Match extracted nodes against existing graph.
        
        Matching strategy (ordered by priority):
        1. Exact ID match (shanghai_port == shanghai_port)
        2. Token-stem overlap >= 0.6 (shanghai_port ~ shanghai_facility)
        3. Type + metric proximity (same type, metrically close in h_state)
        
        Strategy 3 uses the navigator's OWN metric to determine
        if a new entity is "the same" as an existing one.
        """
        resolved_nodes = []
        for node in fragment["nodes"]:
            match = self._find_match(node, existing_state)
            if match:
                # Merge: increment mention count, update last_seen
                node["id"] = match  # redirect to existing ID
            resolved_nodes.append(node)
        
        fragment["nodes"] = resolved_nodes
        # Update edge src/dst to resolved IDs
        fragment["edges"] = self._resolve_edges(fragment["edges"], ...)
        return fragment
```

### 3. Multi-Modal Retrieval

At query time, run BOTH vector and graph retrieval:

```python
class GraphRAGRetriever:
    """Retrieve context from both vector DB and knowledge graph."""
    
    def retrieve(self, query_text, k_vector=10, k_graph=10):
        """
        Returns combined context: vector-similar chunks + structurally relevant chunks.
        
        Graph retrieval modes (from Phase 2 fixes):
        - "graph": causal chain traversal (for "what caused X?")
        - "metric": structural similarity (for "what's similar to X?")
        - "topology": centrality-based (for "what's most critical?")
        - "scope": scope-filtered traversal (for role-specific queries)
        """
        # Vector retrieval (existing RAG — unchanged)
        vector_chunks = self.vector_db.query(query_text, k=k_vector)
        
        # Graph retrieval (new — LiquidARC)
        nav_result = self.navigator.process_interaction(query_text)
        relevant_nodes = nav_result["context_nodes"]
        graph_chunks = self.navigator.state.retrieve_text_for_nodes(
            [n["id"] for n in relevant_nodes], max_segments=k_graph
        )
        
        # Combine and deduplicate
        all_chunks = self._merge_and_dedup(vector_chunks, graph_chunks)
        
        # Add structural hint as metadata
        structural_hint = nav_result.get("structural_hint")
        
        return {
            "chunks": all_chunks,
            "structural_hint": structural_hint,
            "retrieval_stats": {
                "vector_only": len(vector_chunks),
                "graph_only": len(graph_chunks),
                "overlap": len(set(vector_chunks) & set(graph_chunks)),
                "graph_retrieval_mode": nav_result.get("retrieval_mode")
            }
        }
```

### 4. Query Router

Detect which retrieval mode(s) to use based on query structure:

```python
class QueryRouter:
    """Route queries to appropriate retrieval modes.
    
    Not all queries need graph retrieval. Simple factual lookups
    are best served by vector similarity alone.
    """
    def route(self, query_text, extracted_fragment=None):
        """
        Returns: list of retrieval modes to use.
        
        Heuristics (from Phase 2 findings):
        - Causal language ("what caused", "why did", "root cause")    → graph + vector
        - Topology language ("most critical", "biggest risk", "SPOF") → topology + vector
        - Scope language ("for junior analysts", "in department X")   → scope + vector
        - Pattern language ("similar to", "like the last time")       → metric + vector
        - Simple factual ("what is X", "when did Y")                  → vector only
        """
        modes = ["vector"]  # always include vector
        
        query_lower = query_text.lower()
        
        if any(t in query_lower for t in ["cause", "why", "root", "led to", "resulted in"]):
            modes.append("graph")
        if any(t in query_lower for t in ["critical", "important", "risk", "single point", "hub", "bottleneck"]):
            modes.append("topology")
        if any(t in query_lower for t in ["for role", "as a", "department", "scope", "authorized"]):
            modes.append("scope")
        if any(t in query_lower for t in ["similar", "like before", "pattern", "reminds me", "seen this"]):
            modes.append("metric")
        
        # If fragment has disconnected components → add connection check
        if extracted_fragment and has_disconnected_components(extracted_fragment):
            modes.append("connection")
        
        return modes
```

### 5. Scale Strategy: Hierarchical Processing

For large knowledge graphs (1000+ nodes), use community detection to select the relevant subgraph before applying LiquidARC:

```python
class HierarchicalGraphRAG:
    """Two-level graph processing for scale.
    
    Level 1: NetworkX community detection on full graph (fast, O(E))
    Level 2: LiquidARC ODE on relevant subgraph (accurate, ≤200 nodes)
    """
    def process_query(self, query_nodes, full_graph):
        # Level 1: find which communities the query nodes belong to
        communities = nx.community.louvain_communities(full_graph)
        relevant_communities = [c for c in communities 
                               if any(qn in c for qn in query_nodes)]
        
        # Expand: include adjacent communities (1-hop in community graph)
        expanded = self._expand_communities(relevant_communities, communities, full_graph)
        
        # Cap at 200 nodes
        subgraph_nodes = set()
        for comm in expanded:
            subgraph_nodes.update(comm)
            if len(subgraph_nodes) > 200:
                break
        
        # Level 2: LiquidARC processes the subgraph
        subgraph = full_graph.subgraph(subgraph_nodes)
        result = self.navigator.engine.analyze_graph(
            self._to_json(subgraph), self._to_query(query_nodes)
        )
        
        return result
```

## Evaluation Plan

### Benchmark 1: Multi-hop QA (HotPotQA or equivalent)

Test whether graph retrieval improves multi-hop question answering vs vector-only RAG.

```
Dataset: 500 multi-hop questions requiring 2-5 hops
Conditions:
  A: Vector RAG only (standard baseline)
  B: Vector + Graph RAG (LiquidARC)
  C: Vector + Microsoft GraphRAG (community summaries)
  
Metric: Answer accuracy (exact match + F1)
Target: B > A on ≥20% of questions, B competitive with C
```

### Benchmark 2: Long-Session Enterprise Simulation

Scale up Phase 2 to 200 interactions across 5 domains, then 20 cross-domain queries.

```
Dataset: 200 enterprise interactions + 20 queries (generated procedurally)
Conditions: same A/B/C/D/E as Phase 2
Metric: structural score (node ID recall) + LLM judge
Target: B > A by ≥2× structural lift on recall and topology queries
```

### Benchmark 3: Scope-Sensitive Retrieval

Enterprise compliance scenario: role-based access to different policy sections.

```
Dataset: 50 policy documents with scope annotations + 30 role-specific queries
Conditions:
  A: Vector RAG (no scope awareness)
  B: Vector + Graph RAG with scope filtering
  
Metric: precision@5 (are retrieved chunks actually in-scope?)
Target: B precision ≥ 0.8, A precision < 0.5 (vector retrieves out-of-scope chunks)
```

### Benchmark 4: Pattern-Based Precedent Finding

Legal/compliance scenario: find structurally similar prior cases.

```
Dataset: 100 case summaries extracted to graphs + 10 novel cases
Conditions:
  A: Vector RAG (keyword/semantic similarity)
  B: Graph RAG with metric signatures (structural similarity)
  
Metric: Does the retrieved precedent have the same structural pattern?
        (measured by graph isomorphism or cosine on metric signatures)
Target: B finds structurally correct precedent on ≥7/10, A on ≤4/10
```

## Implementation Phases

### Phase 1: Ingestion pipeline (1 week)
- Document chunker + LLM extractor + entity resolver
- Integration with a standard vector DB (ChromaDB or FAISS)
- Test on 100 documents, verify knowledge graph quality

### Phase 2: Retrieval pipeline (1 week)
- QueryRouter + GraphRAGRetriever
- Multi-modal retrieval (vector + graph + topology + scope)
- Test on Phase 2 session data (verify we match Phase 2 results)

### Phase 3: Benchmark evaluation (1-2 weeks)
- Run Benchmarks 1-4
- Compare against vector-only and Microsoft GraphRAG
- Document results

### Phase 4: Scale testing (1 week)
- Implement HierarchicalGraphRAG
- Test on 1000+ node graphs
- Profile latency: target <200ms per query

## What Exists vs What's New

```
EXISTS:
  ✓ GraphEngine with all analysis tools
  ✓ GeometricState with merge/query/persist
  ✓ PatternLibrary with signature matching
  ✓ Navigator orchestrator
  ✓ LLM extraction prompt
  ✓ MCP server with all tools
  ✓ Structural scorer
  
NEW:
  - Document chunker (standard, many libraries available)
  - Entity resolver (fuzzy matching + metric proximity)
  - Vector DB integration (ChromaDB adapter)
  - QueryRouter (heuristic, simple)
  - GraphRAGRetriever (combines vector + graph results)
  - HierarchicalGraphRAG (community detection + subgraph selection)
  - Benchmark datasets (HotPotQA adapter + procedural generation)
```

The core geometric reasoning is proven and deployed. The RAG-specific pieces are engineering — connecting proven components to a standard retrieval pipeline.
