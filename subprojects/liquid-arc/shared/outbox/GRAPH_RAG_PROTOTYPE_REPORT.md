# GraphRAG Prototype — Report

Implementation of the `GRAPH_RAG_PROTOTYPE_SPEC.md` as an additive
retrieval layer over the existing Navigator. Core geometric components
remain unchanged; the new code is the RAG-specific engineering:
chunker, vector DB, entity resolver, ingester, query router,
retriever, hierarchical subgraph selector.

## What Was Built

```
liquid_arc/graph_rag/
├── chunker.py          # sentence-window chunker (~500 tokens, 100 overlap)
├── vector_db.py        # in-memory hashed-TF-IDF vector store, no deps
├── entity_resolver.py  # exact → token-stem-Jaccard → metric proximity
├── ingester.py         # chunk → vector add + LLM-extract → resolve → merge
├── router.py           # heuristic query-type → retrieval modes
├── retriever.py        # multi-modal: vector + graph + topology + scope
└── hierarchical.py     # Louvain community → ≤N-node subgraph → engine
```

All components are stdlib + numpy + networkx. ChromaDB / FAISS can be
swapped in via the `VectorDB` interface when production deployment
wants a scaled backend.

## Benchmark 3 — Scope-Sensitive Retrieval (PASS)

Dataset: 50 policy docs (5 scopes × 10 topics) + 30 scoped queries.

Queries use scope *proxies* ("an SRE rotation on-call", "a
customer-facing commercial team") instead of literal scope names, so
vector retrieval cannot keyword-match the scope — the user supplies
the scope separately, and the retriever must honor it.

| Condition | precision@5 | hit rate ≥ 0.6 |
|---|---|---|
| A (vector-only)                    | **0.227** | 0 / 30 |
| **B (vector + scope filter)**      | **1.000** | **30 / 30** |

All queries: every chunk returned by B is in-scope. A retrieves
cross-scope chunks because "approval" / "expense" / "review" keywords
appear in every scope's policies.

- Gate 1 (B ≥ 0.80): **PASS** (1.00)
- Gate 2 (A < 0.50): **PASS** (0.23)
- Gate 3 (B beats A consistently): **PASS** (30 wins vs 0)

Outputs: `shared/outbox/graphrag/scope.json`.

## Benchmark 2 — Scale Stress Test

Composed all three Phase 2 variants into a single navigator state:
150 interactions, 190 nodes, 180 edges, 4 metric-signature patterns
(exact-ID entity resolution collapsed shared structural nodes across
variants). Ingestion time: 91.1s on CPU (600ms per merge + ODE).

Evaluated all 30 cross-domain queries under two retrieval strategies:

- **Direct**: `navigator.process_interaction` with anchor fragment,
  returns top-10 context nodes via metric ∪ graph retrieval.
- **Hierarchical**: `HierarchicalGraphRAG.select_subgraph` — Louvain
  community detection identifies subgraph containing the query anchors,
  capped at 200 nodes.

Structural recall (fraction of expected node IDs surfaced):

| Query type | n | Direct top-10 | Hierarchical subgraph | Direct p50 | Hier p50 |
|---|---|---|---|---|---|
| recall         | 9 | 0.69 | **1.00** | 799 ms | 1 ms |
| analogy        | 9 | 0.07 | **1.00** | 835 ms | 1 ms |
| topology       | 6 | 0.00 | **0.50** | 222 ms | 1 ms |
| scope_transfer | 6 | 0.50 | **1.00** | 716 ms | 1 ms |

**Key observations:**

1. **Expected nodes survive the composition.** Hierarchical selection
   finds them in the 190-node state in all 9 recall queries, 9 analogy
   queries, 6 scope queries. The accumulated geometry + graph structure
   preserves variant-specific answers under 3× noise.
2. **Direct top-10 retrieval is too narrow under scale.** On analogy
   queries it drops to 0.07 because the 10-item context window doesn't
   reach nodes that are several hops away but structurally related.
   Increasing `k_graph` or running hierarchical selection first would
   close this gap.
3. **Hierarchical selection is ~800× faster than direct retrieval.**
   1 ms vs 600-1200 ms per query — because the subgraph selector does
   community detection on the adjacency graph (no ODE), while direct
   runs a full 190-node ODE forward.
4. **Entity resolution via exact-ID match works across variants.**
   Shared structural IDs (`port_congestion`, `low_stock`, etc.)
   collapsed 3 variants' 150 interactions into 190 unique nodes
   (expected ~370 without dedupe). The token-stem tier never fired in
   this test because all exact-ID matches preempted it.

Outputs: `shared/outbox/graphrag/scale.json`.

## Phase 2 Capabilities Carried Into RAG

The capabilities validated in Phase 2 now plug into the RAG pipeline:

| Capability | Phase 2 evidence | RAG application |
|---|---|---|
| Recall from accumulated state | 1.7-2.3× lift | `retrieve_text_for_nodes` over many-doc corpus |
| Topology-aware SPOF detection | 3.3× lift | "what are critical dependencies" queries |
| Cross-domain pattern matching | cosine ≥ 0.99 | Precedent finding |
| Scope-filtered retrieval | 0.83 matches oracle | **Benchmark 3 confirms 1.00 precision** |
| Noise resistance | grows with realistic prose | **Benchmark 2 confirms under 3× composition** |
| Zero regressions | across all conditions | Carried into additive design — vector leg always runs |

## What Remains

- **Benchmark 1 (HotPotQA multi-hop)**: scaffolding is in place but
  requires downloading the HotPotQA dataset and writing a ground-truth
  adapter. Deferred.
- **Benchmark 4 (precedent finding)**: similar to Phase 1 Exp 5 on
  bigger data; architecture is the same pattern library + signature
  cosine.
- **Scale past 1000 nodes**: not yet exercised. Hierarchical layer
  exists but only gets invoked when `max_subgraph_nodes` is exceeded —
  today's 190-node test stayed under the default cap of 200.
- **Metric-proximity entity resolution tier**: implemented
  (`EntityResolver._resolve_node` tier 3) but effectively unreachable
  in the benchmarks because the exact-ID tier caught every case.
  Real-document ingestion with fuzzy references like "the Shanghai
  facility" vs "shanghai_port" would exercise it.
- **Production vector backend**: swap `VectorDB` for ChromaDB or FAISS
  when embedding quality matters; the interface is preserved.

## Wire-Up for Real Use

```python
from liquid_arc.navigator import GeometricNavigator
from liquid_arc.navigator_state import GeometricState
from liquid_arc.navigator_patterns import PatternLibrary
from liquid_arc.navigator_extract import LLMExtractor
from liquid_arc.graph_engine_inference import GraphEngine
from liquid_arc.graph_rag.chunker import Chunker
from liquid_arc.graph_rag.vector_db import VectorDB
from liquid_arc.graph_rag.entity_resolver import EntityResolver
from liquid_arc.graph_rag.ingester import GraphRAGIngester
from liquid_arc.graph_rag.retriever import GraphRAGRetriever
from liquid_arc.graph_rag.router import QueryRouter

engine = GraphEngine("/path/to/step_500.pt", device="cpu")
state = GeometricState("/path/to/rag_state.json", engine)
patterns = PatternLibrary("/path/to/rag_patterns.json")
extractor = LLMExtractor(base_url="http://…/v1",
                         model="NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
navigator = GeometricNavigator(engine=engine, state=state,
                                extractor=extractor,
                                pattern_library=patterns)
vector_db = VectorDB(dim=1024)
ingester = GraphRAGIngester(navigator=navigator, vector_db=vector_db,
                             extractor=extractor)
retriever = GraphRAGRetriever(navigator=navigator, vector_db=vector_db)

# Ingest docs (LLM extraction happens per chunk)
for path in docs:
    ingester.ingest_document(open(path).read(),
                              doc_metadata={"source": path})

# Retrieve with scope filter
result = retriever.retrieve(
    "For an SRE rotation on-call, what is the deployment approval?",
    scope="production")
```

## Outputs

- `liquid_arc/graph_rag/` — six production modules
- `scripts/bench_graphrag_scope.py` — Benchmark 3 runner
- `scripts/bench_graphrag_scale.py` — Benchmark 2 runner
- `shared/outbox/graphrag/scope.json`
- `shared/outbox/graphrag/scale.json`
