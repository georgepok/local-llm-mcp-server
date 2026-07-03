"""GraphRAG prototype — additive retrieval layer over LiquidARC's navigator.

Modules:
  chunker         — sentence-window chunking (~500 tokens with overlap)
  vector_db       — minimal in-memory vector DB (hashed TF-IDF + cosine)
  entity_resolver — fuzzy + metric-proximity entity merging
  ingester        — ties chunker → extractor → resolver → merge + vector add
  router          — heuristic query router to decide retrieval modes
  retriever       — multi-modal retrieval (vector + graph + topology + scope)
  hierarchical    — community-detection subgraph selector for scale
"""
