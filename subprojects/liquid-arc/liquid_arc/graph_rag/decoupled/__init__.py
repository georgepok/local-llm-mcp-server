"""Decoupled Graph RAG — Graph DB + on-demand ODE subgraph engine.

Separates the scaling regimes:
  - Graph storage/traversal (O(V+E))  — scales to millions of nodes
  - Pattern matching (O(K×d_sig))     — cosine search, unlimited history
  - Geometric ODE (O(N²×steps))       — bounded at N ≤ 200 per invocation,
                                          only invoked on topology / signature queries

The monolithic h_state is replaced by a NetworkX-backed graph DB that
handles BFS, scope filtering, community detection, and text retrieval
without running the ODE. The ODE engine is called only for topology
ranking or pattern signature computation, on an extracted ≤200-node
subgraph.
"""
