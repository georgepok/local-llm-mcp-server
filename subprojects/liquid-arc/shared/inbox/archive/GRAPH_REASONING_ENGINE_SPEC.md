# LIQUIDARC GRAPH REASONING ENGINE — Experiment Spec

## Motivation

Every attempt to integrate LiquidARC with pretrained transformers failed: prefix injection, delta extraction, attention bias, layer-wise perturbation, criticality-trained routing — all produced structured geometry that either got ignored or actively degraded generation. The root cause: pretrained transformer attention is already near-optimal for tasks within the model's capacity. A 60M geometric perturbation can't improve on 4B parameters of learned routing.

But LiquidARC demonstrated genuine capabilities on ARC: 2-3× improvement from criticality, phase transitions in ~1600 steps, distribution-invariant geometry. These capabilities work because ARC inputs have DISCRETE CATEGORICAL STRUCTURE (colors, positions, roles) that creates the embedding clusters the MetricNet needs.

Graphs have the same categorical structure: node types, edge types, subgraph membership. The MetricNet doesn't need to be retrained — the ARC-trained routing patterns (cluster by type, route within connected groups, separate disconnected components) directly apply to graph reasoning.

The integration with LLMs changes from PERTURBATION (modifying attention) to COMMUNICATION (exchanging symbolic structures). LiquidARC processes graphs geometrically. The LLM reads the results as text. No hooks, no biases, no fighting with pretrained weights.

## Architecture

```
LLM (text world)                    LiquidARC (graph world)
─────────────────                   ───────────────────────
User asks question          
  ↓                                 
LLM extracts graph     ──────→     Graph enters as ODE positions:
from text context                    - Nodes → position embeddings
(structured output)                  - Node types → TypeEmbed (like ColorEmbed in ARC)
                                     - Edge types → relation features
                                     - Graph structure → adjacency / mask
                                           ↓
                                     ODE integration (16 Euler steps):
                                       MetricNet computes g per node
                                       Heat kernel routes along edges
                                       Criticality scaffolding active
                                       Phase transition possible
                                           ↓
                                     Geometric analysis:
                                       Shortest paths (geodesic distances)
                                       Connected components (metric clusters)
                                       Root cause tracing (directed diffusion)
                                       Structural isomorphism (metric signature match)
                                           ↓
LLM reads symbolic     ←──────     Output as structured result:
results as text context              - Causal paths with hop counts
                                     - Structural pattern classification
                                     - Analogy mappings between graphs
                                     - Scope/validity annotations
  ↓
LLM generates response
incorporating graph analysis
```

## Phase 1: Graph Encoding for LiquidARC

### Node embedding (directly analogous to ARC)

```python
class GraphNodeEmbedding(nn.Module):
    """Encode graph nodes for LiquidARC ODE processing.
    
    Directly mirrors ARC's embedding structure:
      ARC:   ColorEmbed(color) + PosX(x) + PosY(y) + RoleEmbed(role)
      Graph: TypeEmbed(type) + StructEmbed(features) + RoleEmbed(role)
    """
    def __init__(self, d_model, n_node_types=32, n_edge_types=16, n_roles=8):
        super().__init__()
        self.type_embed = nn.Embedding(n_node_types, d_model)      # like ColorEmbed
        self.role_embed = nn.Embedding(n_roles, d_model)            # like RoleEmbed
        self.struct_proj = nn.Linear(16, d_model)                   # structural features
        self.edge_type_embed = nn.Embedding(n_edge_types, d_model)  # relation encoding
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, node_types, roles, struct_features):
        """
        Args:
            node_types: [B, N] node type indices
            roles: [B, N] role indices (source, target, intermediate, query)
            struct_features: [B, N, 16] (degree, centrality, depth, etc.)
        Returns:
            h: [B, N, d_model] node embeddings with categorical cluster structure
        """
        h = (self.type_embed(node_types) 
             + self.role_embed(roles) 
             + self.struct_proj(struct_features))
        return self.norm(h)
```

### Structural features per node

```python
def compute_structural_features(graph):
    """Extract 16-dim structural feature vector per node.
    
    These are input-INDEPENDENT features from graph topology.
    """
    features = []
    for node in graph.nodes:
        features.append([
            graph.in_degree(node),           # how many incoming edges
            graph.out_degree(node),          # how many outgoing edges
            graph.degree(node),              # total connections
            nx.closeness_centrality(graph, node),  # how central
            1.0 if graph.in_degree(node) == 0 else 0.0,  # is root/source
            1.0 if graph.out_degree(node) == 0 else 0.0,  # is leaf/terminal
            depth_from_root(graph, node),    # distance from source
            depth_to_leaf(graph, node),      # distance to terminal
            # ... pad to 16 features with zeros or additional topology metrics
        ])
    return torch.tensor(features)
```

### Edge encoding as attention mask

The heat kernel routes between ALL positions by default. Edges constrain routing:

```python
def build_edge_mask(graph, n_nodes):
    """Build attention mask from graph edges.
    
    Connected nodes: mask = 0 (allow routing)
    Disconnected nodes: mask = -inf (block routing)
    
    With k-hop relaxation: allow routing between nodes
    within k hops, not just direct edges.
    """
    adj = nx.adjacency_matrix(graph).todense()
    # k-hop: allow routing within 2 hops
    adj_2hop = (adj + adj @ adj).clip(0, 1)
    mask = torch.zeros(n_nodes, n_nodes)
    mask[adj_2hop == 0] = float('-inf')
    # Always allow self-attention
    mask.fill_diagonal_(0)
    return mask
```

### Why this creates the cluster structure MetricNet needs

```
ARC clusters:     same color → same embed component → low D² → heat kernel connects
Graph clusters:   same node type → same TypeEmbed component → low D² → heat kernel connects

ARC routing:      color groups route to each other within transformation
Graph routing:    type groups route to each other along edges

ARC phase transition trigger: embedding clustering reduces D² below softmax threshold
Graph phase transition trigger: type embedding clustering reduces D² — SAME MECHANISM
```

The MetricNet trained on ARC learned: "when I see clustered type embeddings, produce g with high CV that routes within clusters." Graph node type embeddings produce the SAME clustering pattern. The MetricNet should activate without retraining.

## Phase 2: Graph Reasoning Tasks

### Task A: Causal chain tracing

```python
# Input graph (the bridge→shortage chain):
graph = {
    "nodes": [
        {"id": "bridge_closure", "type": "event", "role": "root"},
        {"id": "truck_reroute", "type": "consequence", "role": "intermediate"},
        {"id": "landslide", "type": "event", "role": "intermediate"},
        {"id": "road_blocked", "type": "state", "role": "intermediate"},
        {"id": "food_shortage", "type": "consequence", "role": "terminal"},
    ],
    "edges": [
        {"src": "bridge_closure", "dst": "truck_reroute", "type": "causes"},
        {"src": "truck_reroute", "dst": "landslide", "type": "causes"},
        {"src": "landslide", "dst": "road_blocked", "type": "causes"},
        {"src": "road_blocked", "dst": "food_shortage", "type": "causes"},
    ],
    "query": {"type": "root_cause", "target": "food_shortage"}
}

# Expected output:
result = {
    "path": ["bridge_closure", "truck_reroute", "landslide", "road_blocked", "food_shortage"],
    "root_cause": "bridge_closure",
    "hops": 4,
    "chain_type": "cascading_failure"
}
```

**How LiquidARC processes this:**
1. Nodes embed with TypeEmbed (event/consequence/state create 3 clusters)
2. Edge mask allows routing along causal edges
3. Heat kernel diffuses "query signal" from food_shortage backward through the chain
4. After 16 ODE steps, bridge_closure and food_shortage are geometrically aligned (transitive routing through 4 intermediate nodes)
5. Root cause = the source node with strongest alignment to the query target
6. Path = ordered nodes by geodesic distance from target

**Training signal:** Given graph + query, predict the correct root cause node. CE loss on node classification. The MetricNet learns: "for root_cause queries, route BACKWARD along causal edges."

### Task B: Parallel chain separation

```python
graph = {
    "nodes": [
        # Chain A
        {"id": "pesticide", "type": "cause", "chain": "A"},
        {"id": "insect_death", "type": "event", "chain": "A"},
        {"id": "low_yield", "type": "state", "chain": "A"},
        {"id": "bread_price", "type": "consequence", "chain": "A"},
        # Chain B  
        {"id": "drought", "type": "cause", "chain": "B"},
        {"id": "rice_failure", "type": "event", "chain": "B"},
        {"id": "no_exports", "type": "state", "chain": "B"},
        {"id": "restaurant_closed", "type": "consequence", "chain": "B"},
    ],
    "edges": [
        # Chain A edges
        {"src": "pesticide", "dst": "insect_death", "type": "causes"},
        {"src": "insect_death", "dst": "low_yield", "type": "causes"},
        {"src": "low_yield", "dst": "bread_price", "type": "causes"},
        # Chain B edges
        {"src": "drought", "dst": "rice_failure", "type": "causes"},
        {"src": "rice_failure", "dst": "no_exports", "type": "causes"},
        {"src": "no_exports", "dst": "restaurant_closed", "type": "causes"},
        # NO cross-chain edges
    ],
    "query": {"type": "connection_check", "src": "pesticide", "dst": "restaurant_closed"}
}

# Expected output:
result = {
    "connected": False,
    "reason": "Nodes belong to separate causal chains (A and B) with no connecting edges",
    "chain_A": ["pesticide", "insect_death", "low_yield", "bread_price"],
    "chain_B": ["drought", "rice_failure", "no_exports", "restaurant_closed"]
}
```

**How LiquidARC processes this:**
1. Two disconnected subgraphs → edge mask blocks cross-chain routing
2. Heat kernel diffuses WITHIN each chain but NOT between them
3. MetricNet assigns different metric clusters to chain A and chain B
4. D² between chain A and chain B nodes is large (metrically far)
5. Connection check: geodesic distance from pesticide to restaurant_closed = infinity (no path)

### Task C: Structural analogy detection

```python
graph_A = {  # bridge cascade
    "nodes": [{"id": "A1", "type": "trigger"}, {"id": "A2", "type": "step"}, 
              {"id": "A3", "type": "step"}, {"id": "A4", "type": "outcome"}],
    "edges": [{"src": "A1", "dst": "A2"}, {"src": "A2", "dst": "A3"}, {"src": "A3", "dst": "A4"}]
}
graph_B = {  # ecology cascade
    "nodes": [{"id": "B1", "type": "trigger"}, {"id": "B2", "type": "step"},
              {"id": "B3", "type": "step"}, {"id": "B4", "type": "outcome"}],
    "edges": [{"src": "B1", "dst": "B2"}, {"src": "B2", "dst": "B3"}, {"src": "B3", "dst": "B4"}]
}
query = {"type": "analogy_check", "graph_a": "A", "graph_b": "B"}

# Expected output:
result = {
    "isomorphic": True,
    "mapping": {"A1": "B1", "A2": "B2", "A3": "B3", "A4": "B4"},
    "shared_pattern": "linear_cascade",
    "node_count": 4,
    "topology": "directed_chain"
}
```

**How LiquidARC processes this:**
1. Process graph A through ODE → capture metric signature (CV, D² distribution, tau profile)
2. Process graph B through ODE → capture metric signature
3. Compare signatures: if CV_A ≈ CV_B and D²_distribution_A ≈ D²_distribution_B → isomorphic
4. Node mapping: match by geodesic distance from root (A1 maps to B1 because both are distance 0 from root, A4 maps to B4 because both are distance 3)

### Task D: Scoped logic

```python
graph = {
    "nodes": [
        {"id": "senior_eng", "type": "role", "role": "scope"},
        {"id": "junior_dev", "type": "role", "role": "scope"},
        {"id": "security_cert", "type": "credential"},
        {"id": "network_exam", "type": "requirement"},
        {"id": "crypto_exam", "type": "requirement"},
        {"id": "linear_algebra", "type": "prerequisite"},
        {"id": "alt_pathway", "type": "prerequisite"},
    ],
    "edges": [
        {"src": "senior_eng", "dst": "security_cert", "type": "requires"},
        {"src": "security_cert", "dst": "network_exam", "type": "requires"},
        {"src": "security_cert", "dst": "crypto_exam", "type": "requires"},
        {"src": "crypto_exam", "dst": "linear_algebra", "type": "requires", "scope": "senior_eng"},
        {"src": "crypto_exam", "dst": "alt_pathway", "type": "requires", "scope": "junior_dev"},
    ],
    "query": {
        "type": "implication_check",
        "premise": "passed crypto_exam",
        "conclusion": "completed linear_algebra",
        "context_scope": "junior_dev"
    }
}

# Expected output:
result = {
    "valid": False,
    "reason": "Within junior_dev scope, crypto_exam requires alt_pathway, not linear_algebra",
    "scope_matters": True,
    "would_be_valid_in": "senior_eng"
}
```

## Phase 3: Training

### Dataset construction

Use existing graph reasoning benchmarks + synthetic generation:

1. **Synthetic causal chains:** Generate random DAGs with 3-10 nodes, assign types and edge labels. Create root_cause queries. 10,000 examples.

2. **Synthetic parallel chains:** Generate 2-4 independent DAGs, mix their nodes. Create connection_check queries. 5,000 examples.

3. **Structural analogy pairs:** Generate graph pairs with identical topology but different labels. Create analogy_check queries. 5,000 examples.

4. **Scoped logic graphs:** Generate dependency graphs with scope constraints. Create implication_check queries. 5,000 examples.

5. **Knowledge graph fragments (optional):** Extract small subgraphs from Freebase/Wikidata. Create multi-hop queries. 10,000 examples.

Total: ~35,000 examples. Modest — comparable to ARC training data size.

### Training loop

```python
# Model components
graph_embed = GraphNodeEmbedding(d_model=768)
dynamics = ContinuousDynamics(config)  # existing LiquidARC dynamics
output_head = GraphOutputHead(d_model=768, n_tasks=4)

# Criticality scaffolding (from sustained criticality experiments)
# D²/4τ ≈ 18, tau_quality, convergence coupling — all active

optimizer = Adam([
    {'params': graph_embed.parameters(), 'lr': 1e-3},    # embedding — fast
    {'params': dynamics.parameters(), 'lr': 1e-4},         # MetricNet — slower
    {'params': output_head.parameters(), 'lr': 1e-3},      # output — fast
])

for batch in dataloader:
    # Encode graph nodes
    h = graph_embed(batch.node_types, batch.roles, batch.struct_features)
    
    # Build edge mask
    mask = build_edge_mask(batch.adjacency, batch.n_nodes)
    
    # ODE integration with heat kernel routing
    dynamics.set_mask(mask)
    h_out = euler_integrate(dynamics, h, n_steps=16)
    
    # Task-specific output
    if batch.task == 'root_cause':
        logits = output_head.root_cause(h_out, batch.query_node)
        loss = F.cross_entropy(logits, batch.target_node)
    elif batch.task == 'connection_check':
        score = output_head.connection(h_out, batch.src_node, batch.dst_node)
        loss = F.binary_cross_entropy(score, batch.connected)
    elif batch.task == 'analogy':
        sig_a = output_head.signature(h_out_a)
        sig_b = output_head.signature(h_out_b)
        loss = analogy_loss(sig_a, sig_b, batch.isomorphic)
    elif batch.task == 'scoped_logic':
        logits = output_head.implication(h_out, batch.scope_node)
        loss = F.cross_entropy(logits, batch.valid)
    
    # Criticality losses (same as ARC experiments)
    crit_loss = compute_criticality_loss(dynamics, h_out)  # D²/4τ → 18
    tau_loss = compute_tau_quality_loss(dynamics)            # tau_mean → 1.0
    
    total_loss = loss + 0.01 * crit_loss + 0.05 * tau_loss
    total_loss.backward()
    optimizer.step()
```

### MetricNet initialization

Two options:

**Option A: From ARC checkpoint (recommended).** The d=768 post-transition checkpoint (CV≈7.5) already knows how to produce structured routing on categorical embeddings. Graph node type embeddings have the same categorical structure. The MetricNet may activate immediately on graph input without additional training.

**Option B: From scratch.** Random initialization with criticality scaffolding. The MetricNet learns graph routing from zero. May take longer (~6000 steps based on ARC experiments) but produces graph-specific routing without ARC bias.

Start with Option A. If graph accuracy is high from the beginning → ARC routing transfers. If not → fall back to Option B.

## Phase 4: MCP Integration

### New MCP tools

```python
@tool
def analyze_graph(graph_json: str, query_json: str) -> str:
    """Process a graph through LiquidARC and return structural analysis.
    
    Args:
        graph_json: JSON with nodes, edges, node_types, edge_types
        query_json: JSON with query type and parameters
    
    Returns:
        JSON with analysis results (paths, connections, patterns)
    """
    graph = parse_graph(graph_json)
    query = parse_query(query_json)
    
    # Encode and process
    h = graph_embed(graph)
    mask = build_edge_mask(graph)
    h_out = ode_integrate(h, mask)
    
    # Route to appropriate analysis
    if query.type == 'root_cause':
        return trace_root_cause(h_out, query.target)
    elif query.type == 'connection_check':
        return check_connection(h_out, query.src, query.dst)
    elif query.type == 'analogy':
        return detect_analogy(h_out, query.graph_b)
    elif query.type == 'shortest_path':
        return compute_geodesic_path(h_out, query.src, query.dst)

@tool
def compare_graphs(graph_a_json: str, graph_b_json: str) -> str:
    """Compare two graphs for structural similarity.
    
    Returns isomorphism check, node mapping, shared topology pattern.
    """

@tool  
def get_graph_diagnostics(graph_json: str) -> str:
    """Return geometric diagnostics for a graph.
    
    Returns CV, D²/4τ, tau distribution, metric clusters,
    per-node centrality in metric space.
    """
```

### LLM integration pattern

The LLM calls graph tools when it encounters structural reasoning:

```
User: "A factory closed, workers lost jobs, local shops lost customers, 
       the school lost funding from tax revenue. What started it all?"

LLM thinking: "This is a causal chain question. Let me extract the graph."

LLM → graph extraction (via structured output):
{
  "nodes": [
    {"id": "factory_closed", "type": "event"},
    {"id": "jobs_lost", "type": "consequence"},
    {"id": "shops_lost_customers", "type": "consequence"},
    {"id": "school_lost_funding", "type": "consequence"}
  ],
  "edges": [
    {"src": "factory_closed", "dst": "jobs_lost", "type": "causes"},
    {"src": "jobs_lost", "dst": "shops_lost_customers", "type": "causes"},
    {"src": "jobs_lost", "dst": "school_lost_funding", "type": "causes"}
  ]
}

LLM → analyze_graph(graph, {"type": "root_cause", "target": "school_lost_funding"})

LiquidARC returns:
{
  "path": ["factory_closed", "jobs_lost", "school_lost_funding"],
  "root_cause": "factory_closed",
  "hops": 2,
  "note": "jobs_lost is a branching point affecting both shops and school"
}

LLM generates: "The root cause was the factory closure. When the factory 
closed, workers lost their jobs, which had a branching effect: local shops 
lost customers AND the school lost tax-funded revenue."
```

## Experiment Design

### Experiment 1: ARC checkpoint on graph input (immediate test)

Load the d=768 post-transition checkpoint. Feed graph node embeddings. Measure:
- Does MetricNet produce CV > 3.0 on graph input? (ARC checkpoint produces CV≈7 on ARC)
- Does D²/4τ stay near 18?
- Do same-type nodes cluster (low within-type D²)?
- Do different-type nodes separate (high across-type D²)?

If YES → ARC routing transfers to graphs. Proceed to Task A test immediately.
If NO → MetricNet needs graph-specific training. Proceed to Phase 3.

### Experiment 2: Causal chain accuracy (Task A)

Generate 1000 causal chain graphs (3-7 nodes, linear chains).
Query: root cause for each terminal node.
Metric: accuracy (correct root cause identified).

Compare:
- LiquidARC (geometric routing on graph)
- Baseline: BFS from query node (graph algorithm, no learning)
- Baseline: GCN (standard graph neural network)

LiquidARC should match BFS on simple chains and potentially outperform on chains with branches/cycles where simple BFS is ambiguous.

### Experiment 3: Parallel chain separation (Task B)

Generate 500 mixed-chain scenarios (2-3 chains, 3-5 nodes each, interleaved).
Query: is node X in chain A connected to node Y in chain B?
Metric: accuracy + false positive rate.

This is where the MetricNet's cluster separation (B_within > B_across) directly applies. The same mechanism that emerged in our text experiments — but on data with actual cluster structure.

### Experiment 4: Structural analogy (Task C)

Generate 500 graph pairs (50% isomorphic, 50% not).
Query: are these structurally analogous?
Metric: accuracy, precision, recall.

This tests whether the metric SIGNATURE (CV distribution, D² profile) is a reliable isomorphism detector. Novel capability — standard GNNs need explicit graph matching algorithms for this.

### Experiment 5: Integration with LLM (end-to-end)

Use the causal chain test suite from our transformer experiments (bridge→shortage, parallel chains, etc.).
But now: LLM extracts graph → LiquidARC processes → LLM reads results.

Compare:
- Plain Qwen3-4B (no graph tools)
- Qwen3-4B + LiquidARC graph tools
- Qwen3-4B + hand-written graph algorithms (BFS, etc.)

This tests the FULL pipeline: does LiquidARC's geometric graph analysis actually help the LLM answer correctly on the tasks where plain LLM failed (5-hop chain, scope confusion)?

## Success Criteria

1. **ARC checkpoint transfers:** CV > 3.0 on graph node embeddings without retraining
2. **Causal chain tracing:** >90% accuracy on 3-7 node chains (BFS-level or better)
3. **Chain separation:** >95% accuracy on parallel chain connection queries
4. **Structural analogy:** >85% accuracy on isomorphism detection
5. **LLM integration:** improvement over plain LLM on at least 2 of 3 tasks where plain LLM fails (5-hop chain, scope logic, cross-chain contamination)
6. **Phase transition observed:** CV crosses threshold during graph processing (structural routing reorganization)

## What Transfers From Current Work

Everything geometric:
- ContinuousDynamics (MetricNet, TauNet, heat kernel, FFN, LTC)
- Sustained criticality system (D²/4τ loss, tau_quality, convergence coupling)
- Euler integration solver with norm homeostasis
- Step embeddings for depth-dependent routing
- All diagnostic infrastructure (CV, D², entropy, B_across/B_within)
- Post-transition d=768 checkpoint as initialization

What's new:
- GraphNodeEmbedding (replaces ARC/text embedding)
- Edge mask construction (replaces causal mask)
- Graph output heads (root cause, connection, analogy, logic)
- Synthetic graph dataset generation
- MCP tool interface for graph queries

## Timeline Estimate

- Phase 1 (graph encoding): 1-2 days
- Phase 2 (task implementation): 2-3 days  
- Phase 3 (training, if needed): 1-2 days (ARC-scale training, not LLM-scale)
- Phase 4 (MCP integration): 1-2 days
- Experiments 1-5: 2-3 days

Total: ~2 weeks for the full pipeline from graph encoding through LLM integration testing.

## Why This Should Work When Transformer Integration Didn't

| Transformer integration | Graph engine |
|---|---|
| LiquidARC competes with pretrained attention | LiquidARC IS the processor |
| Continuous text embeddings, no clusters | Categorical node types, natural clusters |
| 60M perturbing 4B | 60M as primary reasoner on its domain |
| MetricNet must learn LLM residual statistics | MetricNet already knows categorical routing (from ARC) |
| Results injected as attention bias (fights model) | Results delivered as text (model reads naturally) |
| Geometry must improve NTP (indirect signal) | Geometry must trace paths (direct signal) |
| Phase transition disrupts pretrained routing | Phase transition discovers routing (its purpose) |
