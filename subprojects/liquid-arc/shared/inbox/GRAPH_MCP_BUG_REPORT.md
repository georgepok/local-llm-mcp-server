# GRAPH MCP SERVER — Runtime Bug Report

## Error

All three graph MCP tools produce the same error on every call:

```
Error calling tool 'analyze_graph': too many dimensions 'str'
Error calling tool 'compare_graphs': too many dimensions 'str'  
Error calling tool 'get_graph_diagnostics': too many dimensions 'str'
```

Tested with multiple graph inputs from minimal (2 nodes) to full (6 nodes). Error is consistent — not input-dependent.

## Diagnosis

`too many dimensions 'str'` is a PyTorch error when a string value reaches tensor construction. This means somewhere in the inference pipeline, a string (likely a node type name, role name, or node ID) is being passed to `torch.tensor()` or `.unsqueeze()` or similar operation instead of an integer index.

## Likely locations (in order of probability)

### 1. Type/role string → index mapping missing in inference path

Training pipeline has the mapping (type string → integer index for embedding lookup). If `graph_engine_inference.py` skips this mapping and passes raw strings from the JSON to `GraphNodeEmbedding`, the `nn.Embedding` lookup would fail because it expects integer indices.

Check in `GraphEngine.analyze_graph()`:
```python
# Does this exist?
node_types = [self.type_to_idx[n["type"]] for n in graph["nodes"]]
# Or is raw string being passed?
node_types = [n["type"] for n in graph["nodes"]]  # ← would cause this error
```

### 2. Structural features returning strings

`compute_structural_features` uses networkx functions that might return string-typed results for some graph configurations. Check that all 16 feature dimensions are float/int, not string.

### 3. Node ID handling

If the edge mask builder uses node IDs directly as tensor indices (expecting integers like 0,1,2) but receives string IDs like "bridge_closure", the adjacency matrix construction would fail.

## Fix

The inference wrapper needs to:
1. Map node type strings to integer indices using the same vocabulary as training
2. Map role strings to integer indices  
3. Map node ID strings to positional integers (0, 1, 2, ...)
4. Ensure all structural features are numeric

The mapping vocabularies should be saved with the checkpoint or defined as constants in the inference module.

## Test cases for verification

After fix, these should all succeed:

```json
// Minimal 2-node graph
{"nodes": [{"id": "A", "type": "cause", "role": "root"}, {"id": "B", "type": "outcome", "role": "terminal"}], 
 "edges": [{"src": "A", "dst": "B", "type": "causes"}]}

// 5-hop causal chain  
{"nodes": [{"id": "bridge_closure", "type": "event", "role": "root"}, ...5 more...],
 "edges": [...5 causal edges...]}

// Disconnected graph (2 chains)
{"nodes": [...chain A nodes..., ...chain B nodes...],
 "edges": [...chain A edges..., ...chain B edges...]}  // no cross-chain edges
```
