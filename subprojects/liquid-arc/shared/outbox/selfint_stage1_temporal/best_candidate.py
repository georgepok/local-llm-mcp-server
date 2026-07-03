import networkx as nx

def query_temporal(self, query_nodes, k):
    if not query_nodes:
        return []
    
    # Compute mean last_seen of query nodes
    query_last_seen = [self.G.nodes[n]["last_seen"] for n in query_nodes if "last_seen" in self.G.nodes[n]]
    if not query_last_seen:
        return []
    q_mean = sum(query_last_seen) / len(query_last_seen)
    
    # Find candidate nodes (not in query_nodes)
    candidates = [n for n in self.G.nodes() if n not in query_nodes and "last_seen" in self.G.nodes[n]]
    if not candidates:
        return []
    
    # Compute temporal scores
    temporal_scores = []
    max_delta = 0.0
    for n in candidates:
        delta = abs(self.G.nodes[n]["last_seen"] - q_mean)
        max_delta = max(max_delta, delta)
    
    for n in candidates:
        delta = abs(self.G.nodes[n]["last_seen"] - q_mean)
        temporal_score = 1.0 - (delta / max_delta) if max_delta > 0 else 1.0
        temporal_scores.append((n, temporal_score))
    
    # Compute causal scores using shortest path on undirected graph
    G_undirected = self.G.to_undirected()
    causal_scores = {}
    for n in candidates:
        min_dist = float('inf')
        for q in query_nodes:
            if nx.has_path(G_undirected, q, n):
                dist = nx.shortest_path_length(G_undirected, q, n)
                min_dist = min(min_dist, dist)
        if min_dist == float('inf'):
            causal_scores[n] = 0.0
        else:
            causal_scores[n] = 1.0 / (1.0 + min_dist)
    
    # Combine scores
    results = []
    for n, temporal_score in temporal_scores:
        causal_score = causal_scores[n]
        combined_score = 0.5 * temporal_score + 0.5 * causal_score
        results.append({
            "id": n,
            "type": self.G.nodes[n]["type"],
            "role": self.G.nodes[n]["role"],
            "temporal_score": temporal_score,
            "causal_score": causal_score,
            "combined_score": combined_score,
            "source": "temporal"
        })
    
    # Sort by combined_score descending and return top-k
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results[:k]
