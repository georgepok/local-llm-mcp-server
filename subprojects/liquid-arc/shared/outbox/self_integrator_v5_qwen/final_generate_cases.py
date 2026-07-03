import json
import os
import random
from typing import List, Dict, Any

def generate_cases(n_cases: int, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    domains = ["contract_breach", "tort_negligence", "ip_infringement"]
    # Define 6 distinct topological shapes to ensure separable signatures
    # Shape 0: linear chain, 3 nodes, edge type 'causes'
    # Shape 1: linear chain, 5 nodes, edge type 'precedes'
    # Shape 2: tree — 1 root with 2 children (3 nodes)
    # Shape 3: diamond — 1 root → 2 middles → 1 terminal
    # Shape 4: star — 1 root with 4 leaves
    # Shape 5: linear chain, 6 nodes, edge type 'enables'
    
    cases: List[Dict[str, Any]] = []
    case_counter = 0

    # We need to generate cases for each domain and each shape
    # The total number of cases is n_cases. We distribute them evenly.
    # However, the problem states we need 6 shapes. Let's assume we want
    # to cover all 6 shapes for each domain if possible, or just generate
    # a mix. The original code did 2 shapes per domain. The prompt says
    # "The 6 shapes in generate_cases produce topologically indistinguishable signatures."
    # and lists 6 shapes. It implies we should generate cases for these 6 shapes.
    # Let's generate cases for each domain and each of the 6 shapes.
    # If n_cases is small, we might not cover all. Let's try to cover all 6 shapes
    # for each domain, cycling through them.
    
    # Actually, looking at the original code, it iterates domains then shapes.
    # Let's keep that structure but expand shapes to 6.
    # We need to ensure we generate enough cases. Let's generate at least 1 case per
    # (domain, shape) pair, and then distribute the rest.
    
    num_domains = len(domains)
    num_shapes = 6
    total_combinations = num_domains * num_shapes
    
    # Calculate how many cases per combination
    cases_per_combo = max(1, n_cases // total_combinations)
    remainder = n_cases - (cases_per_combo * total_combinations)
    
    # If remainder > 0, we add extra cases to the first 'remainder' combinations
    # But to keep it simple and deterministic, let's just generate cases_per_combo for each
    # and if n_cases is larger, we might need more. Let's just generate cases_per_combo
    # for each combination. If n_cases is not exactly divisible, we'll have fewer cases.
    # The prompt says "Keep the same function signature and output shape."
    # The original code generated roughly n_cases cases. Let's try to generate exactly n_cases.
    
    # Let's create a list of (domain, shape_idx) pairs
    combos = []
    for domain in domains:
        for shape_idx in range(num_shapes):
            combos.append((domain, shape_idx))
    
    # Distribute n_cases among combos
    # Each combo gets at least 1 case if n_cases >= total_combinations
    # Otherwise, some combos get 0.
    
    # Let's just iterate through combos and generate cases until we reach n_cases
    case_counter = 0
    for domain, shape_idx in combos:
        if case_counter >= n_cases:
            break
        # Determine how many cases for this combo
        # Simple round-robin distribution
        count = cases_per_combo
        if case_counter + count > n_cases:
            count = n_cases - case_counter
        if count <= 0:
            break
            
        for _ in range(count):
            case_id = f"case_{case_counter}"
            case_counter += 1
            shape_tag = f"{domain}_shape_{shape_idx}"
            text = (
                f"A plaintiff alleges that the defendant failed to perform "
                f"under a {domain} scenario involving a {random.choice(['contract', 'duty', 'licensed work'])}. "
                f"The dispute centers on whether the breach caused direct {random.choice(['damages', 'injury', 'loss'])}. "
                f"Witnesses report conflicting accounts of the event timeline."
            )
            
            nodes = []
            edges = []
            prefix = f"{case_id}_"
            
            # Build the specific topology for shape_idx
            if shape_idx == 0:
                # Linear chain, 3 nodes, edge type 'causes'
                # Node 0 -> Node 1 -> Node 2
                node_ids = [f"{prefix}n0", f"{prefix}n1", f"{prefix}n2"]
                node_types = ["event", "cause", "consequence"]
                for i, nid in enumerate(node_ids):
                    nodes.append({"id": nid, "type": node_types[i], "role": "root" if i == 0 else ("intermediate" if i == 1 else "terminal")})
                edges.append({"src": node_ids[0], "dst": node_ids[1], "type": "causes", "scope": None})
                edges.append({"src": node_ids[1], "dst": node_ids[2], "type": "causes", "scope": None})
                
            elif shape_idx == 1:
                # Linear chain, 5 nodes, edge type 'precedes'
                node_ids = [f"{prefix}n{i}" for i in range(5)]
                node_types = ["event", "state", "event", "state", "consequence"]
                for i, nid in enumerate(node_ids):
                    role = "root" if i == 0 else ("intermediate" if i < 4 else "terminal")
                    nodes.append({"id": nid, "type": node_types[i], "role": role})
                for i in range(4):
                    edges.append({"src": node_ids[i], "dst": node_ids[i+1], "type": "precedes", "scope": None})
                    
            elif shape_idx == 2:
                # Tree — 1 root with 2 children (3 nodes)
                root_id = f"{prefix}root"
                child1_id = f"{prefix}child1"
                child2_id = f"{prefix}child2"
                nodes.append({"id": root_id, "type": "event", "role": "root"})
                nodes.append({"id": child1_id, "type": "consequence", "role": "leaf"})
                nodes.append({"id": child2_id, "type": "consequence", "role": "leaf"})
                edges.append({"src": root_id, "dst": child1_id, "type": "causes", "scope": None})
                edges.append({"src": root_id, "dst": child2_id, "type": "causes", "scope": None})
                
            elif shape_idx == 3:
                # Diamond — 1 root → 2 middles → 1 terminal
                root_id = f"{prefix}root"
                mid1_id = f"{prefix}mid1"
                mid2_id = f"{prefix}mid2"
                term_id = f"{prefix}terminal"
                nodes.append({"id": root_id, "type": "event", "role": "root"})
                nodes.append({"id": mid1_id, "type": "cause", "role": "intermediate"})
                nodes.append({"id": mid2_id, "type": "cause", "role": "intermediate"})
                nodes.append({"id": term_id, "type": "consequence", "role": "terminal"})
                edges.append({"src": root_id, "dst": mid1_id, "type": "causes", "scope": None})
                edges.append({"src": root_id, "dst": mid2_id, "type": "causes", "scope": None})
                edges.append({"src": mid1_id, "dst": term_id, "type": "causes", "scope": None})
                edges.append({"src": mid2_id, "dst": term_id, "type": "causes", "scope": None})
                
            elif shape_idx == 4:
                # Star — 1 root with 4 leaves
                root_id = f"{prefix}root"
                leaf_ids = [f"{prefix}leaf{i}" for i in range(4)]
                nodes.append({"id": root_id, "type": "event", "role": "root"})
                for lid in leaf_ids:
                    nodes.append({"id": lid, "type": "consequence", "role": "leaf"})
                for lid in leaf_ids:
                    edges.append({"src": root_id, "dst": lid, "type": "causes", "scope": None})
                    
            elif shape_idx == 5:
                # Linear chain, 6 nodes, edge type 'enables'
                node_ids = [f"{prefix}n{i}" for i in range(6)]
                node_types = ["event", "state", "event", "state", "event", "consequence"]
                for i, nid in enumerate(node_ids):
                    role = "root" if i == 0 else ("intermediate" if i < 5 else "terminal")
                    nodes.append({"id": nid, "type": node_types[i], "role": role})
                for i in range(5):
                    edges.append({"src": node_ids[i], "dst": node_ids[i+1], "type": "enables", "scope": None})
            
            fragment = {"nodes": nodes, "edges": edges}
            cases.append({
                "case_id": case_id,
                "domain": domain,
                "shape": shape_tag,
                "text": text,
                "fragment": fragment
            })
            
    return cases
