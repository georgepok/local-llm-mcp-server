import json
import os
import random
from typing import List, Dict, Any

def generate_cases(n_cases: int, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    domains = ["contract_breach", "tort_negligence", "ip_infringement"]
    cases: List[Dict[str, Any]] = []
    case_counter = 0

    for domain in domains:
        for shape_idx in range(6):
            for _ in range(max(1, n_cases // (len(domains) * 6))):
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
                node_types = ["event", "consequence", "cause", "state"]
                for i, ntype in enumerate(node_types):
                    node_id = f"{prefix}{ntype}_{i}"
                    role = "root" if i == 0 else ("intermediate" if i < len(node_types) - 1 else "terminal")
                    nodes.append({"id": node_id, "type": ntype, "role": role})
                    if i > 0:
                        edges.append({
                            "src": f"{prefix}{node_types[i-1]}_{i-1}",
                            "dst": node_id,
                            "type": "causes",
                            "scope": None
                        })
                # Shape 0: linear chain of 3 nodes (domain A)
                # Shape 1: linear chain of 5 nodes (domain A)
                # Shape 2: tree with 1 root and 2 children (domain B)
                # Shape 3: diamond with 4 nodes (domain B)
                # Shape 4: star with 1 root and 4 leaves (domain C)
                # Shape 5: linear chain of 6 nodes (domain C)
                if shape_idx == 0:  # linear chain of 3 nodes
                    pass
                elif shape_idx == 1:  # linear chain of 5 nodes
                    node_types = ["event", "consequence", "cause", "state", "event"]
                    for i, ntype in enumerate(node_types):
                        node_id = f"{prefix}{ntype}_{i}"
                        role = "root" if i == 0 else ("intermediate" if i < len(node_types) - 1 else "terminal")
                        nodes.append({"id": node_id, "type": ntype, "role": role})
                        if i > 0:
                            edges.append({
                                "src": f"{prefix}{node_types[i-1]}_{i-1}",
                                "dst": node_id,
                                "type": "causes",
                                "scope": None
                            })
                elif shape_idx == 2:  # tree with 2 children
                    node_types = ["event", "consequence", "cause", "consequence"]
                    for i, ntype in enumerate(node_types):
                        node_id = f"{prefix}{ntype}_{i}"
                        role = "root" if i == 0 else ("intermediate" if i < len(node_types) - 1 else "terminal")
                        nodes.append({"id": node_id, "type": ntype, "role": role})
                        if i > 0:
                            edges.append({
                                "src": f"{prefix}{node_types[i-1]}_{i-1}",
                                "dst": node_id,
                                "type": "causes",
                                "scope": None
                            })
                elif shape_idx == 3:  # diamond
                    node_types = ["event", "consequence", "cause", "consequence", "terminal"]
                    for i, ntype in enumerate(node_types):
                        node_id = f"{prefix}{ntype}_{i}"
                        role = "root" if i == 0 else ("intermediate" if i < len(node_types) - 1 else "terminal")
                        nodes.append({"id": node_id, "type": ntype, "role": role})
                        if i > 0:
                            edges.append({
                                "src": f"{prefix}{node_types[i-1]}_{i-1}",
                                "dst": node_id,
                                "type": "causes",
                                "scope": None
                            })
                elif shape_idx == 4:  # star
                    node_types = ["event", "consequence", "cause", "consequence", "consequence", "consequence"]
                    for i, ntype in enumerate(node_types):
                        node_id = f"{prefix}{ntype}_{i}"
                        role = "root" if i == 0 else ("intermediate" if i < len(node_types) - 1 else "terminal")
                        nodes.append({"id": node_id, "type": ntype, "role": role})
                        if i > 0:
                            edges.append({
                                "src": f"{prefix}{node_types[i-1]}_{i-1}",
                                "dst": node_id,
                                "type": "causes",
                                "scope": None
                            })
                elif shape_idx == 5:  # linear chain of 6 nodes
                    node_types = ["event", "consequence", "cause", "state", "event", "terminal"]
                    for i, ntype in enumerate(node_types):
                        node_id = f"{prefix}{ntype}_{i}"
                        role = "root" if i == 0 else ("intermediate" if i < len(node_types) - 1 else "terminal")
                        nodes.append({"id": node_id, "type": ntype, "role": role})
                        if i > 0:
                            edges.append({
                                "src": f"{prefix}{node_types[i-1]}_{i-1}",
                                "dst": node_id,
                                "type": "causes",
                                "scope": None
                            })
                fragment = {"nodes": nodes, "edges": edges}
                cases.append({
                    "case_id": case_id,
                    "domain": domain,
                    "shape": shape_tag,
                    "text": text,
                    "fragment": fragment
                })
    return cases
