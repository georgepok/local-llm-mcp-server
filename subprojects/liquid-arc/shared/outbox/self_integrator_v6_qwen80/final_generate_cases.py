import json
import os
import random
from typing import List, Dict, Any

def generate_cases(n_cases: int, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    domains = ["contract_breach", "tort_negligence", "ip_infringement"]
    cases: List[Dict[str, Any]] = []
    case_counter = 0

    # Define 6 distinct topological shapes with unique structures
    # Each shape has unique node count, branching, and edge type
    shape_configs = [
        # Shape 0: domain A - linear chain, 3 nodes, edge type 'causes'
        {"domain": "contract_breach", "shape_id": 0, "nodes": 3, "edge_type": "causes", "structure": "linear"},
        # Shape 1: domain A - linear chain, 5 nodes, edge type 'precedes'
        {"domain": "contract_breach", "shape_id": 1, "nodes": 5, "edge_type": "precedes", "structure": "linear"},
        # Shape 2: domain B - tree: 1 root with 2 children (3 nodes total)
        {"domain": "tort_negligence", "shape_id": 0, "nodes": 3, "edge_type": "leads_to", "structure": "tree"},
        # Shape 3: domain B - diamond: 1 root → 2 middles → 1 terminal
        {"domain": "tort_negligence", "shape_id": 1, "nodes": 4, "edge_type": "contributes_to", "structure": "diamond"},
        # Shape 4: domain C - star: 1 root with 4 leaves
        {"domain": "ip_infringement", "shape_id": 0, "nodes": 5, "edge_type": "enables", "structure": "star"},
        # Shape 5: domain C - linear chain, 6 nodes, edge type 'enables'
        {"domain": "ip_infringement", "shape_id": 1, "nodes": 6, "edge_type": "enables", "structure": "linear"},
    ]

    # Distribute cases evenly across the 6 shapes
    cases_per_shape = max(1, n_cases // len(shape_configs))
    
    for config in shape_configs:
        domain = config["domain"]
        shape_id = config["shape_id"]
        n_nodes = config["nodes"]
        edge_type = config["edge_type"]
        structure = config["structure"]
        shape_tag = f"{domain}_shape_{shape_id}"
        
        for _ in range(cases_per_shape):
            case_id = f"case_{case_counter}"
            case_counter += 1
            
            # Generate text based on domain and shape
            if domain == "contract_breach":
                contract_type = random.choice(["service", "sales", "employment"])
                consequence = random.choice(["damages", "financial loss", "breach penalty"])
                text = (
                    f"A plaintiff alleges that the defendant failed to perform under a {contract_type} "
                    f"{domain} scenario. The failure resulted in {consequence}, and the dispute centers "
                    f"on whether the sequence of events constitutes a material breach."
                )
            elif domain == "tort_negligence":
                duty_type = random.choice(["duty of care", "professional obligation", "safety standard"])
                harm = random.choice(["injury", "property damage", "emotional distress"])
                text = (
                    f"A plaintiff claims the defendant breached a {duty_type}, resulting in {harm}. "
                    f"The case examines whether the defendant's actions or omissions directly led to "
                    f"the harm through a chain of events."
                )
            else:  # ip_infringement
                ip_type = random.choice(["patent", "copyright", "trademark"])
                action = random.choice(["unauthorized use", "copying", "distribution"])
                text = (
                    f"A plaintiff alleges unauthorized {action} of their {ip_type} rights. "
                    f"The infringement occurred through a series of steps that enabled the violation, "
                    f"and the plaintiff seeks remedies for the resulting economic harm."
                )
            
            # Build graph based on structure
            nodes = []
            edges = []
            prefix = f"{case_id}_"
            # Use distinct node types per shape to increase topological diversity
            if structure == "linear":
                node_types = ["event", "action", "consequence"] if n_nodes == 3 else \
                            ["event", "action", "state", "trigger", "consequence"] if n_nodes == 5 else \
                            ["event", "action", "state", "trigger", "mediator", "consequence"]
            elif structure == "tree":
                node_types = ["root", "actor", "outcome"]
            elif structure == "diamond":
                node_types = ["origin", "factor_a", "factor_b", "result"]
            elif structure == "star":
                node_types = ["hub", "leaf_a", "leaf_b", "leaf_c", "leaf_d"]
            
            # Generate nodes with unique IDs
            for i in range(n_nodes):
                ntype = node_types[i % len(node_types)]
                node_id = f"{prefix}{ntype}_{i}"
                role = "root" if i == 0 else ("terminal" if i == n_nodes - 1 else "intermediate")
                nodes.append({"id": node_id, "type": ntype, "role": role})
            
            # Build edges based on structure
            if structure == "linear":
                # Linear chain: 0->1->2->...->n-1
                for i in range(1, n_nodes):
                    try:
                        src_id = nodes[i-1]["id"]
                        dst_id = nodes[i]["id"]
                        edges.append({
                            "src": src_id,
                            "dst": dst_id,
                            "type": edge_type,
                            "scope": None
                        })
                    except Exception as e:
                        continue
            
            elif structure == "tree":
                # Root (0) has two children (1, 2)
                try:
                    edges.append({
                        "src": nodes[0]["id"],
                        "dst": nodes[1]["id"],
                        "type": edge_type,
                        "scope": None
                    })
                    edges.append({
                        "src": nodes[0]["id"],
                        "dst": nodes[2]["id"],
                        "type": edge_type,
                        "scope": None
                    })
                except Exception as e:
                    continue
            
            elif structure == "diamond":
                # Root (0) -> two middles (1,2) -> terminal (3)
                try:
                    edges.append({
                        "src": nodes[0]["id"],
                        "dst": nodes[1]["id"],
                        "type": edge_type,
                        "scope": None
                    })
                    edges.append({
                        "src": nodes[0]["id"],
                        "dst": nodes[2]["id"],
                        "type": edge_type,
                        "scope": None
                    })
                    edges.append({
                        "src": nodes[1]["id"],
                        "dst": nodes[3]["id"],
                        "type": edge_type,
                        "scope": None
                    })
                    edges.append({
                        "src": nodes[2]["id"],
                        "dst": nodes[3]["id"],
                        "type": edge_type,
                        "scope": None
                    })
                except Exception as e:
                    continue
            
            elif structure == "star":
                # Root (0) connected to 4 leaves (1,2,3,4)
                try:
                    root_id = nodes[0]["id"]
                    for i in range(1, n_nodes):
                        leaf_id = nodes[i]["id"]
                        edges.append({
                            "src": root_id,
                            "dst": leaf_id,
                            "type": edge_type,
                            "scope": None
                        })
                except Exception as e:
                    continue
            
            fragment = {"nodes": nodes, "edges": edges}
            cases.append({
                "case_id": case_id,
                "domain": domain,
                "shape": shape_tag,
                "text": text,
                "fragment": fragment
            })
    
    # If we need more cases due to rounding, add one per shape until we reach n_cases
    while len(cases) < n_cases:
        config = shape_configs[len(cases) % len(shape_configs)]
        domain = config["domain"]
        shape_id = config["shape_id"]
        shape_tag = f"{domain}_shape_{shape_id}"
        case_id = f"case_{case_counter}"
        case_counter += 1
        
        if domain == "contract_breach":
            contract_type = random.choice(["service", "sales", "employment"])
            consequence = random.choice(["damages", "financial loss", "breach penalty"])
            text = (
                f"A plaintiff alleges that the defendant failed to perform under a {contract_type} "
                f"{domain} scenario. The failure resulted in {consequence}, and the dispute centers "
                f"on whether the sequence of events constitutes a material breach."
            )
        elif domain == "tort_negligence":
            duty_type = random.choice(["duty of care", "professional obligation", "safety standard"])
            harm = random.choice(["injury", "property damage", "emotional distress"])
            text = (
                f"A plaintiff claims the defendant breached a {duty_type}, resulting in {harm}. "
                f"The case examines whether the defendant's actions or omissions directly led to "
                f"the harm through a chain of events."
            )
        else:  # ip_infringement
            ip_type = random.choice(["patent", "copyright", "trademark"])
            action = random.choice(["unauthorized use", "copying", "distribution"])
            text = (
                f"A plaintiff alleges unauthorized {action} of their {ip_type} rights. "
                f"The infringement occurred through a series of steps that enabled the violation, "
                f"and the plaintiff seeks remedies for the resulting economic harm."
            )
        
        n_nodes = config["nodes"]
        edge_type = config["edge_type"]
        structure = config["structure"]
        nodes = []
        edges = []
        prefix = f"{case_id}_"
        
        # Use distinct node types per shape to increase topological diversity
        if structure == "linear":
            node_types = ["event", "action", "consequence"] if n_nodes == 3 else \
                        ["event", "action", "state", "trigger", "consequence"] if n_nodes == 5 else \
                        ["event", "action", "state", "trigger", "mediator", "consequence"]
        elif structure == "tree":
            node_types = ["root", "actor", "outcome"]
        elif structure == "diamond":
            node_types = ["origin", "factor_a", "factor_b", "result"]
        elif structure == "star":
            node_types = ["hub", "leaf_a", "leaf_b", "leaf_c", "leaf_d"]
        
        # Generate nodes with unique IDs
        for i in range(n_nodes):
            ntype = node_types[i % len(node_types)]
            node_id = f"{prefix}{ntype}_{i}"
            role = "root" if i == 0 else ("terminal" if i == n_nodes - 1 else "intermediate")
            nodes.append({"id": node_id, "type": ntype, "role": role})
        
        # Build edges based on structure
        if structure == "linear":
            # Linear chain: 0->1->2->...->n-1
            try:
                for i in range(1, n_nodes):
                    src_id = nodes[i-1]["id"]
                    dst_id = nodes[i]["id"]
                    edges.append({
                        "src": src_id,
                        "dst": dst_id,
                        "type": edge_type,
                        "scope": None
                    })
            except Exception as e:
                continue
        
        elif structure == "tree":
            # Root (0) has two children (1, 2)
            try:
                edges.append({
                    "src": nodes[0]["id"],
                    "dst": nodes[1]["id"],
                    "type": edge_type,
                    "scope": None
                })
                edges.append({
                    "src": nodes[0]["id"],
                    "dst": nodes[2]["id"],
                    "type": edge_type,
                    "scope": None
                })
            except Exception as e:
                continue
        
        elif structure == "diamond":
            # Root (0) -> two middles (1,2) -> terminal (3)
            try:
                edges.append({
                    "src": nodes[0]["id"],
                    "dst": nodes[1]["id"],
                    "type": edge_type,
                    "scope": None
                })
                edges.append({
                    "src": nodes[0]["id"],
                    "dst": nodes[2]["id"],
                    "type": edge_type,
                    "scope": None
                })
                edges.append({
                    "src": nodes[1]["id"],
                    "dst": nodes[3]["id"],
                    "type": edge_type,
                    "scope": None
                })
                edges.append({
                    "src": nodes[2]["id"],
                    "dst": nodes[3]["id"],
                    "type": edge_type,
                    "scope": None
                })
            except Exception as e:
                continue
        
        elif structure == "star":
            # Root (0) connected to 4 leaves (1,2,3,4)
            try:
                root_id = nodes[0]["id"]
                for i in range(1, n_nodes):
                    leaf_id = nodes[i]["id"]
                    edges.append({
                        "src": root_id,
                        "dst": leaf_id,
                        "type": edge_type,
                        "scope": None
                    })
            except Exception as e:
                continue
        
        fragment = {"nodes": nodes, "edges": edges}
        cases.append({
            "case_id": case_id,
            "domain": domain,
            "shape": shape_tag,
            "text": text,
            "fragment": fragment
        })
    
    return cases
