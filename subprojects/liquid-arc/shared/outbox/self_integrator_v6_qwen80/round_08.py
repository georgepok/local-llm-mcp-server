# Auto-assembled by SelfIntegrator v4
import json
import os
import random
import math
from typing import List, Dict, Any, Set, Tuple, Optional


# --- generate_cases ---
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

# --- ingest_and_store ---
def ingest_and_store(cases, db_path, patterns_path, checkpoint):
    import json
    import os
    from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB
    from liquid_arc.graph_rag.decoupled.ode_engine import SubgraphODEEngine
    from liquid_arc.navigator_patterns import PatternLibrary

    n_ingested = 0
    n_signatures = 0
    errors = []

    db = KnowledgeGraphDB(db_path)
    lib = PatternLibrary(patterns_path)
    engine = SubgraphODEEngine(checkpoint, device="cpu")  # Instantiate once

    for case in cases:
        case_id = case.get("case_id")
        fragment = case.get("fragment")
        if not isinstance(fragment, dict) or "nodes" not in fragment or "edges" not in fragment:
            errors.append(f"{case_id}: fragment missing required keys")
            continue

        try:
            db.add_fragment(fragment, source_text=case_id, chunk_id=case_id, autosave=True)
            n_ingested += 1
        except Exception as e:
            errors.append(f"{case_id}: add_fragment failed - {e}")
            continue

        try:
            # Correctly extract node IDs from list of dicts
            node_ids = [n["id"] for n in fragment["nodes"]]
            neighbors = db.get_neighbors(node_ids, hops=2, direction="both")
            # Include original nodes in subgraph extraction
            subgraph_nodes = list(neighbors) + node_ids
            subgraph = db.extract_subgraph(subgraph_nodes, max_nodes=30)
            
            # Ensure subgraph has at least 2 nodes before computing signature
            if len(subgraph["nodes"]) < 2:
                errors.append(f"{case_id}: subgraph has fewer than 2 nodes")
                continue
                
            signature = engine.compute_signature(subgraph)
            # Pass signature directly, not as a list-of-lists
            lib.store(signature, {"label": case_id})
            n_signatures += 1
        except Exception as e:
            errors.append(f"{case_id}: signature computation failed - {e}")

    # Do NOT clear the database - downstream code needs it
    return {"n_ingested": n_ingested, "n_signatures": n_signatures, "errors": errors}

# --- evaluate_precedents ---
def evaluate_precedents(test_cases, patterns_path, checkpoint, work_dir, train_shape_map):
    import os
    from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB
    from liquid_arc.graph_rag.decoupled.ode_engine import SubgraphODEEngine
    from liquid_arc.navigator_patterns import PatternLibrary

    n_cases = len(test_cases)
    per_case = []
    correct = 0

    db = KnowledgeGraphDB(work_dir)
    lib = PatternLibrary(patterns_path)
    engine = SubgraphODEEngine(checkpoint, device="cpu")  # Instantiate once

    for case in test_cases:
        case_id = case.get("case_id")
        expected_shape = case.get("shape")
        fragment = case.get("fragment")

        if not isinstance(fragment, dict) or "nodes" not in fragment or "edges" not in fragment:
            per_case.append({
                "case_id": case_id,
                "expected_shape": expected_shape,
                "matched_label": None,
                "cosine": None,
                "matched_shape": None,
                "error": "'fragment' is not a dict with 'nodes' and 'edges'"
            })
            continue

        try:
            # Extract node IDs correctly from list of dicts
            node_ids = [n["id"] for n in fragment["nodes"]]
            db.add_fragment(fragment, source_text=case_id, chunk_id=case_id, autosave=True)
            
            # Get neighbors of actual node IDs, not case_id
            neighbors = db.get_neighbors(node_ids, hops=2, direction='both')
            # Include original nodes in subgraph
            subgraph_nodes = list(neighbors) + node_ids
            subgraph = db.extract_subgraph(subgraph_nodes, max_nodes=200)
            
            # Compute signature on subgraph, pass directly (not [signature])
            signature = engine.compute_signature(subgraph)
            
            # Find nearest using signature directly (not [signature])
            result = lib.find_nearest(signature, threshold=0.5)

            matched_label = result.get("label") if result else None
            cosine = result.get("similarity") if result else None
            matched_shape = train_shape_map.get(matched_label) if matched_label else None

            if matched_shape == expected_shape:
                correct += 1

            per_case.append({
                "case_id": case_id,
                "expected_shape": expected_shape,
                "matched_label": matched_label,
                "cosine": cosine,
                "matched_shape": matched_shape,
                "error": None
            })
        except Exception as e:
            per_case.append({
                "case_id": case_id,
                "expected_shape": expected_shape,
                "matched_label": None,
                "cosine": None,
                "matched_shape": None,
                "error": f"signature computation failed - {e}"
            })
        finally:
            # Do NOT clear DB - downstream code needs it
            pass

    accuracy = correct / n_cases if n_cases else 0.0
    return {"n_cases": n_cases, "accuracy": accuracy, "per_case": per_case}


if __name__ == '__main__':
    out_dir = os.environ.get('OUT_DIR', '/tmp/self_int_v4_out')
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = os.environ['CHECKPOINT']
    try:
        train = generate_cases(100, seed=42)
        test  = generate_cases(10, seed=99)
        ingest_report = ingest_and_store(
            train,
            os.path.join(out_dir, 'db.json'),
            os.path.join(out_dir, 'patterns.json'),
            checkpoint)
        train_map = {c['case_id']: c['shape'] for c in train}
        eval_report = evaluate_precedents(
            test, os.path.join(out_dir, 'patterns.json'),
            checkpoint,
            os.path.join(out_dir, 'eval_scratch'),
            train_map)
        final = {
            'ingest': ingest_report,
            'n_test_cases': len(test),
            'accuracy': eval_report.get('accuracy'),
            'per_case': eval_report.get('per_case'),
        }
    except Exception as e:
        import traceback
        final = {'fatal_error': str(e),
                  'traceback': traceback.format_exc()}
    print('RESULT_JSON: ' + json.dumps(final, default=str))
