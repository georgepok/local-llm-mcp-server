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
    shape_configs = [
        # Domain A: contract_breach
        {
            "domain": "contract_breach",
            "shape_id": 0,
            "node_count": 3,
            "edge_type": "causes",
            "structure": "linear",  # chain: A -> B -> C
            "node_roles": ["root", "intermediate", "terminal"]
        },
        {
            "domain": "contract_breach",
            "shape_id": 1,
            "node_count": 5,
            "edge_type": "precedes",
            "structure": "linear",  # longer chain: A -> B -> C -> D -> E
            "node_roles": ["root", "intermediate", "intermediate", "intermediate", "terminal"]
        },
        # Domain B: tort_negligence
        {
            "domain": "tort_negligence",
            "shape_id": 0,
            "node_count": 3,
            "edge_type": "causes",
            "structure": "tree",    # root with two children: A -> B, A -> C
            "node_roles": ["root", "leaf", "leaf"]
        },
        {
            "domain": "tort_negligence",
            "shape_id": 1,
            "node_count": 4,
            "edge_type": "contributes",
            "structure": "diamond", # A -> B, A -> C, B -> D, C -> D
            "node_roles": ["root", "intermediate", "intermediate", "terminal"]
        },
        # Domain C: ip_infringement
        {
            "domain": "ip_infringement",
            "shape_id": 0,
            "node_count": 5,
            "edge_type": "enables",
            "structure": "star",    # center with 4 leaves: A -> B, A -> C, A -> D, A -> E
            "node_roles": ["root", "leaf", "leaf", "leaf", "leaf"]
        },
        {
            "domain": "ip_infringement",
            "shape_id": 1,
            "node_count": 6,
            "edge_type": "enables",
            "structure": "linear",  # long chain: A -> B -> C -> D -> E -> F
            "node_roles": ["root", "intermediate", "intermediate", "intermediate", "intermediate", "terminal"]
        }
    ]

    # Distribute cases evenly across the 6 shapes
    cases_per_shape = max(1, n_cases // len(shape_configs))
    
    for config in shape_configs:
        for _ in range(cases_per_shape):
            if case_counter >= n_cases:
                break
            case_id = f"case_{case_counter}"
            case_counter += 1
            shape_tag = f"{config['domain']}_shape_{config['shape_id']}"
            
            # Generate text based on domain and shape
            if config["domain"] == "contract_breach":
                contract_type = random.choice(["employment", "service", "sales"])
                consequence = random.choice(["damages", "financial loss", "breach penalty"])
                text = (
                    f"A plaintiff alleges that the defendant failed to perform under a {contract_type} "
                    f"contract. The dispute centers on whether the breach caused direct {consequence}. "
                    f"Documentation shows a clear sequence of events."
                )
            elif config["domain"] == "tort_negligence":
                duty_type = random.choice(["duty of care", "professional obligation", "safety standard"])
                harm = random.choice(["injury", "property damage", "emotional distress"])
                text = (
                    f"A plaintiff claims the defendant breached a {duty_type}, resulting in {harm}. "
                    f"Multiple witnesses observed the chain of events leading to the incident. "
                    f"The defendant argues the harm was unforeseeable."
                )
            else:  # ip_infringement
                ip_type = random.choice(["patent", "copyright", "trademark"])
                action = random.choice(["unauthorized use", "copying", "distribution"])
                text = (
                    f"A plaintiff alleges unauthorized {action} of their {ip_type}. "
                    f"The infringement enabled commercial gain for the defendant. "
                    f"Digital evidence traces the origin and propagation of the infringing material."
                )
            
            # Build graph based on structure
            nodes = []
            edges = []
            prefix = f"{case_id}_"
            
            # Create nodes
            for i in range(config["node_count"]):
                node_id = f"{prefix}node_{i}"
                role = config["node_roles"][i] if i < len(config["node_roles"]) else "intermediate"
                node_type = "event" if i == 0 else ("consequence" if i == config["node_count"] - 1 else "intermediate")
                nodes.append({"id": node_id, "type": node_type, "role": role})
            
            # Create edges based on structure
            if config["structure"] == "linear":
                for i in range(1, config["node_count"]):
                    edges.append({
                        "src": f"{prefix}node_{i-1}",
                        "dst": f"{prefix}node_{i}",
                        "type": config["edge_type"],
                        "scope": None
                    })
            elif config["structure"] == "tree":
                # Root (node_0) connects to two children (node_1, node_2)
                edges.append({
                    "src": f"{prefix}node_0",
                    "dst": f"{prefix}node_1",
                    "type": config["edge_type"],
                    "scope": None
                })
                edges.append({
                    "src": f"{prefix}node_0",
                    "dst": f"{prefix}node_2",
                    "type": config["edge_type"],
                    "scope": None
                })
            elif config["structure"] == "diamond":
                # Root (0) -> two intermediates (1,2) -> terminal (3)
                edges.append({
                    "src": f"{prefix}node_0",
                    "dst": f"{prefix}node_1",
                    "type": config["edge_type"],
                    "scope": None
                })
                edges.append({
                    "src": f"{prefix}node_0",
                    "dst": f"{prefix}node_2",
                    "type": config["edge_type"],
                    "scope": None
                })
                edges.append({
                    "src": f"{prefix}node_1",
                    "dst": f"{prefix}node_3",
                    "type": config["edge_type"],
                    "scope": None
                })
                edges.append({
                    "src": f"{prefix}node_2",
                    "dst": f"{prefix}node_3",
                    "type": config["edge_type"],
                    "scope": None
                })
            elif config["structure"] == "star":
                # Root (0) connects to all others (1,2,3,4)
                for i in range(1, config["node_count"]):
                    edges.append({
                        "src": f"{prefix}node_0",
                        "dst": f"{prefix}node_{i}",
                        "type": config["edge_type"],
                        "scope": None
                    })
            
            fragment = {"nodes": nodes, "edges": edges}
            cases.append({
                "case_id": case_id,
                "domain": config["domain"],
                "shape": shape_tag,
                "text": text,
                "fragment": fragment
            })
    
    # If we need more cases due to rounding, add one per shape until we reach n_cases
    while len(cases) < n_cases:
        config = shape_configs[len(cases) % len(shape_configs)]
        case_id = f"case_{case_counter}"
        case_counter += 1
        shape_tag = f"{config['domain']}_shape_{config['shape_id']}"
        
        if config["domain"] == "contract_breach":
            contract_type = random.choice(["employment", "service", "sales"])
            consequence = random.choice(["damages", "financial loss", "breach penalty"])
            text = (
                f"A plaintiff alleges that the defendant failed to perform under a {contract_type} "
                f"contract. The dispute centers on whether the breach caused direct {consequence}. "
                f"Documentation shows a clear sequence of events."
            )
        elif config["domain"] == "tort_negligence":
            duty_type = random.choice(["duty of care", "professional obligation", "safety standard"])
            harm = random.choice(["injury", "property damage", "emotional distress"])
            text = (
                f"A plaintiff claims the defendant breached a {duty_type}, resulting in {harm}. "
                f"Multiple witnesses observed the chain of events leading to the incident. "
                f"The defendant argues the harm was unforeseeable."
            )
        else:  # ip_infringement
            ip_type = random.choice(["patent", "copyright", "trademark"])
            action = random.choice(["unauthorized use", "copying", "distribution"])
            text = (
                f"A plaintiff alleges unauthorized {action} of their {ip_type}. "
                f"The infringement enabled commercial gain for the defendant. "
                f"Digital evidence traces the origin and propagation of the infringing material."
            )
        
        nodes = []
        edges = []
        prefix = f"{case_id}_"
        
        for i in range(config["node_count"]):
            node_id = f"{prefix}node_{i}"
            role = config["node_roles"][i] if i < len(config["node_roles"]) else "intermediate"
            node_type = "event" if i == 0 else ("consequence" if i == config["node_count"] - 1 else "intermediate")
            nodes.append({"id": node_id, "type": node_type, "role": role})
        
        if config["structure"] == "linear":
            for i in range(1, config["node_count"]):
                edges.append({
                    "src": f"{prefix}node_{i-1}",
                    "dst": f"{prefix}node_{i}",
                    "type": config["edge_type"],
                    "scope": None
                })
        elif config["structure"] == "tree":
            edges.append({
                "src": f"{prefix}node_0",
                "dst": f"{prefix}node_1",
                "type": config["edge_type"],
                "scope": None
            })
            edges.append({
                "src": f"{prefix}node_0",
                "dst": f"{prefix}node_2",
                "type": config["edge_type"],
                "scope": None
            })
        elif config["structure"] == "diamond":
            edges.append({
                "src": f"{prefix}node_0",
                "dst": f"{prefix}node_1",
                "type": config["edge_type"],
                "scope": None
            })
            edges.append({
                "src": f"{prefix}node_0",
                "dst": f"{prefix}node_2",
                "type": config["edge_type"],
                "scope": None
            })
            edges.append({
                "src": f"{prefix}node_1",
                "dst": f"{prefix}node_3",
                "type": config["edge_type"],
                "scope": None
            })
            edges.append({
                "src": f"{prefix}node_2",
                "dst": f"{prefix}node_3",
                "type": config["edge_type"],
                "scope": None
            })
        elif config["structure"] == "star":
            for i in range(1, config["node_count"]):
                edges.append({
                    "src": f"{prefix}node_0",
                    "dst": f"{prefix}node_{i}",
                    "type": config["edge_type"],
                    "scope": None
                })
        
        fragment = {"nodes": nodes, "edges": edges}
        cases.append({
            "case_id": case_id,
            "domain": config["domain"],
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
