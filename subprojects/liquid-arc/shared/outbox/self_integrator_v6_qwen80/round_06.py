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
    shapes = [0, 1]
    cases: List[Dict[str, Any]] = []
    case_counter = 0

    for domain in domains:
        for shape_idx in shapes:
            for _ in range(max(1, n_cases // (len(domains) * len(shapes)))):
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
