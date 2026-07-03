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
                if shape_idx == 0:  # linear chain of 3 nodes
                    node_ids = [f"{prefix}event_0", f"{prefix}cause_1", f"{prefix}consequence_2"]
                    nodes = [
                        {"id": node_ids[0], "type": "event", "role": "root"},
                        {"id": node_ids[1], "type": "cause", "role": "intermediate"},
                        {"id": node_ids[2], "type": "consequence", "role": "terminal"}
                    ]
                    edges = [
                        {"src": node_ids[0], "dst": node_ids[1], "type": "causes", "scope": None},
                        {"src": node_ids[1], "dst": node_ids[2], "type": "causes", "scope": None}
                    ]
                elif shape_idx == 1:  # linear chain of 5 nodes
                    node_ids = [f"{prefix}event_0", f"{prefix}cause_1", f"{prefix}state_2", f"{prefix}consequence_3", f"{prefix}terminal_4"]
                    nodes = [
                        {"id": node_ids[0], "type": "event", "role": "root"},
                        {"id": node_ids[1], "type": "cause", "role": "intermediate"},
                        {"id": node_ids[2], "type": "state", "role": "intermediate"},
                        {"id": node_ids[3], "type": "consequence", "role": "intermediate"},
                        {"id": node_ids[4], "type": "terminal", "role": "terminal"}
                    ]
                    edges = [
                        {"src": node_ids[0], "dst": node_ids[1], "type": "causes", "scope": None},
                        {"src": node_ids[1], "dst": node_ids[2], "type": "causes", "scope": None},
                        {"src": node_ids[2], "dst": node_ids[3], "type": "causes", "scope": None},
                        {"src": node_ids[3], "dst": node_ids[4], "type": "causes", "scope": None}
                    ]
                elif shape_idx == 2:  # tree with 1 root and 2 children (3 nodes total)
                    root_id = f"{prefix}event_0"
                    child1_id = f"{prefix}cause_1"
                    child2_id = f"{prefix}consequence_2"
                    nodes = [
                        {"id": root_id, "type": "event", "role": "root"},
                        {"id": child1_id, "type": "cause", "role": "intermediate"},
                        {"id": child2_id, "type": "consequence", "role": "terminal"}
                    ]
                    edges = [
                        {"src": root_id, "dst": child1_id, "type": "causes", "scope": None},
                        {"src": root_id, "dst": child2_id, "type": "causes", "scope": None}
                    ]
                elif shape_idx == 3:  # diamond with 4 nodes
                    root_id = f"{prefix}event_0"
                    mid1_id = f"{prefix}cause_1"
                    mid2_id = f"{prefix}state_2"
                    term_id = f"{prefix}consequence_3"
                    nodes = [
                        {"id": root_id, "type": "event", "role": "root"},
                        {"id": mid1_id, "type": "cause", "role": "intermediate"},
                        {"id": mid2_id, "type": "state", "role": "intermediate"},
                        {"id": term_id, "type": "consequence", "role": "terminal"}
                    ]
                    edges = [
                        {"src": root_id, "dst": mid1_id, "type": "causes", "scope": None},
                        {"src": root_id, "dst": mid2_id, "type": "causes", "scope": None},
                        {"src": mid1_id, "dst": term_id, "type": "causes", "scope": None},
                        {"src": mid2_id, "dst": term_id, "type": "causes", "scope": None}
                    ]
                elif shape_idx == 4:  # star with 1 root and 4 leaves (5 nodes total)
                    root_id = f"{prefix}event_0"
                    leaf_ids = [f"{prefix}{t}_{i}" for i, t in enumerate(["cause", "state", "event", "terminal"], start=1)]
                    nodes = [{"id": root_id, "type": "event", "role": "root"}] + [
                        {"id": leaf, "type": t, "role": "terminal"} for leaf, t in zip(leaf_ids, ["cause", "state", "event", "terminal"])
                    ]
                    edges = [
                        {"src": root_id, "dst": leaf, "type": "causes", "scope": None} for leaf in leaf_ids
                    ]
                elif shape_idx == 5:  # linear chain of 6 nodes
                    node_ids = [f"{prefix}{t}_{i}" for i, t in enumerate(["event", "cause", "state", "event", "state", "terminal"], start=0)]
                    nodes = [
                        {"id": node_ids[0], "type": "event", "role": "root"},
                        {"id": node_ids[1], "type": "cause", "role": "intermediate"},
                        {"id": node_ids[2], "type": "state", "role": "intermediate"},
                        {"id": node_ids[3], "type": "event", "role": "intermediate"},
                        {"id": node_ids[4], "type": "state", "role": "intermediate"},
                        {"id": node_ids[5], "type": "terminal", "role": "terminal"}
                    ]
                    edges = [
                        {"src": node_ids[0], "dst": node_ids[1], "type": "causes", "scope": None},
                        {"src": node_ids[1], "dst": node_ids[2], "type": "causes", "scope": None},
                        {"src": node_ids[2], "dst": node_ids[3], "type": "causes", "scope": None},
                        {"src": node_ids[3], "dst": node_ids[4], "type": "causes", "scope": None},
                        {"src": node_ids[4], "dst": node_ids[5], "type": "causes", "scope": None}
                    ]
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

    for case in cases:
        case_id = case.get("case_id")
        fragment = case.get("fragment")
        if not isinstance(fragment, dict) or "nodes" not in fragment or "edges" not in fragment:
            errors.append(f"{case_id}: 'list' object has no attribute 'keys'")
            continue
        try:
            db.add_fragment(fragment, source_text=case_id, chunk_id=case_id, autosave=True)
            n_ingested += 1
        except Exception as e:
            errors.append(f"{case_id}: add_fragment failed - {e}")
            continue

        try:
            node_ids = [n["id"] for n in fragment["nodes"]]
            neighbors = db.get_neighbors(node_ids, hops=2, direction="both")
            subgraph = db.extract_subgraph(list(neighbors) + node_ids, max_nodes=30)
            if len(subgraph["nodes"]) >= 2:
                signature = SubgraphODEEngine(checkpoint, device="cpu").compute_signature(subgraph)
                lib.store(signature, {"label": case_id})
                n_signatures += 1
        except Exception as e:
            errors.append(f"{case_id}: signature computation failed - {e}")

    db.clear()
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
            db.add_fragment(fragment, source_text=case_id, chunk_id=case_id, autosave=True)
        except Exception as e:
            per_case.append({
                "case_id": case_id,
                "expected_shape": expected_shape,
                "matched_label": None,
                "cosine": None,
                "matched_shape": None,
                "error": f"add_fragment failed - {e}"
            })
            continue

        try:
            node_ids = [n["id"] for n in fragment["nodes"]]
            neighbors = db.get_neighbors(node_ids, hops=2, direction="both")
            subgraph = db.extract_subgraph(list(neighbors) + node_ids, max_nodes=30)
            if len(subgraph["nodes"]) < 2:
                raise ValueError("Extracted subgraph has fewer than 2 nodes")
            signature = SubgraphODEEngine(checkpoint, device="cpu").compute_signature(subgraph)
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
            db.clear()

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
