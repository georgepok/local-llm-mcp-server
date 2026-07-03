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
    engine = SubgraphODEEngine(checkpoint, device="cpu")

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
            
            if len(subgraph["nodes"]) >= 2:
                sig = engine.compute_signature(subgraph)
                match = lib.find_nearest(sig, threshold=0.5)
                
                if match:
                    matched_label = match["label"]
                    cosine = match["similarity"]
                    matched_shape = train_shape_map.get(matched_label)
                else:
                    matched_label = None
                    cosine = None
                    matched_shape = None

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
            else:
                per_case.append({
                    "case_id": case_id,
                    "expected_shape": expected_shape,
                    "matched_label": None,
                    "cosine": None,
                    "matched_shape": None,
                    "error": "subgraph too small"
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

    accuracy = correct / n_cases if n_cases else 0.0
    return {"n_cases": n_cases, "accuracy": accuracy, "per_case": per_case}
