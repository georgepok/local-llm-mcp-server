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
