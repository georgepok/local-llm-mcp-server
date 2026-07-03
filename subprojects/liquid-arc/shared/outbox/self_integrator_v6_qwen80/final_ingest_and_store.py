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
