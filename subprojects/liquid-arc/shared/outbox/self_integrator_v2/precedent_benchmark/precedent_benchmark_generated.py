import os
import json
import random
import math
import sys
from typing import List, Dict, Any

import networkx as nx

from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB
from liquid_arc.graph_rag.decoupled.ode_engine import SubgraphODEEngine
from liquid_arc.navigator_patterns import PatternLibrary
from liquid_arc.graph_rag.vector_db import VectorDB

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def generate_case(domain: str, shape_idx: int, case_id: int) -> Dict[str, Any]:
    """Generate a deterministic case dict with text and fragment."""
    random.seed(case_id)
    # Simple templates for each shape
    if domain == "contract_breach":
        if shape_idx == 0:
            # breach → damages → remedy_sought → settlement
            nodes = [
                {"id": "n0", "type": "event", "role": "breach"},
                {"id": "n1", "type": "event", "role": "damages"},
                {"id": "n2", "type": "event", "role": "remedy_sought"},
                {"id": "n3", "type": "event", "role": "settlement"},
            ]
            edges = [
                {"src": "n0", "dst": "n1", "type": "causes", "scope": None},
                {"src": "n1", "dst": "n2", "type": "causes", "scope": None},
                {"src": "n2", "dst": "n3", "type": "causes", "scope": None},
            ]
            text = (
                f"In this contract dispute, the plaintiff alleges a breach of "
                f"the agreement, which caused damages to the plaintiff. "
                f"The plaintiff seeks a specific remedy, and the parties have "
                f"reached a settlement."
            )
        else:
            # obligation_failure → reliance → loss → court_filing
            nodes = [
                {"id": "n0", "type": "event", "role": "obligation_failure"},
                {"id": "n1", "type": "event", "role": "reliance"},
                {"id": "n2", "type": "event", "role": "loss"},
                {"id": "n3", "type": "event", "role": "court_filing"},
            ]
            edges = [
                {"src": "n0", "dst": "n1", "type": "causes", "scope": None},
                {"src": "n1", "dst": "n2", "type": "causes", "scope": None},
                {"src": "n2", "dst": "n3", "type": "causes", "scope": None},
            ]
            text = (
                f"The defendant failed to fulfill an contractual obligation, "
                f"which the plaintiff relied upon, resulting in a loss. "
                f"The plaintiff filed a lawsuit in court."
            )
    elif domain == "tort_negligence":
        if shape_idx == 0:
            # negligent act → duty → breach → injury
            nodes = [
                {"id": "n0", "type": "event", "role": "negligent_act"},
                {"id": "n1", "type": "event", "role": "duty"},
                {"id": "n2", "type": "event", "role": "breach"},
                {"id": "n3", "type": "event", "role": "injury"},
            ]
            edges = [
                {"src": "n0", "dst": "n1", "type": "causes", "scope": None},
                {"src": "n1", "dst": "n2", "type": "causes", "scope": None},
                {"src": "n2", "dst": "n3", "type": "causes", "scope": None},
            ]
            text = (
                f"The defendant performed a negligent act that violated a duty "
                f"owed to the plaintiff, resulting in a breach. The breach "
                f"directly caused injury to the plaintiff."
            )
        else:
            # causation → damages → settlement → appeal
            nodes = [
                {"id": "n0", "type": "event", "role": "causation"},
                {"id": "n1", "type": "event", "role": "damages"},
                {"id": "n2", "type": "event", "role": "settlement"},
                {"id": "n3", "type": "event", "role": "appeal"},
            ]
            edges = [
                {"src": "n0", "dst": "n1", "type": "causes", "scope": None},
                {"src": "n1", "dst": "n2", "type": "causes", "scope": None},
                {"src": "n2", "dst": "n3", "type": "causes", "scope": None},
            ]
            text = (
                f"The plaintiff proved causation and resulting damages, leading "
                f"to a settlement. The defendant appealed the settlement decision."
            )
    elif domain == "ip_infringement":
        if shape_idx == 0:
            # infringement → copying → distribution → injunction
            nodes = [
                {"id": "n0", "type": "event", "role": "infringement"},
                {"id": "n1", "type": "event", "role": "copying"},
                {"id": "n2", "type": "event", "role": "distribution"},
                {"id": "n3", "type": "event", "role": "injunction"},
            ]
            edges = [
                {"src": "n0", "dst": "n1", "type": "causes", "scope": None},
                {"src": "n1", "dst": "n2", "type": "causes", "scope": None},
                {"src": "n2", "dst": "n3", "type": "causes", "scope": None},
            ]
            text = (
                f"The plaintiff alleges that the defendant infringed a patent by "
                f"copying the invention, which was then distributed. The court "
                f"granted an injunction."
            )
        else:
            # licensing → royalty → breach → litigation
            nodes = [
                {"id": "n0", "type": "event", "role": "licensing"},
                {"id": "n1", "type": "event", "role": "royalty"},
                {"id": "n2", "type": "event", "role": "breach"},
                {"id": "n3", "type": "event", "role": "litigation"},
            ]
            edges = [
                {"src": "n0", "dst": "n1", "type": "causes", "scope": None},
                {"src": "n1", "dst": "n2", "type": "causes", "scope": None},
                {"src": "n2", "dst": "n3", "type": "causes", "scope": None},
            ]
            text = (
                f"The licensor granted a license that included royalty payments, "
                f"but the licensee breached the agreement by failing to pay. "
                f"The licensor initiated litigation."
            )
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    fragment = {
        "nodes": nodes,
        "edges": edges,
    }
    return {"text": text, "fragment": fragment, "case_id": case_id, "domain": domain, "shape_idx": shape_idx}


def ingest_fragment(db: KnowledgeGraphDB, fragment: Dict, source_text: str, case_id: int) -> None:
    """Ingest a fragment into the KnowledgeGraphDB."""
    db.add_fragment(
        fragment=fragment,
        source_text=source_text,
        chunk_id=case_id,
        doc_metadata={"case_id": case_id, "source_text": source_text},
        autosave=True,
    )


def compute_and_store_signature(
    db: KnowledgeGraphDB,
    case_data: Dict,
    pattern_lib: PatternLibrary,
    scratch_db_path: str,
    case_id: int,
) -> Dict[str, Any]:
    """Compute signature for a case and store it in the pattern library."""
    # Load scratch DB (or create fresh)
    scratch_db = KnowledgeGraphDB(db_path=scratch_db_path)
    # Ingest fragment
    ingest_fragment(scratch_db, case_data["fragment"], case_data["text"], case_id)

    # Load subgraph (2-hop, max 30 nodes) around a central node
    # For simplicity, we take the whole graph (it is tiny) as the subgraph
    subgraph = db.extract_subgraph(list(scratch_db.graph.nodes()), max_nodes=30)

    # Compute signature
    engine = SubgraphODEEngine(checkpoint_path=os.getenv("CHECKPOINT", "/workspace/liquid-arc/output_graph_engine_final/checkpoints/step_500.pt"), device="cpu")
    signature = engine.compute_signature(subgraph)

    # Store in pattern library
    pattern_lib.store(signature, {"case_id": case_id, "domain": case_data["domain"], "shape_idx": case_data["shape_idx"]})

    # Retrieve stored signature for later nearest‑neighbor lookup
    stored_sig = pattern_lib.find_nearest(signature, threshold=0.5)
    return {
        "signature": signature,
        "stored_sig": stored_sig,
        "case_id": case_id,
        "domain": case_data["domain"],
        "shape_idx": case_data["shape_idx"],
    }


def evaluate_test_cases(
    test_cases: List[Dict],
    db_path: str,
    pattern_lib_path: str,
) -> Dict[str, Any]:
    """Run evaluation on the 10 novel test cases."""
    # Load persistent pattern library
    pattern_lib = PatternLibrary(library_path=pattern_lib_path)

    # Scratch DB path for test case ingestion
    scratch_db_path = os.path.join(os.getenv("OUT_DIR", "/tmp/out"), "scratch_db.json")

    results = []
    correct_shape_matches = 0

    for idx, case in enumerate(test_cases, start=1):
        try:
            # Compute signature for this test case using a fresh scratch DB
            scratch_db = KnowledgeGraphDB(db_path=scratch_db_path)
            ingest_fragment(scratch_db, case["fragment"], case["text"], case["case_id"])

            # Extract subgraph (2‑hop, cap 30 nodes)
            subgraph = scratch_db.extract_subgraph(list(scratch_db.graph.nodes()), max_nodes=30)

            # Compute signature
            engine = SubgraphODEEngine(checkpoint_path=os.getenv("CHECKPOINT", "/workspace/liquid-arc/output_graph_engine_final/checkpoints/step_500.pt"), device="cpu")
            signature = engine.compute_signature(subgraph)

            # Find nearest stored pattern
            nearest = pattern_lib.find_nearest(signature, threshold=0.5)

            if nearest is None:
                matched_case_id = None
                matched_shape = None
                cosine = 0.0
            else:
                # nearest is a dict with 'metadata' containing 'case_id' and 'shape_idx'
                matched_case_id = nearest.get("metadata", {}).get("case_id")
                matched_shape = nearest.get("metadata", {}).get("shape_idx")
                # cosine similarity between current signature and stored signature
                stored_sig = nearest.get("signature", [])
                cosine = cosine_similarity(signature, stored_sig)

            expected_shape = case["expected_shape"]
            matched_ok = (matched_shape == expected_shape)
            if matched_ok:
                correct_shape_matches += 1

            results.append({
                "expected_shape": expected_shape,
                "matched_case_id": matched_case_id,
                "cosine": cosine,
                "matched_shape": matched_shape,
            })
        except Exception as e:
            print(f"[ERROR] Failed to process test case {idx}: {e}", file=sys.stderr)
            results.append({
                "expected_shape": case.get("expected_shape"),
                "matched_case_id": None,
                "cosine": 0.0,
                "matched_shape": None,
            })
            continue

    accuracy = correct_shape_matches / len(test_cases) if test_cases else 0.0
    return {
        "n_cases_ingested": len(test_cases),
        "n_patterns_stored": len(pattern_lib.library) if hasattr(pattern_lib, "library") else 0,
        "n_test_cases": len(test_cases),
        "n_correct_shape_match": correct_shape_matches,
        "accuracy": accuracy,
        "per_test_case": results,
    }


# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
def main() -> None:
    random.seed(42)

    # ------------------------------------------------------------------
    # 1. Setup environment & paths
    # ------------------------------------------------------------------
    out_dir = os.getenv("OUT_DIR", "/tmp/out")
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = os.getenv("CHECKPOINT", "/workspace/liquid-arc/output_graph_engine_final/checkpoints/step_500.pt")
    vllm_url = os.getenv("VLLM_URL")  # not used directly in this script

    # Paths for persistent DB and pattern library
    db_path = os.path.join(out_dir, "precedent_db.json")
    pattern_lib_path = os.path.join(out_dir, "pattern_library")

    # ------------------------------------------------------------------
    # 2. Generate training corpus (100 cases)
    # ------------------------------------------------------------------
    domains = ["contract_breach", "tort_negligence", "ip_infringement"]
    shape_counts_per_domain = {
        "contract_breach": 2,
        "tort_negligence": 2,
        "ip_infringement": 2,
    }
    train_cases = []
    case_id_counter = 0

    for domain in domains:
        shapes = shape_counts_per_domain[domain]
        # Distribute roughly equal counts per shape (15-17 each)
        per_shape_target = 100 // (len(domains) * shapes)
        for shape_idx in range(shapes):
            count = per_shape_target
            # Ensure at least 1 case per shape
            if count == 0:
                count = 1
            for _ in range(count):
                case = generate_case(domain, shape_idx, case_id_counter)
                train_cases.append(case)
                case_id_counter += 1

    # ------------------------------------------------------------------
    # 3. Ingest training cases into KnowledgeGraphDB
    # ------------------------------------------------------------------
    try:
        db = KnowledgeGraphDB(db_path=db_path)
    except Exception as e:
        print(f"[ERROR] Could not initialize KnowledgeGraphDB: {e}", file=sys.stderr)
        sys.exit(1)

    for case in train_cases:
        try:
            ingest_fragment(db, case["fragment"], case["text"], case["case_id"])
        except Exception as e:
            print(f"[ERROR] Failed to ingest case {case['case_id']}: {e}", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Build PatternLibrary of signatures from training cases
    # ------------------------------------------------------------------
    try:
        pattern_lib = PatternLibrary(library_path=pattern_lib_path)
    except Exception as e:
        print(f"[ERROR] Could not initialize PatternLibrary: {e}", file=sys.stderr)
        sys.exit(1)

    # Compute signatures for all training cases and store them
    for case in train_cases:
        try:
            # Use a scratch DB for subgraph extraction (same logic as evaluation)
            scratch_db_path = os.path.join(out_dir, f"scratch_{case['case_id']}.json")
            result = compute_and_store_signature(db, case, pattern_lib, scratch_db_path, case["case_id"])
            # No further action needed; signatures are stored inside pattern_lib
        except Exception as e:
            print(f"[ERROR] Signature computation failed for case {case['case_id']}: {e}", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Generate 10 novel test cases (3‑4 per domain)
    # ------------------------------------------------------------------
    test_cases = []
    for domain in domains:
        # Pick 3 or 4 shapes (some domains may have only 2 shapes, so we repeat)
        shape_indices = list(range(shape_counts_per_domain[domain]))
        # Duplicate to reach 3‑4 cases per domain
        while len(test_cases) < 10:
            for d in domains:
                if len(test_cases) >= 10:
                    break
                # Randomly pick a shape index for this domain
                shape_idx = random.choice(shape_indices)
                case_id = 1000 + len(test_cases)  # deterministic id
                case_data = generate_case(domain, shape_idx, case_id)
                # Determine expected_shape based on shape_idx
                expected_shape = "shape_0" if shape_idx == 0 else "shape_1"
                case_data["expected_shape"] = expected_shape
                test_cases.append(case_data)
                if len(test_cases) >= 10:
                    break

    # ------------------------------------------------------------------
    # 6. Evaluate on test cases
    # ------------------------------------------------------------------
    benchmark_result = evaluate_test_cases(test_cases, db_path, pattern_lib_path)

    # ------------------------------------------------------------------
    # 7. Output RESULT_JSON line
    # ------------------------------------------------------------------
    result_json = {
        "n_cases_ingested": benchmark_result["n_cases_ingested"],
        "n_patterns_stored": benchmark_result["n_patterns_stored"],
        "n_test_cases": benchmark_result["n_test_cases"],
        "n_correct_shape_match": benchmark_result["n_correct_shape_match"],
        "accuracy": benchmark_result["accuracy"],
        "per_test_case": benchmark_result["per_test_case"],
    }

    print(f"RESULT_JSON: {json.dumps(result_json)}")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)
