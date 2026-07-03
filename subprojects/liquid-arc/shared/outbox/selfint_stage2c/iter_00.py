
import json, os, sys, typing, collections, math, itertools, heapq, functools, networkx
sys.path.insert(0, "/workspace/liquid-arc")

from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB

def db_factory():
    path = f"/tmp/stage2c_{os.getpid()}_{id(object()):x}.json"
    if os.path.exists(path):
        os.remove(path)
    return KnowledgeGraphDB(path)

IMPL_CODE = r"""
def retrieve_text(self, node_ids: Iterable[str], max_segments: int = 10) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for nid in node_ids:
        segments.extend(self.text_segments.get(nid, []))
    segments.sort(key=lambda s: -s.get("timestamp", 0))
    seen: Set[tuple] = set()
    unique: List[Dict[str, Any]] = []
    for s in segments:
        t = s.get("text", "")
        chunk_id = s.get("chunk_id")
        doc_metadata = frozenset(s.get("doc_metadata", {}).items())
        key = (t, chunk_id, doc_metadata)
        if t and key not in seen:
            seen.add(key)
            unique.append(s)
        if len(unique) >= max_segments:
            break
    return unique
"""
TEST_CODE = r"""
def test_improvement(db_factory):
    db = db_factory()
    # Create two segments with identical text but different chunk_id and doc_metadata
    segment1 = {
        "text": "The cat sat on the mat",
        "chunk_id": "chunk_1",
        "doc_metadata": {"source": "book_a", "page": 1},
        "timestamp": 100
    }
    segment2 = {
        "text": "The cat sat on the mat",
        "chunk_id": "chunk_2",
        "doc_metadata": {"source": "book_b", "page": 2},
        "timestamp": 90
    }
    
    # Add both segments to the db under different node_ids
    db.text_segments["node1"] = [segment1]
    db.text_segments["node2"] = [segment2]
    
    # Retrieve segments - original impl would deduplicate (same text), candidate keeps both
    result = db.retrieve_text(["node1", "node2"], max_segments=10)
    
    # Candidate impl should return both segments (different context), original returns only one
    assert len(result) == 2, "Candidate implementation must preserve both segments with identical text but different metadata"
    assert result[0]["text"] == "The cat sat on the mat", "First segment text should match"
    assert result[1]["text"] == "The cat sat on the mat", "Second segment text should match"
    assert result[0]["chunk_id"] != result[1]["chunk_id"], "Chunk IDs must differ"
    assert result[0]["doc_metadata"] != result[1]["doc_metadata"], "Document metadata must differ"

"""
TARGET_METHOD = "retrieve_text"

result = {
    "shape_ok": False,
    "impl_exec_ok": False,
    "regression_pass": False,
    "new_test_passes_on_improved": False,
    "new_test_fails_on_original": False,
    "errors": [],
}

_typing_names = ("List","Dict","Set","Tuple","Iterable","Optional","Any",
                  "Callable","Sequence","Union","Mapping")
_base_ns = {"nx": networkx, "networkx": networkx,
            "typing": typing, "collections": collections,
            "math": math, "itertools": itertools,
            "heapq": heapq, "functools": functools}
for _name in _typing_names:
    _base_ns[_name] = getattr(typing, _name)

if TARGET_METHOD and IMPL_CODE.strip() and TEST_CODE.strip():
    result["shape_ok"] = True

impl_ns = dict(_base_ns); impl_ns["__name__"] = "impl"
try:
    exec(IMPL_CODE, impl_ns)
    assert TARGET_METHOD in impl_ns, f"impl did not define {TARGET_METHOD}"
    result["impl_exec_ok"] = True
except Exception as e:
    result["errors"].append(f"impl_exec: {e}")

if result["impl_exec_ok"]:
    test_ns = dict(_base_ns); test_ns["__name__"] = "test_pre"
    try:
        exec(TEST_CODE, test_ns)
    except Exception as e:
        result["errors"].append(f"test_exec: {e}")
    fn = test_ns.get("test_improvement")
    if fn is None:
        result["errors"].append("test did not define test_improvement")
    else:
        failed = False
        try:
            fn(db_factory)
        except AssertionError:
            failed = True
        except Exception:
            failed = True
        result["new_test_fails_on_original"] = failed
        if not failed:
            result["errors"].append("test passed on unpatched class — trivial")

if result["impl_exec_ok"]:
    original_method = getattr(KnowledgeGraphDB, TARGET_METHOD, None)
    setattr(KnowledgeGraphDB, TARGET_METHOD, impl_ns[TARGET_METHOD])
    test_ns2 = dict(_base_ns); test_ns2["__name__"] = "test_post"
    try:
        exec(TEST_CODE, test_ns2)
        fn = test_ns2.get("test_improvement")
        if fn is not None:
            fn(db_factory)
            result["new_test_passes_on_improved"] = True
    except Exception as e:
        result["errors"].append(f"improved_test: {type(e).__name__}: {e}")

    try:
        db = db_factory()
        frag = {
            "nodes": [
                {"id": "root", "type": "event", "role": "root"},
                {"id": "mid1", "type": "state", "role": "intermediate"},
                {"id": "mid2", "type": "state", "role": "intermediate"},
                {"id": "leaf", "type": "consequence", "role": "terminal"},
            ],
            "edges": [
                {"src": "root", "dst": "mid1", "type": "causes", "scope": None},
                {"src": "mid1", "dst": "mid2", "type": "causes", "scope": "prod"},
                {"src": "mid2", "dst": "leaf", "type": "causes", "scope": None},
            ],
        }
        rep = db.add_fragment(frag, source_text="hello", autosave=False)
        assert rep["added_nodes"] == 4 and rep["added_edges"] == 3
        chain = db.trace_causal_chain("leaf", max_hops=10)
        assert chain["root"] == "root" and chain["hops"] == 3
        reach = db.get_reachable("root", max_hops=5)
        assert {"mid1", "mid2", "leaf"} <= reach
        sub = db.extract_subgraph(["root", "mid1"], max_nodes=10)
        assert len(sub["nodes"]) == 2
        txt = db.retrieve_text(["root"], max_segments=2)
        assert len(txt) >= 1
        stats = db.stats()
        assert stats["n_nodes"] == 4
        filt = db.scope_filter("prod")
        assert filt.has_edge("mid1", "mid2")
        neigh = db.get_neighbors(["root"], hops=2)
        assert "mid1" in neigh and "mid2" in neigh
        _ = db.find_communities(min_size=2)
        db.clear()
        result["regression_pass"] = True
    except Exception as e:
        result["errors"].append(f"regression: {type(e).__name__}: {e}")

    if original_method is not None:
        setattr(KnowledgeGraphDB, TARGET_METHOD, original_method)
    else:
        try: delattr(KnowledgeGraphDB, TARGET_METHOD)
        except Exception: pass

reward = (0.15 * (1 if result["shape_ok"] else 0)
          + 0.15 * (1 if result["impl_exec_ok"] else 0)
          + 0.25 * (1 if result["regression_pass"] else 0)
          + 0.30 * (1 if result["new_test_passes_on_improved"] else 0)
          + 0.15 * (1 if result["new_test_fails_on_original"] else 0))
result["reward"] = reward
print("RESULT_JSON: " + json.dumps(result, default=str))
