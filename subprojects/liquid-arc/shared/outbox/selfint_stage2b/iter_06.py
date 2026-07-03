
import json, os, sys, traceback
sys.path.insert(0, "/workspace/liquid-arc")

from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB

def db_factory():
    path = f"/tmp/stage2_{os.getpid()}_{id(object()):x}.json"
    if os.path.exists(path):
        os.remove(path)
    return KnowledgeGraphDB(path)

# ---- Ingest candidate impl + test into namespaces ----
IMPL_CODE = r"""
def get_neighbors(self, node_ids: Iterable[str], hops: int = 2, direction: str = "both") -> Set[str]:
    seeds = {n for n in node_ids if n in self.G}
    if not seeds:
        return set()
    hood: Set[str] = set(seeds)
    frontier: Set[str] = set(seeds)
    for _ in range(hops):
        nxt: Set[str] = set()
        for node in frontier:
            if direction in ("both", "forward"):
                nxt.update(self.G.successors(node))
            if direction in ("both", "backward"):
                nxt.update(self.G.predecessors(node))
        frontier = nxt - hood
        hood.update(frontier)
        if not frontier:
            break
    return hood
"""
TEST_CODE = r"""
def test_improvement(db_factory):
    db = db_factory()
    # Add nodes A, B, C with edges A->B, B->C
    db.add_fragment({"nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "edges": [{"src": "A", "dst": "B", "type": "causes"}, {"src": "B", "dst": "C", "type": "causes"}]})
    # Call get_neighbors with duplicate node_ids
    result = db.get_neighbors(["A", "A", "B"], hops=2, direction="forward")
    # Should return {B, C} — duplicates in input must not affect output
    assert result == {"B", "C"}, f"Expected {{'B', 'C'}}, got {result}"
"""
TARGET_METHOD = "get_neighbors"

result = {
    "shape_ok": False,
    "impl_exec_ok": False,
    "regression_pass": False,
    "new_test_passes_on_improved": False,
    "new_test_fails_on_original": False,
    "errors": [],
}

# ---- shape_ok: required fields were parsed already by harness driver ----
if TARGET_METHOD and IMPL_CODE.strip() and TEST_CODE.strip():
    result["shape_ok"] = True

# ---- impl_exec_ok: exec impl into a scratch namespace preloaded with
# common imports so Qwen can type-hint with typing names freely ----
import typing, networkx, collections, math, itertools, heapq, functools
impl_ns = {"__name__": "impl",
           "nx": networkx,
           "networkx": networkx,
           "typing": typing,
           "collections": collections,
           "math": math,
           "itertools": itertools,
           "heapq": heapq,
           "functools": functools}
# Expose common typing names directly
for _name in ("List","Dict","Set","Tuple","Iterable","Optional","Any",
              "Callable","Sequence","Union","Mapping"):
    impl_ns[_name] = getattr(typing, _name)
try:
    exec(IMPL_CODE, impl_ns)
    assert TARGET_METHOD in impl_ns, f"impl did not define {TARGET_METHOD}"
    result["impl_exec_ok"] = True
except Exception as e:
    result["errors"].append(f"impl_exec: {e}")

# ---- test_fails_on_original: run test BEFORE monkey-patching ----
if result["impl_exec_ok"]:
    test_ns = dict(impl_ns)
    test_ns["__name__"] = "test"
    try:
        exec(TEST_CODE, test_ns)
    except Exception as e:
        result["errors"].append(f"test_exec: {e}")
    else:
        fn = test_ns.get("test_improvement")
        if fn is None:
            result["errors"].append("test did not define test_improvement")
        else:
            failed = False
            try:
                fn(db_factory)
            except AssertionError as e:
                failed = True
            except Exception as e:
                # Unexpected error ≠ clean AssertionError; count as failed test
                failed = True
            result["new_test_fails_on_original"] = failed
            if not failed:
                result["errors"].append(
                    "test passed on unpatched class — improvement is trivial")

# ---- Monkey-patch: install impl as method on KnowledgeGraphDB ----
if result["impl_exec_ok"]:
    original_method = getattr(KnowledgeGraphDB, TARGET_METHOD, None)
    setattr(KnowledgeGraphDB, TARGET_METHOD, impl_ns[TARGET_METHOD])

    # ---- new_test_passes_on_improved ----
    test_ns2 = dict(impl_ns)
    test_ns2["__name__"] = "test2"
    try:
        exec(TEST_CODE, test_ns2)
        fn = test_ns2.get("test_improvement")
        if fn is not None:
            fn(db_factory)
            result["new_test_passes_on_improved"] = True
    except Exception as e:
        result["errors"].append(f"improved_test: {e}")

    # ---- regression_pass: exercise a battery of existing methods ----
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
        # scope_filter
        filt = db.scope_filter("prod")
        assert filt.has_edge("mid1", "mid2")
        # get_neighbors
        neigh = db.get_neighbors(["root"], hops=2)
        assert "mid1" in neigh and "mid2" in neigh
        # find_communities shouldn't crash
        _ = db.find_communities(min_size=2)
        db.clear()
        result["regression_pass"] = True
    except Exception as e:
        result["errors"].append(f"regression: {type(e).__name__}: {e}")

    # Restore original method (cleanliness)
    if original_method is not None:
        setattr(KnowledgeGraphDB, TARGET_METHOD, original_method)
    else:
        try:
            delattr(KnowledgeGraphDB, TARGET_METHOD)
        except Exception:
            pass

# ---- reward ----
reward = (0.15 * (1 if result["shape_ok"] else 0)
          + 0.15 * (1 if result["impl_exec_ok"] else 0)
          + 0.25 * (1 if result["regression_pass"] else 0)
          + 0.30 * (1 if result["new_test_passes_on_improved"] else 0)
          + 0.15 * (1 if result["new_test_fails_on_original"] else 0))
result["reward"] = reward
print("RESULT_JSON: " + json.dumps(result, default=str))
