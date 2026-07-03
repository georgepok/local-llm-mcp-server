"""FastMCP server exposing the LiquidARC graph reasoning engine to Claude.

Three tools per the spec (GRAPH_REASONING_ENGINE_SPEC.md lines 384-430):
  - analyze_graph(graph_json, query_json)
  - compare_graphs(graph_a_json, graph_b_json)
  - get_graph_diagnostics(graph_json)

Usage:
    python -m liquid_arc.mcp_graph_serve \
      --checkpoint /workspace/liquid-arc/output_graph_engine_final/checkpoints/step_500.pt \
      --port 8421
"""

import argparse
import json
import signal
import sys
import time
from typing import Optional

from fastmcp import FastMCP

# Apply the same session auto-init patch as mcp_serve.py
try:
    from mcp.server.session import ServerSession, InitializationState
    _orig_received_request = ServerSession._received_request
    async def _patched_received_request(self, request):
        if self._initialization_state != InitializationState.Initialized:
            self._initialization_state = InitializationState.Initialized
        return await _orig_received_request(self, request)
    ServerSession._received_request = _patched_received_request

    _orig_received_notification = ServerSession._received_notification
    async def _patched_received_notification(self, notification):
        if self._initialization_state != InitializationState.Initialized:
            self._initialization_state = InitializationState.Initialized
        return await _orig_received_notification(self, notification)
    ServerSession._received_notification = _patched_received_notification
except ImportError:
    pass

from .graph_engine_inference import GraphEngine
from .navigator import GeometricNavigator
from .navigator_extract import LLMExtractor
from .navigator_patterns import PatternLibrary
from .navigator_state import GeometricState


mcp = FastMCP("LiquidARC Graph Engine")

_engine: Optional[GraphEngine] = None
_navigator: Optional[GeometricNavigator] = None


@mcp.tool()
def analyze_graph(graph_json: str, query_json: str) -> str:
    """Analyze a graph with a structured query.

    Node/edge TYPES and ROLES may be given as either integers or friendly
    string names; strings are mapped to the learned vocabulary.
      Node types: event, consequence, state, cause, role, credential,
                  requirement, prerequisite, trigger, step, outcome, entity,
                  node, concept, attribute, other  (or integer 0..31)
      Roles:      root, intermediate, terminal, scope, query, other,
                  source, target  (or integer 0..7)
      Edge types: causes, requires, precedes, enables, depends_on,
                  related_to, blocks, is_a  (or integer 0..7)
      Node `id` fields are free-form strings.

    Args:
        graph_json: JSON string:
          {
            "nodes": [{"id": str, "type": int|str, "role": int|str}, ...],
            "edges": [{"src": str, "dst": str, "type": int|str, "scope": str?}, ...]
          }
        query_json: JSON string with one of these shapes:
          root_cause:        {"type": "root_cause", "target": id}
          connection_check:  {"type": "connection_check", "src": id, "dst": id}
          implication_check: {"type": "implication_check", "premise": id,
                              "conclusion": id, "context_scope": id}
          shortest_path:     {"type": "shortest_path", "src": id, "dst": id}

    Returns:
        JSON string with analysis results.

    Example:
        graph_json = '{"nodes": [{"id":"A","type":"event","role":"root"},'
                     ' {"id":"B","type":"consequence","role":"terminal"}],'
                     '"edges":[{"src":"A","dst":"B","type":"causes"}]}'
        query_json = '{"type":"root_cause","target":"B"}'
    """
    assert _engine is not None, "graph engine not initialized"
    return _engine.analyze_graph(graph_json, query_json)


@mcp.tool()
def compare_graphs(graph_a_json: str, graph_b_json: str) -> str:
    """Compare two graphs for structural similarity.

    Args:
        graph_a_json: first graph JSON
        graph_b_json: second graph JSON

    Returns:
        JSON string with {isomorphic (authoritative), signature_cosine (LiquidARC estimate), n_a, n_b}.
    """
    assert _engine is not None
    return _engine.compare_graphs(graph_a_json, graph_b_json)


@mcp.tool()
def get_graph_diagnostics(graph_json: str) -> str:
    """Return geometric diagnostics on a graph processed through LiquidARC.

    Args:
        graph_json: graph JSON

    Returns:
        JSON with metric CV, D²-median, criticality ratio D²/4τ, τ statistics,
        per-node τ distribution, metric clusters, per-node centrality in
        metric space, and size.
    """
    assert _engine is not None
    return _engine.get_graph_diagnostics(graph_json)


@mcp.tool()
def correct_answer(graph_json: str, query_json: str,
                   correct_answer_json: str) -> str:
    """Teach the model with a user correction (online learning).

    Use when `analyze_graph` returned a wrong result. The output head takes
    one gradient step toward the correct answer you supply, the correction
    is persisted to disk, and future calls benefit from the learning.

    The learned geometry (dynamics, context pool, embedding) is FROZEN —
    only the task-specific output head updates. This prevents catastrophic
    drift in the geometric substrate while allowing task-level correction.

    Args:
        graph_json: same graph JSON passed to analyze_graph
        query_json: same query JSON passed to analyze_graph
        correct_answer_json: user-supplied correct answer. Shape depends on
          the query type:
            root_cause:        {"root_cause": <node_id>}
            connection_check:  {"connected": true|false}
            implication_check: {"valid": true|false}

    Returns:
        JSON with:
          - learned: whether a gradient step was applied
          - before_loss / after_loss: loss on this example pre/post correction
          - current_prediction: the model's updated answer
          - corrections_applied: running count in this session

    Example:
        # Previously analyze_graph returned valid: false (wrong).
        correct_answer(
          graph_json=...,
          query_json='{"type":"implication_check","premise":"crypto_exam",
                       "conclusion":"linear_algebra",
                       "context_scope":"senior_engineer"}',
          correct_answer_json='{"valid": true}'
        )
    """
    assert _engine is not None
    return _engine.correct_answer(graph_json, query_json, correct_answer_json)


# ──────────────────────────────────────────────────────────────────────
# Navigator tools — text↔structure bridge over the graph engine
# See GEOMETRIC_NAVIGATOR_SPEC.md §6 (MCP Integration)
# ──────────────────────────────────────────────────────────────────────


@mcp.tool()
def navigate(user_text: str, pre_extracted_fragment_json: str = "") -> str:
    """Process a user interaction through the geometric navigator.

    The navigator extracts structure from text via the LLM, merges it
    into the persistent geometric state, runs graph-engine analysis on
    the relevant subgraph, and returns a structural hint the caller's
    LLM can use to compose a response. Text is never fed to the ODE.

    Args:
        user_text: the user's message (used only for extraction)
        pre_extracted_fragment_json: optional pre-extracted graph fragment
            as JSON. When supplied, the navigator skips LLM extraction.
            Shape: {"nodes": [{"id","type","role"}, ...],
                    "edges": [{"src","dst","type"[,"scope"]}, ...]}

    Returns:
        JSON with:
          structural_hint  — dict of analysis (type, root, chain, etc.)
          rendered_hint    — text block for inclusion in an LLM prompt
          context_nodes    — metric-nearest related nodes in state
          diagnostics      — CV, D²/4τ, clusters over accumulated graph
          query            — the inferred query the navigator ran
          analysis         — raw graph-engine output
          pattern_match    — known-pattern info if library matched
    """
    assert _navigator is not None, "navigator not initialized"
    pre = None
    if pre_extracted_fragment_json:
        try:
            pre = json.loads(pre_extracted_fragment_json)
        except Exception:
            pre = None
    result = _navigator.process_interaction(user_text, pre_extracted=pre)
    return json.dumps(result, default=str)


@mcp.tool()
def get_navigator_state() -> str:
    """Return a summary of the current geometric state: size, clusters,
    patterns, recent nodes."""
    assert _navigator is not None
    state = _navigator.state
    recent = sorted(state.nodes.items(),
                    key=lambda x: x[1].get("last_seen", 0),
                    reverse=True)[:10]
    return json.dumps({
        "n_nodes": len(state.nodes),
        "n_edges": len(state.edges),
        "n_clusters": len(state.clusters),
        "n_patterns": (len(_navigator.patterns.patterns)
                       if _navigator.patterns else 0),
        "clusters": state.clusters,
        "recent_nodes": [
            {"id": nid, "type": meta["type"], "role": meta["role"],
             "mention_count": meta.get("mention_count", 1)}
            for nid, meta in recent
        ],
    }, default=str)


@mcp.tool()
def reset_navigator_state() -> str:
    """Clear persistent geometric state. Pattern library is preserved."""
    assert _navigator is not None
    _navigator.state.reset()
    return json.dumps({"reset": True,
                       "n_nodes": 0, "n_edges": 0, "n_clusters": 0})


@mcp.tool()
def query_navigator(anchor_node_ids_json: str, k: int = 10) -> str:
    """Return the k metric-nearest nodes to a given anchor set.

    Args:
        anchor_node_ids_json: JSON list of node IDs to use as the query
        k: max number of nearest nodes to return

    Returns:
        JSON list of {id, type, role, metric_distance, cluster_id}.
    """
    assert _navigator is not None
    try:
        anchors = json.loads(anchor_node_ids_json)
        if not isinstance(anchors, list):
            raise ValueError("expected JSON list")
    except Exception as exc:
        return json.dumps({"error": f"bad anchor list: {exc}"})
    hits = _navigator.state.query_relevant([str(a) for a in anchors], k=k)
    return json.dumps(hits, default=str)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8421)
    p.add_argument('--corrections_log',
                   default='/workspace/liquid-arc/graph_corrections.jsonl',
                   help='persistent JSONL log of user corrections; replayed on startup')
    p.add_argument('--correction_lr', type=float, default=1e-5,
                   help='learning rate for online correction steps (default 1e-5)')
    p.add_argument('--max_corrections_per_example', type=int, default=3,
                   help='cap on corrections per unique (graph, query) pair')
    p.add_argument('--replay_per_correction', type=int, default=8,
                   help='diverse replay examples to run per correction')
    p.add_argument('--health_check_variance_threshold', type=float, default=0.02,
                   help='head-output variance threshold below which head is considered collapsed')
    # Navigator
    p.add_argument('--navigator_state_path',
                   default='/workspace/liquid-arc/navigator_state.json',
                   help='persistent geometric-state JSON path')
    p.add_argument('--navigator_pattern_library',
                   default='/workspace/liquid-arc/navigator_patterns.json',
                   help='persistent pattern-library JSON path')
    p.add_argument('--navigator_max_nodes', type=int, default=512)
    p.add_argument('--navigator_pattern_threshold', type=float, default=0.85)
    p.add_argument('--extract_vllm_url',
                   default='http://localhost:30000/v1',
                   help='vLLM endpoint used for text→graph extraction')
    p.add_argument('--extract_model',
                   default='NVIDIA-Nemotron-3-Nano-30B-A3B-FP8')
    p.add_argument('--disable_navigator_extractor', action='store_true',
                   help=('do not instantiate the LLM extractor — callers must '
                         'pass pre_extracted_fragment_json to navigate()'))
    args = p.parse_args()

    global _engine, _navigator
    print(f"loading GraphEngine from {args.checkpoint} on {args.device}...",
          flush=True)
    print(f"  corrections log: {args.corrections_log}", flush=True)
    t0 = time.time()
    _engine = GraphEngine(
        args.checkpoint, device=args.device,
        corrections_log=args.corrections_log,
        correction_lr=args.correction_lr,
        max_corrections_per_example=args.max_corrections_per_example,
        replay_per_correction=args.replay_per_correction,
        health_check_variance_threshold=args.health_check_variance_threshold)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    # Navigator — shares the engine's ODE/MetricNet; adds persistent state
    # + pattern library + LLM extractor for text↔structure.
    state = GeometricState(args.navigator_state_path, _engine,
                           max_nodes=args.navigator_max_nodes)
    patterns = PatternLibrary(args.navigator_pattern_library)
    extractor: Optional[LLMExtractor] = None
    if not args.disable_navigator_extractor:
        extractor = LLMExtractor(base_url=args.extract_vllm_url,
                                 model=args.extract_model)
    _navigator = GeometricNavigator(
        engine=_engine, state=state, extractor=extractor,
        pattern_library=patterns,
        pattern_threshold=args.navigator_pattern_threshold,
    )
    print(f"navigator ready: state={args.navigator_state_path} "
          f"nodes={len(state.nodes)} edges={len(state.edges)} "
          f"patterns={len(patterns.patterns)} "
          f"extractor={'on' if extractor else 'off'}",
          flush=True)

    def _shutdown(signum, frame):
        print(f"\nreceived signal {signum}, shutting down", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"starting MCP server on {args.host}:{args.port}", flush=True)
    mcp.run(transport='sse', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
