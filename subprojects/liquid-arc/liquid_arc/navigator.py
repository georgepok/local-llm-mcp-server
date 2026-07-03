"""GeometricNavigator — orchestrates geometric reasoning alongside an LLM.

Spec: GEOMETRIC_NAVIGATOR_SPEC.md §3 (Structural Hint Generator)
       GEOMETRIC_NAVIGATOR_SPEC.md §4 (Navigator Orchestrator)

The navigator NEVER processes text. The LLM translates text↔structure;
the navigator operates on typed graphs and returns structured hints.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import networkx as nx

from .graph_engine_inference import GraphEngine
from .navigator_extract import LLMExtractor, extract_graph
from .navigator_patterns import PatternLibrary
from .navigator_state import GeometricState


# ----------------------------------------------------------------------
# Hint generation
# ----------------------------------------------------------------------


class HintGenerator:
    """Convert graph-engine analysis into a structured hint dict.

    The hint is JSON-serializable. A downstream LLM prompt template is
    responsible for rendering it into natural language — this module
    never produces prose.

    Phase 2 adds confidence-tiered verbosity (NAVIGATOR_CONTINUATION_SPEC
    Fix 2): high-confidence analysis → minimal hint (matches the 29/30
    networkx behaviour on simple problems); low-confidence → full
    geometric context (patterns, metric-nearest).
    """

    HIGH_CONFIDENCE = 0.95
    MEDIUM_CONFIDENCE = 0.70

    def generate_hint(self,
                      analysis: Optional[Dict[str, Any]],
                      state: GeometricState,
                      query: Optional[Dict[str, Any]],
                      pattern_match: Optional[Dict[str, Any]] = None,
                      ) -> Dict[str, Any]:
        # Pick the verbosity tier up front — downstream renderers use
        # hint['tier'] to decide what to include.
        confidence = (analysis or {}).get("confidence")
        if confidence is None:
            # connection_check / implication_check have deterministic
            # authoritative answers (nx.has_path) — treat them as high-conf.
            qtype = (query or {}).get("type")
            if qtype in ("connection_check", "implication_check",
                         "shortest_path"):
                confidence = 1.0
            else:
                confidence = 0.5
        if confidence >= self.HIGH_CONFIDENCE:
            tier = "minimal"
        elif confidence >= self.MEDIUM_CONFIDENCE:
            tier = "standard"
        else:
            tier = "full"

        hint: Dict[str, Any] = {"tier": tier,
                                 "_confidence": float(confidence)}
        if analysis is None or query is None:
            hint["analysis_type"] = None
            hint["analysis"] = None
        else:
            qtype = query.get("type")
            hint["analysis_type"] = qtype
            if qtype == "root_cause":
                hint["root"] = analysis.get("root_cause")
                hint["chain"] = analysis.get("path")
                hint["hops"] = analysis.get("hops")
                hint["confidence"] = analysis.get("confidence")
                hint["target"] = query.get("target")
            elif qtype == "connection_check":
                hint["connected"] = analysis.get("connected")
                hint["head_prob"] = analysis.get("connected_head_prob")
                hint["src"] = query.get("src")
                hint["dst"] = query.get("dst")
            elif qtype == "implication_check":
                hint["valid"] = analysis.get("valid")
                hint["head_valid_prob"] = analysis.get("head_valid_prob")
                hint["premise"] = query.get("premise")
                hint["conclusion"] = query.get("conclusion")
                hint["scope"] = query.get("context_scope")
                hint["n_edges_after_scope_filter"] = analysis.get(
                    "n_edges_after_scope_filter")
            elif qtype == "shortest_path":
                hint["path"] = analysis.get("path")
                hint["hops"] = analysis.get("hops")

        # Related-context retrieval is skipped for minimal-tier hints —
        # on high-confidence answers, extra nodes are noise (Phase 1 showed
        # the navigator loses vs plain networkx when overstuffed).
        if tier != "minimal":
            anchors: List[str] = []
            if query:
                for key in ("target", "src", "premise", "context_scope"):
                    if key in query and query[key] in state.nodes:
                        anchors.append(query[key])
            if anchors:
                # Both metric and graph adjacency give complementary info.
                related = state.query_relevant(anchors, k=5, mode="both")
                hint["related_context"] = [
                    {"id": r["id"], "type": r.get("type"),
                     "source": r.get("source", "?"),
                     "distance": round(r.get("metric_distance",
                                              r.get("graph_distance", -1)), 4),
                     "cluster_id": r.get("cluster_id")}
                    for r in related
                ]
            else:
                hint["related_context"] = []
        else:
            hint["related_context"] = []

        if pattern_match:
            hint["known_pattern"] = {
                "label": pattern_match.get("label"),
                "similarity": round(pattern_match.get("similarity", 0.0), 3),
                "prior_occurrences": pattern_match.get("count", 1),
            }
        return hint


HINT_RENDER_TEMPLATE = """\
Structural analysis of the situation graph:
{body}
{related_section}{pattern_section}\
Use the ANSWER line above as your authoritative answer unless the question \
asks for something different.\
"""

MINIMAL_HINT_TEMPLATE = "Structural analysis: {body}"


def render_topology_digest(digest: Dict[str, Any]) -> str:
    """Render a topology digest (global graph view) for the LLM prompt.

    Emits bullet-point centrality hubs + clusters + pattern inventory.
    This is the geometric answer to global questions like 'what are the
    single points of failure across everything we've discussed'.
    """
    if not digest:
        return ""
    lines: List[str] = [
        "Structural digest of the accumulated graph:",
        f"  size: {digest.get('n_nodes', '?')} nodes, "
        f"{digest.get('n_edges', '?')} edges, "
        f"{digest.get('n_clusters', '?')} metric clusters",
    ]
    diag = digest.get("diagnostics") or {}
    if diag.get("cv_g") is not None:
        lines.append(
            f"  geometry: CV(g)={diag['cv_g']:.2f}, "
            f"tau_mean={diag.get('tau_mean', 0):.2f}, "
            f"crit={diag.get('criticality_ratio', 0):.2f}")

    spof = digest.get("top_spof") or []
    if spof:
        lines.append("  top-5 single points of failure "
                     "(by downstream causal reach):")
        for s in spof:
            lines.append(
                f"    - {s['id']} ({s.get('type')}/{s.get('role')}, "
                f"affects {s['downstream_reach']} downstream nodes)")

    bet = digest.get("top_betweenness") or []
    if bet:
        lines.append("  top-5 bridge/bottleneck nodes (by betweenness):")
        for b in bet:
            lines.append(
                f"    - {b['id']} ({b.get('type')}/{b.get('role')}, "
                f"b={b['betweenness']:.3f})")

    clusters = digest.get("clusters") or []
    if clusters:
        lines.append(f"  cascade 'shapes' ({digest.get('n_clusters', '?')} "
                     "distinct metric clusters total — top 6 shown):")
        for c in clusters:
            exemplars = ", ".join(c.get("exemplars", [])[:3])
            lines.append(
                f"    - cluster {c['cluster_id']}: size={c['size']} "
                f"[{exemplars}{'…' if c['size'] > 3 else ''}]")

    patterns = digest.get("patterns") or []
    if patterns:
        lines.append("  pattern library (structural signatures seen so far):")
        for p in patterns:
            lines.append(f"    - {p['label']} (seen {p['count']}×)")

    lines.append("")
    lines.append(
        "The top-SPOF list answers 'what are the single points of failure' "
        "directly: each entry's removal would cascade into `downstream_reach` "
        "many nodes. The cluster count answers 'how many distinct cascade "
        "shapes'.")
    return "\n".join(lines)


def render_hint(hint: Dict[str, Any]) -> str:
    """Render a hint dict into a text block suitable for a user-prompt.

    Verbosity controlled by hint['tier']:
      - minimal  → single-line answer (matches the networkx optimal shape)
      - standard → answer + chain/implication details
      - full     → + related-context nodes + pattern-library match
    """
    if not hint:
        return ""
    tier = hint.get("tier", "standard")
    atype = hint.get("analysis_type") or "structural_context"
    lines: List[str] = []
    if atype == "root_cause":
        # Lead with the answer. Most questions with a root_cause analysis
        # ARE asking for the root cause — state it plainly first.
        lines.append(f"ANSWER: The root cause is `{hint.get('root')}`.")
        chain = hint.get("chain") or []
        if chain:
            lines.append(f"Causal chain: {' → '.join(chain)} "
                         f"({hint.get('hops')} hops).")
    elif atype == "connection_check":
        verdict = "Yes" if hint.get("connected") else "No"
        lines.append(f"ANSWER: {verdict} — `{hint.get('src')}` is "
                     f"{'connected to' if hint.get('connected') else 'NOT connected to'} "
                     f"`{hint.get('dst')}`.")
    elif atype == "implication_check":
        verdict = "Yes" if hint.get("valid") else "No"
        lines.append(f"ANSWER: {verdict} — under scope `{hint.get('scope')}`, "
                     f"`{hint.get('premise')}` {'does' if hint.get('valid') else 'does NOT'} "
                     f"transitively require `{hint.get('conclusion')}`.")
        if tier == "full" and hint.get("n_edges_after_scope_filter") is not None:
            lines.append(f"(Scope-filtered graph has "
                         f"{hint['n_edges_after_scope_filter']} edges.)")
    elif atype == "shortest_path":
        path = hint.get("path") or []
        lines.append(f"Shortest path: {' → '.join(path)}")

    body = "\n".join(lines) if lines else "(no specific analysis available)"

    # Minimal tier: just the ANSWER line, nothing else. This matches the
    # networkx-style hint that achieved 29/30 in Phase 1.
    if tier == "minimal":
        return MINIMAL_HINT_TEMPLATE.format(body=body)

    # Standard / full: related-context is shown only when there's NO
    # explicit ANSWER line (to keep signal clean on confident answers),
    # or when the tier is 'full' (uncertain answer — give the LLM context).
    has_answer = any(l.startswith("ANSWER:") for l in lines)
    related_section = ""
    if tier == "full" or not has_answer:
        related = hint.get("related_context") or []
        if related:
            related_section = (
                "\nRelated context nodes:\n" + "\n".join(
                    f"  - {r['id']} ({r.get('type')}, via {r.get('source')}, "
                    f"d={r['distance']})"
                    for r in related) + "\n")

    pattern = hint.get("known_pattern")
    pattern_section = ""
    if pattern:
        pattern_section = (
            f"\n(Structural pattern similar to `{pattern['label']}` "
            f"seen previously, similarity={pattern['similarity']:.2f}.)\n")

    return HINT_RENDER_TEMPLATE.format(
        body=body,
        related_section=related_section,
        pattern_section=pattern_section,
    )


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------


class GeometricNavigator:
    """Main loop: text → structure → analysis → hint."""

    def __init__(self,
                 engine: GraphEngine,
                 state: GeometricState,
                 extractor: Optional[LLMExtractor] = None,
                 pattern_library: Optional[PatternLibrary] = None,
                 pattern_threshold: float = 0.85):
        self.engine = engine
        self.state = state
        self.extractor = extractor
        self.patterns = pattern_library
        self.pattern_threshold = pattern_threshold
        self.hints = HintGenerator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_interaction(self, user_text: str,
                            pre_extracted: Optional[Dict[str, Any]] = None,
                            ) -> Dict[str, Any]:
        """Main entry. Returns hint + context + diagnostics.

        Args:
            user_text:      the raw user message (used only for extraction)
            pre_extracted:  if provided, skip LLM extraction and use this
                            fragment directly (useful for evaluation).
        """
        t0 = time.time()

        # Step 1: text → structure
        if pre_extracted is not None:
            fragment = pre_extracted
        elif self.extractor is not None:
            try:
                fragment = extract_graph(user_text, self.extractor)
            except Exception as exc:
                print(f"[navigator] extraction error: {exc}", flush=True)
                fragment = None
        else:
            fragment = None

        if not fragment or not fragment.get("nodes"):
            # Phase 2 refinement: when the caller asks about the global
            # state without naming anchors (topology / "what are the hubs"
            # / "cascade shapes across everything"), synthesize a topology
            # digest from the accumulated graph. Geometry-native answers
            # to topology-native questions.
            if len(self.state.nodes) >= 4:
                digest = self._topology_digest()
                return {
                    "structural_hint": digest,
                    "rendered_hint": render_topology_digest(digest),
                    "context_nodes": [],
                    "relevant_text_segments": [],
                    "diagnostics": digest.get("diagnostics", {}),
                    "fragment": None,
                    "query": {"type": "topology_digest"},
                    "analysis": None,
                    "elapsed_s": time.time() - t0,
                }
            return {
                "structural_hint": None,
                "rendered_hint": "",
                "context_nodes": [],
                "relevant_text_segments": [],
                "diagnostics": {},
                "fragment": None,
                "query": None,
                "analysis": None,
                "elapsed_s": time.time() - t0,
            }

        # Step 2: merge into persistent state, indexing source text under
        # every node so we can later retrieve snippets for LLM context.
        merge_report = self.state.merge_fragment(fragment, source_text=user_text)

        # Step 3: infer query type from the fragment shape
        query = self._infer_query(fragment)

        # Step 4: run graph-engine analysis on the relevant subgraph
        analysis: Optional[Dict[str, Any]] = None
        if query is not None:
            subgraph = self._extract_relevant_subgraph(query)
            # Guarantee the anchor node is in the subgraph
            required = self._query_anchors(query)
            missing = [r for r in required if r not in {n["id"] for n in subgraph["nodes"]}]
            if not missing:
                try:
                    raw = self.engine.analyze_graph(
                        json.dumps(subgraph), json.dumps(query))
                    analysis = json.loads(raw)
                    if not isinstance(analysis, dict) or "error" in analysis:
                        analysis = None
                except Exception as exc:
                    print(f"[navigator] analyze_graph error: {exc}", flush=True)
                    analysis = None

        # Step 5: pattern-library lookup
        pattern_match = None
        current_sig = None
        if self.patterns is not None:
            current_sig = self.state.get_signature()
            if current_sig is not None:
                pattern_match = self.patterns.find_nearest(
                    current_sig, threshold=self.pattern_threshold)

        # Step 6: structural hint
        hint = self.hints.generate_hint(analysis, self.state, query,
                                        pattern_match=pattern_match)

        # Step 7: record novel pattern if we have one and it's unseen
        if (self.patterns is not None and current_sig is not None
                and pattern_match is None and analysis is not None
                and query is not None):
            self.patterns.store(current_sig, {
                "label": self._auto_label(query, analysis),
                "source_query": query,
                "timestamp": time.time(),
            })

        # Step 8: diagnostics
        diagnostics: Dict[str, Any] = {}
        if len(self.state.nodes) >= 2:
            try:
                diagnostics = json.loads(self.engine.get_graph_diagnostics(
                    json.dumps(self.state.to_graph_dict())))
                diagnostics["n_clusters"] = len(self.state.clusters)
            except Exception as exc:
                print(f"[navigator] diagnostics error: {exc}", flush=True)
                diagnostics = {}

        anchor_ids = self._query_anchors(query) if query else [
            n["id"] for n in fragment["nodes"]
        ]
        context_nodes = self.state.query_relevant(anchor_ids, k=10, mode="both")

        # Fix 3: retrieve text snippets associated with the top context
        # nodes so the caller can compose historical context for the LLM.
        relevant_ids = [n["id"] for n in context_nodes] + anchor_ids
        relevant_text_segments = self.state.retrieve_text_for_nodes(
            relevant_ids, max_segments=5)

        return {
            "structural_hint": hint,
            "rendered_hint": render_hint(hint),
            "context_nodes": context_nodes,
            "relevant_text_segments": relevant_text_segments,
            "diagnostics": diagnostics,
            "fragment": fragment,
            "merge_report": merge_report,
            "query": query,
            "analysis": analysis,
            "pattern_match": pattern_match,
            "elapsed_s": time.time() - t0,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _query_anchors(query: Dict[str, Any]) -> List[str]:
        anchors: List[str] = []
        for key in ("target", "src", "dst", "premise", "conclusion",
                    "context_scope"):
            if key in query and query[key]:
                anchors.append(str(query[key]))
        return anchors

    @staticmethod
    def _fragment_has_disconnected_components(frag: Dict[str, Any]) -> bool:
        g = nx.Graph()
        for n in frag["nodes"]:
            g.add_node(n["id"])
        for e in frag["edges"]:
            g.add_edge(e["src"], e["dst"])
        try:
            return nx.number_connected_components(g) > 1
        except Exception:
            return False

    @staticmethod
    def _fragment_components(frag: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        g = nx.Graph()
        for n in frag["nodes"]:
            g.add_node(n["id"])
        for e in frag["edges"]:
            g.add_edge(e["src"], e["dst"])
        id_to_node = {n["id"]: n for n in frag["nodes"]}
        comps = []
        for cc in nx.connected_components(g):
            comps.append([id_to_node[i] for i in cc if i in id_to_node])
        return comps

    def _infer_query(self, fragment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Heuristic: decide what question the fragment is asking.

        Based entirely on node roles / types / fragment shape — no text.
        """
        nodes = fragment["nodes"]
        edges = fragment["edges"]
        scopes = [n for n in nodes if n.get("role") == "scope"]
        terminals = [n for n in nodes if n.get("role") == "terminal"]

        causal_edges = [e for e in edges
                        if e.get("type") in ("causes", "precedes")]
        req_edges = [e for e in edges
                     if e.get("type") in ("requires", "depends_on", "enables")]

        # Implication: scopes + premise/conclusion-ish structure
        if scopes and req_edges and len(nodes) >= 3:
            premises = [n for n in nodes if n.get("type") in (
                "requirement", "event", "role")]
            conclusions = [n for n in nodes if n.get("type") in (
                "prerequisite", "credential", "state")]
            if premises and conclusions:
                return {
                    "type": "implication_check",
                    "premise": premises[0]["id"],
                    "conclusion": conclusions[-1]["id"],
                    "context_scope": scopes[0]["id"],
                }

        # Root cause: terminals + causal chain
        if terminals and causal_edges:
            return {"type": "root_cause", "target": terminals[0]["id"]}

        # Connection: multiple components and no causal structure
        if self._fragment_has_disconnected_components(fragment):
            comps = self._fragment_components(fragment)
            if len(comps) >= 2 and comps[0] and comps[1]:
                return {
                    "type": "connection_check",
                    "src": comps[0][0]["id"],
                    "dst": comps[1][0]["id"],
                }

        # Last resort: root_cause toward the highest-degree terminal-ish node.
        if nodes and causal_edges:
            # Use the destination of the last causal edge as target.
            return {"type": "root_cause", "target": causal_edges[-1]["dst"]}
        return None

    def _extract_relevant_subgraph(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the subgraph around a query anchor (capped at 20 nodes)."""
        anchors = self._query_anchors(query)
        anchor_ids: List[str] = [a for a in anchors if a in self.state.nodes]

        if not anchor_ids:
            return self.state.to_graph_dict()

        relevant = self.state.query_relevant(anchor_ids, k=20)
        ids = set(anchor_ids) | {r["id"] for r in relevant}
        # Always pull in everything reachable from anchors within 2 hops in
        # the accumulated edge set (ensures connection_check / implication
        # actually has the relevant edges).
        two_hop = self._two_hop_ids(anchor_ids)
        ids |= two_hop
        if len(ids) > 30:
            ids = set(anchor_ids) | set(list(ids)[:30])

        nodes_out = [
            {"id": nid, "type": self.state.nodes[nid]["type"],
             "role": self.state.nodes[nid]["role"]}
            for nid in ids if nid in self.state.nodes
        ]
        edges_out = [
            {k: v for k, v in e.items() if v is not None}
            for e in self.state.edges
            if e["src"] in ids and e["dst"] in ids
        ]
        return {"nodes": nodes_out, "edges": edges_out}

    def _two_hop_ids(self, anchor_ids: List[str]) -> set:
        g = nx.Graph()
        for nid in self.state.nodes:
            g.add_node(nid)
        for e in self.state.edges:
            if e["src"] in self.state.nodes and e["dst"] in self.state.nodes:
                g.add_edge(e["src"], e["dst"])
        reach = set()
        for a in anchor_ids:
            if a not in g:
                continue
            try:
                dists = nx.single_source_shortest_path_length(g, a, cutoff=2)
            except Exception:
                dists = {a: 0}
            reach.update(dists.keys())
        return reach

    def _topology_digest(self) -> Dict[str, Any]:
        """Summarize the full accumulated graph geometrically + structurally.

        Used when the user asks a global question ('what are the hubs?',
        'how many cascade shapes?') instead of a query about a specific
        node. The content layer here is what the trained graph engine's
        diagnostics already computes — this surfaces it at navigator level.
        """
        digest: Dict[str, Any] = {
            "n_nodes": len(self.state.nodes),
            "n_edges": len(self.state.edges),
            "n_clusters": len(self.state.clusters),
        }

        # Per-node metric centrality + cluster memberships: straight from
        # the engine's get_graph_diagnostics on the full state.
        diag: Dict[str, Any] = {}
        try:
            diag = json.loads(self.engine.get_graph_diagnostics(
                json.dumps(self.state.to_graph_dict())))
        except Exception as exc:
            print(f"[navigator] diagnostics error: {exc}", flush=True)
        if diag:
            digest["diagnostics"] = {
                "cv_g": diag.get("cv_g"),
                "criticality_ratio": diag.get("criticality_ratio"),
                "tau_mean": diag.get("tau_mean"),
            }
            centrality = diag.get("per_node_centrality_metric_space", {}) or {}
            top_metric = sorted(centrality.items(),
                                key=lambda kv: -kv[1])[:5]
            digest["top_metric_centrality"] = [
                {"id": nid,
                 "centrality": round(v, 4),
                 "type": self.state.nodes.get(nid, {}).get("type"),
                 "cluster_id": self.state._get_cluster(nid)}
                for nid, v in top_metric
            ]

        # Single-point-of-failure hubs: for each node, count transitive
        # descendants in the directed causal graph. Ranked high = "if this
        # goes down, the most downstream stuff is affected". This is the
        # structurally-correct SPOF metric, not metric centrality (which
        # peaks on consequences in the densest cluster).
        try:
            g = nx.DiGraph()
            for nid in self.state.nodes:
                g.add_node(nid)
            for e in self.state.edges:
                if e["src"] in self.state.nodes and e["dst"] in self.state.nodes:
                    g.add_edge(e["src"], e["dst"])
            reach: Dict[str, int] = {}
            for nid in g.nodes:
                try:
                    reach[nid] = len(nx.descendants(g, nid))
                except Exception:
                    reach[nid] = 0
            # Betweenness ranking on the undirected view for bridge/bottleneck.
            bet = nx.betweenness_centrality(g.to_undirected())

            top_spof = sorted(
                ((nid, r) for nid, r in reach.items() if r > 0),
                key=lambda kv: -kv[1])[:5]
            digest["top_spof"] = [
                {"id": nid, "downstream_reach": r,
                 "type": self.state.nodes.get(nid, {}).get("type"),
                 "role": self.state.nodes.get(nid, {}).get("role")}
                for nid, r in top_spof
            ]
            top_bet = sorted(bet.items(), key=lambda kv: -kv[1])[:5]
            digest["top_betweenness"] = [
                {"id": nid, "betweenness": round(v, 4),
                 "type": self.state.nodes.get(nid, {}).get("type"),
                 "role": self.state.nodes.get(nid, {}).get("role")}
                for nid, v in top_bet if v > 0
            ]
        except Exception as exc:
            print(f"[navigator] spof error: {exc}", flush=True)

        # Cluster summary
        digest["clusters"] = [
            {"cluster_id": c["cluster_id"],
             "size": c["size"],
             "exemplars": c["members"][:3]}
            for c in self.state.clusters[:6]
        ]

        # Pattern library inventory
        if self.patterns:
            digest["patterns"] = [
                {"label": p.get("label"),
                 "count": p.get("count", 1)}
                for p in self.patterns.patterns[:8]
            ]
        return digest

    @staticmethod
    def _auto_label(query: Dict[str, Any],
                    analysis: Dict[str, Any]) -> str:
        qtype = query.get("type", "unknown")
        if qtype == "root_cause":
            return f"root_cause:{analysis.get('root_cause','?')}"
        if qtype == "connection_check":
            return f"connection:{analysis.get('connected')}"
        if qtype == "implication_check":
            return f"implication:{analysis.get('valid')}"
        return f"query:{qtype}"
