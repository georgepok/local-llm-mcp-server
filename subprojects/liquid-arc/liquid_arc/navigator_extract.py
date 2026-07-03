"""LLM structure extraction for the geometric navigator.

Spec: GEOMETRIC_NAVIGATOR_SPEC.md §2 (Structure Extraction Prompt).

The LLM extracts a typed-graph fragment from free text. The navigator
never processes text — the LLM is the sole bridge between text and
structure. This module owns:

  1. EXTRACT_GRAPH_PROMPT — the exact prompt used across experiments
  2. LLMExtractor       — thin HTTP client around vLLM /v1/chat/completions
  3. extract_graph()    — prompt → JSON parse → schema validation
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import requests


NODE_TYPES = {
    "event", "consequence", "state", "cause", "role", "credential",
    "requirement", "prerequisite", "trigger", "step", "outcome",
    "concept", "entity", "attribute", "other",
}

ROLE_TYPES = {
    "root", "intermediate", "terminal", "scope", "query", "source", "target",
    "other",
}

EDGE_TYPES = {
    "causes", "requires", "precedes", "enables", "depends_on",
    "related_to", "blocks", "is_a",
}


EXTRACT_GRAPH_PROMPT = """Extract entities and relationships from the text below \
as a JSON graph.

Rules:
- Each entity becomes a node with:
    id:   snake_case unique identifier
    type: one of event, consequence, state, cause, role, credential,
          requirement, prerequisite, concept, entity
    role: one of root, intermediate, terminal, scope
- Each relationship becomes an edge with:
    src:  source node id
    dst:  destination node id
    type: one of causes, requires, precedes, enables, depends_on,
          related_to, blocks, is_a
- If a relationship only applies in a specific context, add
  a "scope" key with a scope node id.
- Extract ONLY what is stated or directly implied. Do not invent nodes.
- role=root means an upstream/originating entity (no incoming causes).
- role=terminal means a downstream/outcome entity (no outgoing causes).
- Keep ids short and descriptive (under 30 characters).

Text:
__TEXT__

Respond with ONLY valid JSON matching this shape, no explanation or \
markdown fence:
{"nodes": [{"id":"...","type":"...","role":"..."}, ...], \
"edges": [{"src":"...","dst":"...","type":"..."}, ...]}"""


class LLMExtractor:
    """Minimal HTTP client for a vLLM /chat/completions endpoint.

    Does NOT use prompt_embeds. Plain text-in / text-out. Disables
    thinking on models that support `enable_thinking` in the chat
    template (Nemotron, Qwen3) so the response is structured JSON only.
    """

    def __init__(self,
                 base_url: str = "http://spark-129a.local:30000/v1",
                 model: str = "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                 timeout: float = 60.0,
                 max_tokens: int = 800,
                 temperature: float = 0.1):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": "You extract structured data. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = requests.post(f"{self.base_url}/chat/completions",
                          json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


def _extract_first_json(text: str) -> Optional[str]:
    """Pull the first balanced {...} block from a string. Handles leading
    markdown fences, partial prefixes, and trailing commentary.
    """
    if not text:
        return None
    text = text.strip()
    # Strip ```json ... ``` fences if present
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    # Otherwise find first '{' and match braces
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _validate_and_normalize(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce keys, drop malformed nodes/edges, keep legal types only.

    Nodes missing id are dropped. Edges pointing to unknown nodes are
    dropped (they'd be rejected downstream by merge_fragment anyway).
    """
    nodes_raw = obj.get("nodes", []) or []
    edges_raw = obj.get("edges", []) or []

    seen_ids = set()
    nodes: List[Dict[str, Any]] = []
    for n in nodes_raw:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id", "")).strip()
        if not nid or nid in seen_ids:
            continue
        seen_ids.add(nid)
        ntype = str(n.get("type", "entity")).strip().lower()
        if ntype not in NODE_TYPES:
            ntype = "entity"
        role = str(n.get("role", "intermediate")).strip().lower()
        if role not in ROLE_TYPES:
            role = "intermediate"
        nodes.append({"id": nid, "type": ntype, "role": role})

    edges: List[Dict[str, Any]] = []
    for e in edges_raw:
        if not isinstance(e, dict):
            continue
        src = str(e.get("src", "")).strip()
        dst = str(e.get("dst", "")).strip()
        if not src or not dst or src not in seen_ids or dst not in seen_ids:
            continue
        et = str(e.get("type", "related_to")).strip().lower()
        if et not in EDGE_TYPES:
            et = "related_to"
        edge = {"src": src, "dst": dst, "type": et}
        if "scope" in e and e["scope"]:
            edge["scope"] = str(e["scope"]).strip()
        edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def extract_graph(text: str, extractor: LLMExtractor) -> Optional[Dict[str, Any]]:
    """Extract a typed graph fragment from text.

    Returns None when the LLM response cannot be parsed into valid JSON
    with at least one node. Returns the validated fragment otherwise.
    """
    prompt = EXTRACT_GRAPH_PROMPT.replace("__TEXT__", text)
    raw = extractor.generate(prompt)
    js = _extract_first_json(raw)
    if js is None:
        return None
    try:
        obj = json.loads(js)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    fragment = _validate_and_normalize(obj)
    if not fragment["nodes"]:
        return None
    return fragment
