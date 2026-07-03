"""Sentence-window chunker.

Splits a document into overlapping ~target_tokens chunks. Token count is
approximated as whitespace-separated words (good enough for retrieval
boundaries — real token count depends on the LLM).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


class Chunker:
    def __init__(self, target_tokens: int = 500, overlap_tokens: int = 100):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None
              ) -> List[Dict[str, Any]]:
        sentences = [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]
        if not sentences:
            return []
        chunks: List[Dict[str, Any]] = []
        buf: List[str] = []
        tok_count = 0
        idx = 0
        for s in sentences:
            s_tok = len(s.split())
            if buf and tok_count + s_tok > self.target_tokens:
                chunks.append(self._emit(buf, idx, metadata))
                idx += 1
                # Carry overlap (last few sentences up to overlap_tokens)
                buf = self._carry_overlap(buf)
                tok_count = sum(len(x.split()) for x in buf)
            buf.append(s)
            tok_count += s_tok
        if buf:
            chunks.append(self._emit(buf, idx, metadata))
        return chunks

    def _carry_overlap(self, buf: List[str]) -> List[str]:
        out: List[str] = []
        tok = 0
        for s in reversed(buf):
            s_tok = len(s.split())
            if tok + s_tok > self.overlap_tokens and out:
                break
            out.insert(0, s)
            tok += s_tok
        return out

    def _emit(self, buf: List[str], idx: int,
              metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        text = " ".join(buf)
        chunk: Dict[str, Any] = {
            "text": text,
            "chunk_id": idx,
            "n_tokens": len(text.split()),
        }
        if metadata:
            chunk["metadata"] = dict(metadata)
        return chunk
