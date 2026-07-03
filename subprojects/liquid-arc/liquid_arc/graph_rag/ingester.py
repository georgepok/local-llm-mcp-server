"""GraphRAGIngester — chunker → extractor → resolver → merge + vector add.

Processes a document end-to-end:
  1. Chunk the document into ~500-token overlapping windows
  2. For each chunk, add it to the vector DB
  3. LLM-extract a typed-graph fragment from the chunk
  4. Entity-resolve against existing state (dedupe)
  5. Merge resolved fragment into navigator.state with source_text=chunk
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .chunker import Chunker
from .entity_resolver import EntityResolver
from .vector_db import VectorDB


class GraphRAGIngester:
    def __init__(self, navigator, vector_db: VectorDB,
                 extractor, chunker: Optional[Chunker] = None,
                 resolver: Optional[EntityResolver] = None):
        self.navigator = navigator
        self.vector_db = vector_db
        self.extractor = extractor
        self.chunker = chunker or Chunker()
        self.resolver = resolver or EntityResolver(navigator.state)

    def ingest_document(self, doc_text: str,
                        doc_metadata: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
        """Ingest one document into the vector DB + knowledge graph.

        Returns a report: n_chunks, n_nodes_added, n_edges_added,
        extraction_failures, resolver_merges.
        """
        chunks = self.chunker.chunk(doc_text, metadata=doc_metadata)
        n_nodes_before = len(self.navigator.state.nodes)
        n_edges_before = len(self.navigator.state.edges)
        extraction_failures = 0
        resolver_merges = 0

        for chunk in chunks:
            # Vector DB: every chunk becomes a retrievable item
            meta = dict(chunk.get("metadata") or {})
            meta["chunk_id"] = chunk["chunk_id"]
            self.vector_db.add(chunk["text"], metadata=meta)

            # Knowledge graph: LLM extraction → resolution → merge
            try:
                fragment = self.extractor.extract(chunk["text"])
            except Exception:
                fragment = None
            if not fragment or not fragment.get("nodes"):
                extraction_failures += 1
                continue
            fragment = self.resolver.resolve(fragment)
            resolver_merges += len(fragment.get("_resolver_map", {}))
            # Attach doc metadata to every node via scope-like tag (stored
            # in state.nodes[...]["metadata"] for retrieval).
            self.navigator.state.merge_fragment(
                fragment, source_text=chunk["text"])

        return {
            "n_chunks": len(chunks),
            "n_nodes_added": len(self.navigator.state.nodes) - n_nodes_before,
            "n_edges_added": len(self.navigator.state.edges) - n_edges_before,
            "extraction_failures": extraction_failures,
            "resolver_merges": resolver_merges,
        }
