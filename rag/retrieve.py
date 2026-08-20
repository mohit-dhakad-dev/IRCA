"""RAG retrieval layer: search the runbook chunks built by rag/ingest.py for
the top-k sections most relevant to a query, with a confidence gate so a
weak match is never handed to the agent as if it were a real answer.

See docs/design.md, "RAG + Memory — Retrieval Contract (Session 6 spec)".
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from rag.ingest import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from vectorstore import _reset_cache
from vectorstore import get_collection as _vs_get_collection
from vectorstore import index_not_found_error, query_collection

# Session 6 spec: below this similarity score, the agent must not trust the
# match enough to act on it — return no_confident_match instead. Do not
# change this value; it is a contract, not a tuning knob.
SCORE_THRESHOLD = 0.5

# Session 6 spec: top-3 chunks per query.
DEFAULT_TOP_K = 3

# The client/collection cache, the cache-reset test hook, the "index not
# found" error builder, and collection resolution all now live in
# vectorstore.py (shared with memory/) -- re-exported here at their original
# names since existing tests and agent/tool_executor.py import them from
# rag.retrieve.
_reset_cache = _reset_cache


def _index_not_found_error(chroma_path: Path, collection_name: str) -> dict:
    """The standard status="error" result for a missing/unbuilt index, shared
    by both call sites that can hit chromadb.errors.NotFoundError."""
    return index_not_found_error(chroma_path, collection_name, "rag.ingest")


def _get_collection(chroma_path: Path, collection_name: str) -> chromadb.Collection:
    """Return the (possibly cached) collection for chroma_path/collection_name.

    Raises chromadb.errors.NotFoundError if the collection does not exist
    (e.g. rag.ingest.build_index() has never been run against this path) --
    callers must catch this narrowly and turn it into a status="error" result,
    never let it propagate as an unhandled traceback.
    """
    return _vs_get_collection(chroma_path, collection_name, EMBEDDING_MODEL)


def search_runbooks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    chroma_path: Path | None = None,
    collection_name: str | None = None,
) -> dict:
    """Search the runbook chunk index for the top_k sections most relevant to
    query. Returns the repo's standard tool-result shape:
    {"status": ..., "data": {...}, "summary": "..."}.

    Statuses:
      "ok"                 -- at least one chunk scored >= SCORE_THRESHOLD.
                               data = {"query", "chunks": [...], "top_score"}.
      "no_confident_match"  -- chunks were found but none scored >= SCORE_THRESHOLD
                               (or the collection returned zero results). The
                               chunks that WERE found are still returned in data
                               (for calibration/debugging) but the caller must
                               not act on them as a fix.
      "error"               -- empty/whitespace query, or the index does not
                               exist yet (never a raw traceback).
    """
    if not query or not query.strip():
        return {
            "status": "error",
            "data": {},
            "summary": "Query must be a non-empty string.",
        }

    path = Path(chroma_path) if chroma_path is not None else CHROMA_PATH
    name = collection_name if collection_name is not None else COLLECTION_NAME

    # query_collection (vectorstore.py) handles the evict-and-retry-once
    # repair of a stale cached collection handle -- see its docstring. A
    # rebuild (rag.ingest.build_index) between the time this collection
    # handle was cached and now gives the collection a new internal chroma
    # UUID, so the cached handle's query() would otherwise raise
    # NotFoundError even though the cache lookup just succeeded.
    try:
        matches = query_collection(path, name, query, top_k, EMBEDDING_MODEL)
    except NotFoundError:
        return _index_not_found_error(path, name)

    chunks = []
    for match in matches:
        metadata = match["metadata"]
        chunks.append(
            {
                "doc_id": metadata.get("doc_id"),
                "section": metadata.get("section"),
                "text": match["text"],
                "score": match["score"],
            }
        )

    top_score = chunks[0]["score"] if chunks else 0.0

    if not chunks or top_score < SCORE_THRESHOLD:
        return {
            "status": "no_confident_match",
            "data": {"query": query, "chunks": chunks, "top_score": top_score},
            "summary": (
                f"No runbook section matched confidently (best score {top_score:.2f} < "
                f"{SCORE_THRESHOLD:.2f} threshold). Do not base a fix on these; escalate "
                "or gather more evidence."
            ),
        }

    named = ", ".join(f"{c['doc_id']}/{c['section']} ({c['score']:.2f})" for c in chunks)
    summary = f"Top runbook matches for '{query}': {named}."

    return {
        "status": "ok",
        "data": {"query": query, "chunks": chunks, "top_score": top_score},
        "summary": summary,
    }
