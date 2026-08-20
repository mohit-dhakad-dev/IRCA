"""Memory retrieval layer: search the past-incidents collection built by
memory/ingest.py for the top-k prior incidents most similar to a symptom
query, with a confidence gate mirroring rag/retrieve.py.

See docs/design.md, "RAG + Memory — Retrieval Contract (Session 6 spec)".
"""

from __future__ import annotations

from pathlib import Path

from chromadb.errors import NotFoundError

from memory.ingest import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from vectorstore import index_not_found_error, query_collection

# 0.40, NOT the 0.5 used by rag/retrieve.py. Deliberately independent: memory
# scores on a different scale, so a threshold calibrated on one collection does
# not transfer to the other. Measured by eval/calibrate_retrieval.py over the 13
# gold-bearing tickets (see docs/decisions.md E5):
#
#   correct top-1 scores   memory 0.3244-0.6170 (median 0.5719)
#                        runbooks 0.5511-0.7992 (median 0.7171)
#
# A ticket-to-incident match is short symptom-paraphrase against short
# symptom-paraphrase; a runbook match is a query against a long prose section.
# The same 0.5 that rejects 0/15 on runbooks rejected 4 of the 13 tickets whose
# gold incident WAS present in top-3 (T001 0.4824, T002 0.3244, T003 0.4944,
# T007 0.3816). At 0.40 the sweep admits 11 with 10 correct and wrongly rejects
# 1 correct top-1, against 3 at 0.50.
#
# Why a looser gate is defensible HERE and not for search_runbooks: per
# docs/design.md a memory hit is a HINT that must be independently verified via
# query_logs/query_metrics before its root cause is adopted. The gate is not the
# safeguard for this tool -- verification is -- so suppressing a correct prior
# incident costs more than surfacing a weak one the agent must check anyway.
# Note this does NOT cleanly separate correct from wrong: a wrong top-1 scored
# 0.5981, above the correct median. No threshold does. See watched case T002.
SCORE_THRESHOLD = 0.40

# Session 6 spec: top-3 incidents per query.
DEFAULT_TOP_K = 3


def search_past_incidents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    chroma_path: Path | None = None,
    collection_name: str | None = None,
) -> dict:
    """Search the past-incidents index for the top_k prior incidents most
    similar to query. Returns the repo's standard tool-result shape:
    {"status": ..., "data": {...}, "summary": "..."}.

    Statuses:
      "ok"                  -- at least one incident scored >= SCORE_THRESHOLD.
                                data = {"query", "incidents": [...], "top_score"}.
      "no_confident_match"  -- incidents were found but none scored >=
                                SCORE_THRESHOLD (or the collection returned
                                zero results). The incidents that WERE found
                                are still returned in data (for
                                calibration/debugging) but the caller must
                                not treat them as a match.
      "error"                -- empty/whitespace query, or the index does
                                not exist yet (never a raw traceback).
    """
    if not query or not query.strip():
        return {
            "status": "error",
            "data": {},
            "summary": "Query must be a non-empty string.",
        }

    path = Path(chroma_path) if chroma_path is not None else CHROMA_PATH
    name = collection_name if collection_name is not None else COLLECTION_NAME

    try:
        matches = query_collection(path, name, query, top_k, EMBEDDING_MODEL)
    except NotFoundError:
        return index_not_found_error(path, name, "memory.ingest")

    incidents = []
    for match in matches:
        metadata = match["metadata"]
        incidents.append(
            {
                "incident_id": metadata.get("incident_id"),
                "symptom_summary": match["text"],
                "resolved_root_cause": metadata.get("resolved_root_cause"),
                "resolution": metadata.get("resolution"),
                "similarity_score": match["score"],
            }
        )

    top_score = incidents[0]["similarity_score"] if incidents else 0.0

    if not incidents or top_score < SCORE_THRESHOLD:
        return {
            "status": "no_confident_match",
            "data": {"query": query, "incidents": incidents, "top_score": top_score},
            "summary": (
                f"No prior incident matched confidently (best score {top_score:.2f} < "
                f"{SCORE_THRESHOLD:.2f} threshold). The absence of a match is not evidence "
                "either way -- do not treat it as ruling anything in or out; diagnose from "
                "logs/metrics/runbooks as usual."
            ),
        }

    named = ", ".join(
        f"{inc['incident_id']} ({inc['resolved_root_cause']}, {inc['similarity_score']:.2f})"
        for inc in incidents
    )
    summary = (
        f"Similar prior incidents for '{query}': {named}. These are PRIOR similar incidents, "
        "not evidence about the current one -- the root cause must be independently confirmed "
        "against query_logs/query_metrics before being adopted."
    )

    return {
        "status": "ok",
        "data": {"query": query, "incidents": incidents, "top_score": top_score},
        "summary": summary,
    }
