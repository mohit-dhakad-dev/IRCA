"""Memory ingestion layer: parses data/past_incidents.json into a Chroma
collection of past-incident records, mirroring rag/ingest.py in spirit.

Embedding decision: the embedded document is `symptom_summary` ONLY.
search_past_incidents is looked up by symptom text (what was observed about
the CURRENT incident), so the vector must represent symptoms. Embedding
`resolved_root_cause`/`resolution` too would let a query match on fix
wording rather than on what was observed -- the wrong retrieval axis for a
"has this happened before" lookup. Those two fields instead live in
metadata, retrieved alongside the match but never contributing to it.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

from vectorstore import CHROMA_PATH, EMBEDDING_MODEL, rebuild_collection

INCIDENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "past_incidents.json"
COLLECTION_NAME = "past_incidents"

REQUIRED_KEYS = {"incident_id", "symptom_summary", "resolved_root_cause", "resolution"}


def load_incidents(path: Path = INCIDENTS_PATH) -> list[dict]:
    """Load and validate data/past_incidents.json.

    Raises ValueError loudly on any structural violation -- silent partial
    ingestion is worse than a loud failure here (same principle as
    rag/ingest.py's parse_runbook).
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Past incidents file not found: {path}")

    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON array of incidents")

    seen_ids: set[str] = set()
    incidents: list[dict] = []
    for i, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: incident at index {i} is not a JSON object")

        keys = set(record)
        if keys != REQUIRED_KEYS:
            missing = REQUIRED_KEYS - keys
            extra = keys - REQUIRED_KEYS
            problems = []
            if missing:
                problems.append(f"missing {sorted(missing)}")
            if extra:
                problems.append(f"unexpected {sorted(extra)}")
            raise ValueError(f"{path}: incident at index {i} has wrong keys ({'; '.join(problems)})")

        for key in REQUIRED_KEYS:
            value = record[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{path}: incident at index {i} field '{key}' must be a non-empty string"
                )

        incident_id = record["incident_id"]
        if incident_id in seen_ids:
            raise ValueError(f"{path}: duplicate incident_id '{incident_id}'")
        seen_ids.add(incident_id)

        incidents.append(record)

    return incidents


def _incidents_to_chroma(incidents: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    """Map a list of incident dicts to the (ids, documents, metadatas) triple
    chroma wants. See module docstring for why the document is
    symptom_summary only."""
    ids = [inc["incident_id"] for inc in incidents]
    documents = [inc["symptom_summary"] for inc in incidents]
    metadatas = [
        {
            "incident_id": inc["incident_id"],
            "resolved_root_cause": inc["resolved_root_cause"],
            "resolution": inc["resolution"],
        }
        for inc in incidents
    ]
    return ids, documents, metadatas


def build_index(
    incidents_path: Path = INCIDENTS_PATH,
    chroma_path: Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
) -> chromadb.Collection:
    """Build (or idempotently rebuild) the Chroma collection over all past
    incidents. Deletes any existing collection of the same name first, so a
    rerun overwrites rather than appends."""
    incidents = load_incidents(incidents_path)
    ids, documents, metadatas = _incidents_to_chroma(incidents)
    return rebuild_collection(
        chroma_path=chroma_path,
        collection_name=collection_name,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        model_name=model_name,
    )


if __name__ == "__main__":
    collection = build_index()

    incidents = load_incidents()
    counts: dict[str, int] = {}
    for inc in incidents:
        root_cause = inc["resolved_root_cause"]
        counts[root_cause] = counts.get(root_cause, 0) + 1

    print("Cleared and rebuilt collection from scratch.")
    print("Incidents per root cause:")
    for root_cause in sorted(counts):
        print(f"  {root_cause}: {counts[root_cause]}")
    print(f"Total incidents in collection: {collection.count()}")
