"""Retrieval-grounded classification of which tickets WARRANT a call to
memory.search_past_incidents, persisted to eval/results/memory_warrant.json.

A ticket "warrants" memory only if a correct prior incident is actually
retrievable for it -- not because its category/expected_behavior looks like
it should. Every main (non-adversarial) ticket in data/tickets.json is
classified into exactly one of three buckets, using the LIVE
memory.store.search_past_incidents() call over the ticket's ticket_text:

    "warranted"  -- status == "ok" AND the top incident's resolved_root_cause
                    equals the ticket's gold_root_cause. Memory would have
                    handed the agent the right lead.

    "hazard"     -- the ticket's gold_root_cause is null (an ambiguous
                     ticket that should escalate) AND status == "ok". Memory
                     offers a confident-looking hit here, but the correct
                     agent behaviour is to escalate, not adopt a lead --
                     calling memory on these tickets is a FALSE-LEAD HAZARD,
                     not an opportunity, and must never be scored as if
                     "warranted".

    "no_signal"  -- everything else. This deliberately includes the case
                     where gold_root_cause is non-null and the top hit is
                     CONFIDENT but WRONG (status == "ok" but
                     top_resolved_root_cause != gold_root_cause): a
                     confident wrong hit is not a reason to call memory, it
                     is a reason NOT to -- classifying it "warranted" would
                     reward the agent for retrieving a wrong lead. It also
                     includes plain misses (status != "ok").

This is why the classification is retrieval-grounded rather than derived
from ticket metadata (e.g. required_tools): zero of the 63 main tickets list
search_past_incidents in required_tools, so there is no existing gold signal
to classify against -- this script IS the gold signal, measured directly
against the live memory corpus.

CRITICAL: this is the ONLY module in this increment allowed to touch chroma.
eval/memory_metrics.py and its tests must read ONLY the JSON this script
persists -- never call search_past_incidents directly -- so that the metrics
layer (and the test suite) stays offline and deterministic even on a machine
where the chroma index has not been built. Mirrors the persist-then-read
split used by eval/calibrate_retrieval.py (a live diagnostic) versus
eval/metrics.py (pure functions over persisted raw results) elsewhere in
this repo.

Usage (from repo root, using the repo's venv):

    venv/bin/python -m eval.memory_warrant
    venv/bin/python -m eval.memory_warrant --out eval/results/memory_warrant.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory.ingest import EMBEDDING_MODEL
from memory.store import SCORE_THRESHOLD, search_past_incidents

REPO_ROOT = Path(__file__).resolve().parent.parent
TICKETS_PATH = REPO_ROOT / "data" / "tickets.json"
PAST_INCIDENTS_PATH = REPO_ROOT / "data" / "past_incidents.json"
DEFAULT_OUT_PATH = REPO_ROOT / "eval" / "results" / "memory_warrant.json"

SCHEMA_VERSION = 1


def classify(gold_root_cause: str | None, status: str, top_resolved_root_cause: str | None) -> str:
    """Pure classification given the three inputs the docstring above
    depends on. Kept separate from the chroma-calling loop below so it can
    be unit-tested against synthetic inputs without any index."""
    if gold_root_cause is None:
        return "hazard" if status == "ok" else "no_signal"
    if status == "ok" and top_resolved_root_cause == gold_root_cause:
        return "warranted"
    return "no_signal"


def classify_ticket(ticket: dict) -> dict:
    """Run search_past_incidents live for one ticket and return its
    persisted row (see module docstring for the field list)."""
    result = search_past_incidents(ticket["ticket_text"])
    status = result["status"]
    incidents = (result.get("data") or {}).get("incidents") or []
    top = incidents[0] if incidents else None

    top_incident_id = top.get("incident_id") if top else None
    top_resolved_root_cause = top.get("resolved_root_cause") if top else None
    top_score = top.get("similarity_score") if top else None

    gold_root_cause = ticket.get("gold_root_cause")
    classification = classify(gold_root_cause, status, top_resolved_root_cause)

    return {
        "id": ticket["id"],
        "category": ticket.get("category"),
        "gold_root_cause": gold_root_cause,
        "memory_status": status,
        "top_incident_id": top_incident_id,
        "top_resolved_root_cause": top_resolved_root_cause,
        "top_score": top_score,
        "classification": classification,
    }


def build_warrant_report(tickets: list[dict]) -> dict:
    """Classify every main (non-adversarial) ticket and assemble the full
    persisted document, including the header block."""
    main_tickets = [t for t in tickets if t.get("category") != "adversarial"]
    rows = [classify_ticket(t) for t in main_tickets]

    past_incidents = (
        json.loads(PAST_INCIDENTS_PATH.read_text(encoding="utf-8"))
        if PAST_INCIDENTS_PATH.exists()
        else []
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "threshold": SCORE_THRESHOLD,
        "embedding_model": EMBEDDING_MODEL,
        "n_past_incidents": len(past_incidents),
        "n_tickets": len(rows),
        "tickets": rows,
    }


def summarize(report: dict) -> str:
    rows = report["tickets"]
    counts = {"warranted": 0, "hazard": 0, "no_signal": 0}
    for row in rows:
        counts[row["classification"]] += 1
    return (
        f"Classified {len(rows)} main tickets against {report['n_past_incidents']} past "
        f"incidents (threshold={report['threshold']}, model={report['embedding_model']}):\n"
        f"  warranted: {counts['warranted']}\n"
        f"  hazard:    {counts['hazard']}\n"
        f"  no_signal: {counts['no_signal']}"
    )


def main(out_path: Path) -> None:
    tickets = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))
    assert tickets, "data/tickets.json is empty"

    report = build_warrant_report(tickets)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(summarize(report))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify which main tickets warrant search_past_incidents, "
        "measured against the live memory corpus, and persist the result."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT_PATH),
        help=f"Output JSON path (default: {DEFAULT_OUT_PATH}).",
    )
    args = parser.parse_args()
    main(Path(args.out))
