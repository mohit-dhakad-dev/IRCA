"""Retrieval calibration diagnostic for the two SCORE_THRESHOLD gates in this
repo: rag/retrieve.py (runbooks) and memory/store.py (past incidents).

This is a standalone diagnostic SCRIPT, not a pytest test. It measures how
well search_runbooks() ranks the correct runbook, and how well
search_past_incidents() ranks a matching prior incident, for the tickets in
data/tickets.json, and reports the numbers needed to judge whether the two
independent SCORE_THRESHOLD gates (0.5 for runbooks, 0.40 for memory) are
actually supported by data.

It does NOT modify SCORE_THRESHOLD in either module, or any other file. Any
change to those constants is a deliberate human decision made after reading
this script's output -- not something this script does for you.

Usage (from repo root, using the repo's venv):

    venv/bin/python -m eval.calibrate_retrieval
    venv/bin/python -m eval.calibrate_retrieval --layer runbooks --top-k 3 --rebuild
    venv/bin/python -m eval.calibrate_retrieval --layer memory
    venv/bin/python -m eval.calibrate_retrieval --layer both --rebuild

Flags:
    --top-k N     Size of the "official" top-k search reported in the per-
                   ticket rows and used for the Recall@top-k metric.
                   Defaults to 3 (matches rag.retrieve.DEFAULT_TOP_K and
                   memory.store.DEFAULT_TOP_K).
    --layer L     Which layer(s) to calibrate: "runbooks", "memory", or
                   "both" (default: "both").
    --rebuild     Force a clean rebuild of the relevant chroma index/indexes
                   before querying (rag.ingest.build_index() and/or
                   memory.ingest.build_index(), depending on --layer).
                   Without this flag, the script queries the existing
                   ./chroma_db index(es) and only falls back to building if
                   the search call reports status="error" (e.g. missing
                   collection).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from memory.ingest import build_index as build_memory_index
from memory.store import SCORE_THRESHOLD as MEMORY_SCORE_THRESHOLD
from memory.store import search_past_incidents
from rag.ingest import build_index as build_runbook_index
from rag.retrieve import SCORE_THRESHOLD as RUNBOOK_SCORE_THRESHOLD
from rag.retrieve import search_runbooks

REPO_ROOT = Path(__file__).resolve().parent.parent
TICKETS_PATH = REPO_ROOT / "data" / "tickets.json"

WIDE_TOP_K = 10
MEMORY_WIDE_TOP_K = 8
THRESHOLDS_TO_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# Watched cases: tickets whose retrieval behavior is known and recorded here
# so a regression (or drift) is loud and unmissable, not buried in the
# aggregate table. Add more entries as more cases are pinned down -- the
# printing code below does not need to change.
#
# T015 (runbooks) is a CONFIDENT WRONG retrieval: its top-1 score sits inside
# the normal correct-match score range, so no value of SCORE_THRESHOLD can
# ever reject it.
#
# T002 (memory) is the mirror-image failure: a CORRECT retrieval whose top-1
# score (0.3244) sits far BELOW even the lowered live memory gate (0.40), so
# the threshold suppresses it even though the gold root cause IS retrievable.
# No practical gate admits this match. Together T015 and T002 show a single
# threshold cannot be tuned to fix both failure modes at once: raising it
# does nothing for T015 (already inside the correct range), and lowering it
# far enough to admit T002 would let more wrong matches through elsewhere.
#
# See docs/design.md, "RAG + Memory — Retrieval Contract (Session 6 spec)"
# for the fuller note this cross-references.
WATCHED_CASES = {
    "T015": {
        "layer": "runbooks",
        "gold_doc_id": "RB-NETWORK-001",
        "observed_wrong_top1_doc_id": "RB-DB-001",
        "observed_top1_score": 0.5739,
        "observed_rank_of_gold": 4,
    },
    "T002": {
        "layer": "memory",
        "gold_root_cause": None,  # filled from data/tickets.json at run time
        "observed_top1_score": 0.3244,
        "observed_rank_of_gold": 1,
        "observed_status": "no_confident_match",
    },
}
WATCHED_SCORE_TOLERANCE = 0.05


def gold_doc_id(ticket: dict) -> str:
    """Strip the .md suffix from gold_runbook_id to get the bare doc_id
    used as chunk metadata (e.g. "RB-DB-001.md" -> "RB-DB-001")."""
    return Path(ticket["gold_runbook_id"]).stem


def rank_of_gold(chunks: list[dict], gold: str) -> int | None:
    """1-based rank of the first chunk whose doc_id == gold, or None if
    gold never appears among the given chunks."""
    for i, chunk in enumerate(chunks, start=1):
        if chunk.get("doc_id") == gold:
            return i
    return None


def rank_of_matching_root_cause(incidents: list[dict], gold_root_cause: str) -> int | None:
    """1-based rank of the first incident whose resolved_root_cause equals
    gold_root_cause, or None if no incident in the given list matches.

    There are 8 incidents over 6 root causes (two root causes have 2
    incidents each), so more than one incident can legitimately be "the"
    right answer for a given ticket -- this deliberately checks the field
    value, not a single pinned incident_id."""
    for i, incident in enumerate(incidents, start=1):
        if incident.get("resolved_root_cause") == gold_root_cause:
            return i
    return None


def ensure_runbook_index_available() -> str:
    """Probe the existing runbook index with a throwaway query. If it errors
    (e.g. collection missing), build it. Returns a short description of
    which path was taken."""
    probe = search_runbooks("probe query to check index availability")
    if probe["status"] == "error":
        build_runbook_index()
        return "existing index unusable (status=error) -- built a fresh index"
    return "used existing ./chroma_db index"


def ensure_memory_index_available() -> str:
    """Same as ensure_runbook_index_available(), for the memory collection."""
    probe = search_past_incidents("probe query to check index availability")
    if probe["status"] == "error":
        build_memory_index()
        return "existing index unusable (status=error) -- built a fresh index"
    return "used existing ./chroma_db index"


def histogram(values: list[float], bucket: float = 0.1, width: int = 100) -> list[str]:
    """Text bar histogram of values in [0, 1], bucketed at `bucket`."""
    n_buckets = int(round(1.0 / bucket))
    counts = [0] * n_buckets
    for v in values:
        idx = min(int(v / bucket), n_buckets - 1)
        counts[idx] = counts[idx] + 1

    max_count = max(counts) if counts else 0
    bar_budget = min(40, width - 20)
    lines = []
    for i in range(n_buckets):
        lo = i * bucket
        hi = lo + bucket
        bar_len = 0 if max_count == 0 else round(counts[i] / max_count * bar_budget)
        bar = "#" * bar_len
        lines.append(f"  [{lo:.2f}, {hi:.2f}) {counts[i]:>2} {bar}")
    return lines


def fmt_stats(label: str, values: list[float]) -> str:
    if not values:
        return f"  {label}: (no tickets in this group)"
    values_sorted = sorted(values)
    return (
        f"  {label}: n={len(values_sorted)} "
        f"min={values_sorted[0]:.4f} median={statistics.median(values_sorted):.4f} "
        f"max={values_sorted[-1]:.4f}\n"
        f"    raw sorted: {[round(v, 4) for v in values_sorted]}"
    )


def ranges_overlap(a: list[float], b: list[float]) -> bool:
    if not a or not b:
        return False
    return not (min(a) > max(b) or min(b) > max(a))


def print_watched_cases(runbook_rows: list[dict], memory_rows: list[dict]) -> None:
    """Report observed-vs-recorded state for each entry in WATCHED_CASES.

    These are known, previously-investigated retrieval cases pinned to a
    recorded expectation so that a change in the corpus, chunking, or
    embedding model shows up as a loud CHANGED verdict here rather than
    quietly shifting numbers in the aggregate table above."""
    print("\n" + "=" * 100)
    print("WATCHED CASES")
    print("=" * 100)
    print(
        "\nThese tickets have known, previously-investigated retrieval behavior recorded in\n"
        "WATCHED_CASES (this file), one per retrieval layer. Each run re-measures them and\n"
        "reports CHANGED/UNCHANGED against the recorded values -- a CHANGED verdict means the\n"
        "corpus, chunking, or embedding model likely changed, and the recorded values need a\n"
        "deliberate update, not a silent one."
    )

    runbook_by_id = {r["id"]: r for r in runbook_rows}
    memory_by_id = {r["id"]: r for r in memory_rows}

    for ticket_id, recorded in WATCHED_CASES.items():
        print(f"\n--- {ticket_id} ({recorded['layer']}) ---")

        if recorded["layer"] == "runbooks":
            row = runbook_by_id.get(ticket_id)
            if row is None:
                print(f"  >>> {ticket_id} not found in this run's ticket set -- cannot compare.")
                continue

            observed_top1_doc_id = row["top1_doc_id"]
            observed_top1_score = row["top1_score"]
            observed_rank_of_gold = row["rank"]

            print(f"  gold doc_id:              {recorded['gold_doc_id']}")
            print(
                f"  top-1 doc_id:             recorded={recorded['observed_wrong_top1_doc_id']:<16} "
                f"observed={observed_top1_doc_id}"
            )
            print(
                f"  top-1 score:              recorded={recorded['observed_top1_score']:<16.4f} "
                f"observed={observed_top1_score:.4f}"
            )
            print(
                f"  rank of gold (wide top-{WIDE_TOP_K}): recorded={recorded['observed_rank_of_gold']!s:<9} "
                f"observed={observed_rank_of_gold!s}"
            )

            doc_id_changed = observed_top1_doc_id != recorded["observed_wrong_top1_doc_id"]
            rank_changed = observed_rank_of_gold != recorded["observed_rank_of_gold"]
            score_drift = abs(observed_top1_score - recorded["observed_top1_score"])
            score_changed = score_drift > WATCHED_SCORE_TOLERANCE

            changed = doc_id_changed or rank_changed or score_changed

            if changed:
                print(
                    "\n  " + "!" * 90 + "\n"
                    "  !!! WATCHED CASE CHANGED -- update WATCHED_CASES in eval/calibrate_retrieval.py\n"
                    "  !!! deliberately after reviewing why. A change here may mean the corpus, the\n"
                    "  !!! chunking logic, or the embedding model changed -- do not silently accept it.\n"
                    "  " + "!" * 90
                )
            else:
                print("\n  UNCHANGED -- matches recorded expectation within tolerance.")

            print(
                "\n  WHY THIS IS WATCHED: this is a confidently WRONG retrieval -- its top-1 score\n"
                "  lies INSIDE the normal correct-match score range, not below it. Therefore NO value\n"
                "  of SCORE_THRESHOLD can reject it: raising the threshold is not a fix for this case.\n"
                "  See docs/design.md, 'RAG + Memory — Retrieval Contract (Session 6 spec)' for the\n"
                "  fuller note."
            )

        elif recorded["layer"] == "memory":
            row = memory_by_id.get(ticket_id)
            if row is None:
                print(f"  >>> {ticket_id} not found in this run's ticket set -- cannot compare.")
                continue

            observed_top1_score = row["top1_score"]
            observed_rank_of_gold = row["rank"]
            observed_status = row["status"]

            print(f"  gold root cause:          {row['gold']}")
            print(
                f"  top-1 score:              recorded={recorded['observed_top1_score']:<16.4f} "
                f"observed={observed_top1_score:.4f}"
            )
            print(
                f"  rank of gold (wide top-{MEMORY_WIDE_TOP_K}): recorded={recorded['observed_rank_of_gold']!s:<9} "
                f"observed={observed_rank_of_gold!s}"
            )
            print(
                f"  live gate status:        recorded={recorded['observed_status']:<20} "
                f"observed={observed_status}"
            )

            rank_changed = observed_rank_of_gold != recorded["observed_rank_of_gold"]
            score_drift = abs(observed_top1_score - recorded["observed_top1_score"])
            score_changed = score_drift > WATCHED_SCORE_TOLERANCE
            status_changed = observed_status != recorded["observed_status"]

            changed = rank_changed or score_changed or status_changed

            if changed:
                print(
                    "\n  " + "!" * 90 + "\n"
                    "  !!! WATCHED CASE CHANGED -- update WATCHED_CASES in eval/calibrate_retrieval.py\n"
                    "  !!! deliberately after reviewing why. A change here may mean the corpus, the\n"
                    "  !!! incident set, or the embedding model changed -- do not silently accept it.\n"
                    "  " + "!" * 90
                )
            else:
                print("\n  UNCHANGED -- matches recorded expectation within tolerance.")

            print(
                f"\n  WHY THIS IS WATCHED: this is a CORRECT retrieval (the gold root cause IS the\n"
                f"  top-ranked incident) whose top-1 score (0.3244) is rejected even at the lowered\n"
                f"  live memory gate ({MEMORY_SCORE_THRESHOLD}) -- the live status is no_confident_match\n"
                f"  even though the match is right, and no practical gate admits it. This is the\n"
                f"  OPPOSITE failure mode from T015 (a WRONG retrieval no gate rejects, because its\n"
                f"  score is too high). Together, T015 and T002 prove a single gate cannot be tuned\n"
                f"  to fix both problems: raising it does not help T015, and lowering it far enough\n"
                f"  to admit T002 would only let more wrong matches through elsewhere.\n"
                f"  See docs/design.md, 'RAG + Memory — Retrieval Contract (Session 6 spec)' for the\n"
                f"  fuller note."
            )


def run_runbook_calibration(tickets: list[dict], top_k: int) -> list[dict]:
    """Measure runbook retrieval quality for every ticket that has a
    gold_runbook_id, returning one row per ticket."""
    rows = []
    for ticket in tickets:
        if not ticket.get("gold_runbook_id"):
            continue
        gold = gold_doc_id(ticket)
        text = ticket["ticket_text"]

        narrow = search_runbooks(text, top_k=top_k)
        wide = search_runbooks(text, top_k=WIDE_TOP_K)

        narrow_chunks = narrow["data"].get("chunks", [])
        wide_chunks = wide["data"].get("chunks", [])

        top1_doc_id = narrow_chunks[0]["doc_id"] if narrow_chunks else None
        top1_score = narrow_chunks[0]["score"] if narrow_chunks else 0.0
        top1_correct = top1_doc_id == gold

        rank = rank_of_gold(wide_chunks, gold)
        in_topk = rank is not None and rank <= top_k

        rows.append(
            {
                "id": ticket["id"],
                "difficulty": ticket["category"],
                "gold": gold,
                "rank": rank,
                "top1_score": top1_score,
                "top1_doc_id": top1_doc_id,
                "top1_correct": top1_correct,
                "in_topk": in_topk,
                "status": narrow["status"],
            }
        )
    return rows


def run_memory_calibration(tickets: list[dict], top_k: int) -> list[dict]:
    """Measure memory retrieval quality for every ticket that has a
    non-null gold_root_cause, returning one row per ticket.

    Ground truth is `resolved_root_cause`, not a specific incident_id --
    rank-of-gold means "rank of the first incident whose resolved_root_cause
    matches the ticket's gold_root_cause", since more than one incident can
    legitimately be correct for a given ticket."""
    rows = []
    for ticket in tickets:
        gold = ticket.get("gold_root_cause")
        if not gold:
            continue
        text = ticket["ticket_text"]

        narrow = search_past_incidents(text, top_k=top_k)
        wide = search_past_incidents(text, top_k=MEMORY_WIDE_TOP_K)

        narrow_incidents = narrow["data"].get("incidents", [])
        wide_incidents = wide["data"].get("incidents", [])

        top1_incident_id = narrow_incidents[0]["incident_id"] if narrow_incidents else None
        top1_root_cause = narrow_incidents[0]["resolved_root_cause"] if narrow_incidents else None
        top1_score = narrow_incidents[0]["similarity_score"] if narrow_incidents else 0.0
        top1_correct = top1_root_cause == gold

        rank = rank_of_matching_root_cause(wide_incidents, gold)
        in_topk = rank is not None and rank <= top_k

        rows.append(
            {
                "id": ticket["id"],
                "difficulty": ticket["category"],
                "gold": gold,
                "rank": rank,
                "top1_score": top1_score,
                "top1_incident_id": top1_incident_id,
                "top1_root_cause": top1_root_cause,
                "top1_correct": top1_correct,
                "in_topk": in_topk,
                "status": narrow["status"],
            }
        )
    return rows


def print_runbook_section(rows: list[dict], top_k: int) -> dict:
    """Print the full runbook calibration section and return summary stats
    used later by the side-by-side comparison and verdict."""
    print("\n" + "=" * 100)
    print("LAYER 1: RUNBOOKS -- rag/retrieve.py SCORE_THRESHOLD")
    print("=" * 100)
    print(f"\nLive SCORE_THRESHOLD in rag/retrieve.py: {RUNBOOK_SCORE_THRESHOLD}")
    print(f"Reported top-k for Recall@top-k: {top_k} (wide search for rank-of-gold: {WIDE_TOP_K})\n")

    print("-" * 100)
    header = (
        f"{'ID':<6}{'diff':<8}{'gold':<16}{'rank(wide)':<12}{'top1_score':<12}"
        f"{'top1_doc_id':<16}{'top1_ok':<9}{'status':<20}"
    )
    print(header)
    print("-" * 100)
    for r in rows:
        rank_str = "-" if r["rank"] is None else str(r["rank"])
        print(
            f"{r['id']:<6}{r['difficulty']:<8}{r['gold']:<16}{rank_str:<12}"
            f"{r['top1_score']:<12.4f}{str(r['top1_doc_id']):<16}{str(r['top1_correct']):<9}{r['status']:<20}"
        )

    n = len(rows)
    recall_at_1 = sum(1 for r in rows if r["top1_correct"]) / n
    recall_at_k = sum(1 for r in rows if r["in_topk"]) / n

    print(f"\nRecall@1: {recall_at_1:.4f} ({sum(1 for r in rows if r['top1_correct'])}/{n})")
    print(f"Recall@{top_k}: {recall_at_k:.4f} ({sum(1 for r in rows if r['in_topk'])}/{n})")

    print("\nTop-1 score histogram (all tickets):")
    for line in histogram([r["top1_score"] for r in rows]):
        print(line)

    correct_scores = [r["top1_score"] for r in rows if r["top1_correct"]]
    wrong_scores = [r["top1_score"] for r in rows if not r["top1_correct"]]

    print("\nTop-1 score separation: CORRECT vs WRONG top-1 retrievals")
    print(fmt_stats("CORRECT top-1", correct_scores))
    print(fmt_stats("WRONG top-1", wrong_scores))

    overlap = ranges_overlap(correct_scores, wrong_scores)
    if overlap:
        print(
            "\n  >>> The CORRECT and WRONG score ranges OVERLAP. No single threshold "
            "cleanly separates them."
        )
    elif correct_scores and wrong_scores:
        print("\n  >>> The CORRECT and WRONG score ranges do NOT overlap in this sample.")
    else:
        print("\n  >>> One of the two groups is empty; separation is not meaningful here.")

    print("\nThreshold sweep (status=ok admission, top-1 per ticket):")
    print(f"{'thresh':<10}{'admitted':<12}{'admitted_correct':<20}{'correct_wrongly_rejected':<26}")
    for t in THRESHOLDS_TO_SWEEP:
        admitted = [r for r in rows if r["top1_score"] >= t]
        admitted_correct = [r for r in admitted if r["top1_correct"]]
        correct_rejected = [r for r in rows if r["top1_correct"] and r["top1_score"] < t]
        print(
            f"{t:<10.2f}{len(admitted):<12}{len(admitted_correct):<20}{len(correct_rejected):<26}"
        )

    n_no_confident = sum(1 for r in rows if r["status"] == "no_confident_match")
    print(
        f"\nTickets with status=no_confident_match at live threshold "
        f"{RUNBOOK_SCORE_THRESHOLD}: {n_no_confident}/{n}"
    )

    return {
        "n": n,
        "recall_at_1": recall_at_1,
        "recall_at_k": recall_at_k,
        "correct_scores": correct_scores,
        "wrong_scores": wrong_scores,
        "overlap": overlap,
        "n_no_confident": n_no_confident,
    }


def print_memory_section(rows: list[dict], top_k: int) -> dict:
    """Print the full memory calibration section and return summary stats
    used later by the side-by-side comparison and verdict."""
    print("\n" + "=" * 100)
    print("LAYER 2: MEMORY (past incidents) -- memory/store.py SCORE_THRESHOLD")
    print("=" * 100)
    print(f"\nLive SCORE_THRESHOLD in memory/store.py: {MEMORY_SCORE_THRESHOLD}")
    print(
        f"Reported top-k for Recall@top-k: {top_k} (wide search for rank-of-gold: {MEMORY_WIDE_TOP_K})\n"
    )
    print(
        "Ground truth here is `resolved_root_cause`, not a single pinned incident_id: with 8\n"
        "incidents over 6 root causes, two root causes have 2 legitimately-correct incidents\n"
        "each. rank(wide) below is the rank of the FIRST incident whose resolved_root_cause\n"
        "matches the ticket's gold_root_cause.\n"
    )

    print("-" * 100)
    header = (
        f"{'ID':<6}{'diff':<8}{'gold':<20}{'rank(wide)':<12}{'top1_score':<12}"
        f"{'top1_incident':<16}{'top1_cause':<20}{'top1_ok':<9}{'status':<20}"
    )
    print(header)
    print("-" * 100)
    for r in rows:
        rank_str = "-" if r["rank"] is None else str(r["rank"])
        print(
            f"{r['id']:<6}{r['difficulty']:<8}{r['gold']:<20}{rank_str:<12}"
            f"{r['top1_score']:<12.4f}{str(r['top1_incident_id']):<16}"
            f"{str(r['top1_root_cause']):<20}{str(r['top1_correct']):<9}{r['status']:<20}"
        )

    n = len(rows)
    recall_at_1 = sum(1 for r in rows if r["top1_correct"]) / n
    recall_at_k = sum(1 for r in rows if r["in_topk"]) / n

    print(f"\nRecall@1: {recall_at_1:.4f} ({sum(1 for r in rows if r['top1_correct'])}/{n})")
    print(f"Recall@{top_k}: {recall_at_k:.4f} ({sum(1 for r in rows if r['in_topk'])}/{n})")

    print("\nTop-1 score histogram (all tickets with a gold_root_cause):")
    for line in histogram([r["top1_score"] for r in rows]):
        print(line)

    correct_scores = [r["top1_score"] for r in rows if r["top1_correct"]]
    wrong_scores = [r["top1_score"] for r in rows if not r["top1_correct"]]

    print("\nTop-1 score separation: CORRECT vs WRONG top-1 retrievals")
    print(fmt_stats("CORRECT top-1", correct_scores))
    print(fmt_stats("WRONG top-1", wrong_scores))

    overlap = ranges_overlap(correct_scores, wrong_scores)
    if overlap:
        print(
            "\n  >>> The CORRECT and WRONG score ranges OVERLAP. No single threshold "
            "cleanly separates them."
        )
    elif correct_scores and wrong_scores:
        print("\n  >>> The CORRECT and WRONG score ranges do NOT overlap in this sample.")
    else:
        print("\n  >>> One of the two groups is empty; separation is not meaningful here.")

    print("\nThreshold sweep (status=ok admission, top-1 per ticket):")
    print(f"{'thresh':<10}{'admitted':<12}{'admitted_correct':<20}{'correct_wrongly_rejected':<26}")
    for t in THRESHOLDS_TO_SWEEP:
        admitted = [r for r in rows if r["top1_score"] >= t]
        admitted_correct = [r for r in admitted if r["top1_correct"]]
        correct_rejected = [r for r in rows if r["top1_correct"] and r["top1_score"] < t]
        print(
            f"{t:<10.2f}{len(admitted):<12}{len(admitted_correct):<20}{len(correct_rejected):<26}"
        )

    n_no_confident = sum(1 for r in rows if r["status"] == "no_confident_match")
    print(
        f"\nTickets with status=no_confident_match at live threshold "
        f"{MEMORY_SCORE_THRESHOLD}: {n_no_confident}/{n}"
    )

    return {
        "n": n,
        "recall_at_1": recall_at_1,
        "recall_at_k": recall_at_k,
        "correct_scores": correct_scores,
        "wrong_scores": wrong_scores,
        "overlap": overlap,
        "n_no_confident": n_no_confident,
    }


def print_side_by_side(runbook_stats: dict, memory_stats: dict) -> None:
    print("\n" + "=" * 100)
    print("SIDE-BY-SIDE: RUNBOOKS vs MEMORY")
    print("=" * 100)

    def fmt_stat(scores: list[float], stat: str) -> str:
        if not scores:
            return "n/a"
        if stat == "min":
            value = min(scores)
        elif stat == "max":
            value = max(scores)
        else:
            value = statistics.median(scores)
        return f"{value:.4f}"

    print(f"\n{'metric':<38}{'runbooks':<30}{'memory':<30}")
    print("-" * 98)
    print(
        f"{'Recall@1':<38}"
        f"{runbook_stats['recall_at_1']:<30.4f}"
        f"{memory_stats['recall_at_1']:<30.4f}"
    )
    print(
        f"{'Recall@top-k':<38}"
        f"{runbook_stats['recall_at_k']:<30.4f}"
        f"{memory_stats['recall_at_k']:<30.4f}"
    )
    for stat, label in (("min", "CORRECT top-1 score (min)"), ("median", "CORRECT top-1 score (median)"), ("max", "CORRECT top-1 score (max)")):
        print(
            f"{label:<38}"
            f"{fmt_stat(runbook_stats['correct_scores'], stat):<30}"
            f"{fmt_stat(memory_stats['correct_scores'], stat):<30}"
        )
    print(
        f"{'Live gate value':<38}"
        f"{RUNBOOK_SCORE_THRESHOLD:<30.2f}"
        f"{MEMORY_SCORE_THRESHOLD:<30.2f}"
    )
    print(
        f"{'Rejected by live gate':<38}"
        f"{str(runbook_stats['n_no_confident']) + '/' + str(runbook_stats['n']):<30}"
        f"{str(memory_stats['n_no_confident']) + '/' + str(memory_stats['n']):<30}"
    )

    print(
        "\nThe two layers' CORRECT top-1 scores sit in structurally different ranges: runbook\n"
        "matches score far higher than memory matches. The two layers now run independent gates\n"
        f"({RUNBOOK_SCORE_THRESHOLD} for runbooks, {MEMORY_SCORE_THRESHOLD} for memory) because\n"
        "mirroring one layer's value onto the other was measured to reject correct matches: at\n"
        "0.5, memory rejected 4 of 13 correct top-1 matches, against 0/15 on runbooks. This is\n"
        "not noise -- a threshold calibrated on one collection does NOT transfer to the other.\n"
        "\n"
        "Likely reason: runbook chunks are long prose sections with substantial topical overlap\n"
        "to a ticket's wording, so a correct match tends to score high. Incident symptom_summary\n"
        "records are short paraphrases, and a ticket-to-incident match is symptom-paraphrase-to-\n"
        "symptom-paraphrase -- two independently-written short descriptions of the same\n"
        "underlying problem -- which scores lower under cosine/embedding similarity for the same\n"
        "degree of semantic closeness. Anyone reading only this comparison block should conclude:\n"
        "do not reuse rag/retrieve.py's SCORE_THRESHOLD value for memory/store.py (or vice versa)\n"
        "just because the two constants happen to share a name and a default."
    )


def main(top_k: int, rebuild: bool, layer: str) -> None:
    print("=" * 100)
    print("RETRIEVAL CALIBRATION -- rag/retrieve.py and memory/store.py SCORE_THRESHOLD")
    print("=" * 100)

    do_runbooks = layer in ("runbooks", "both")
    do_memory = layer in ("memory", "both")

    if rebuild:
        if do_runbooks:
            build_runbook_index()
        if do_memory:
            build_memory_index()
        index_path_note = f"--rebuild passed -- forced a fresh build_index() for: {layer}"
    else:
        notes = []
        if do_runbooks:
            notes.append(f"runbooks: {ensure_runbook_index_available()}")
        if do_memory:
            notes.append(f"memory: {ensure_memory_index_available()}")
        index_path_note = "; ".join(notes)
    print(f"\nIndex path taken: {index_path_note}")
    print(f"Layer(s) calibrated this run: {layer}")

    tickets = json.loads(TICKETS_PATH.read_text())
    assert len(tickets) == 15, f"expected 15 tickets, found {len(tickets)}"

    runbook_rows: list[dict] = []
    memory_rows: list[dict] = []
    runbook_stats: dict = {}
    memory_stats: dict = {}

    if do_runbooks:
        runbook_rows = run_runbook_calibration(tickets, top_k)
        runbook_stats = print_runbook_section(runbook_rows, top_k)

    if do_memory:
        memory_rows = run_memory_calibration(tickets, top_k)
        memory_stats = print_memory_section(memory_rows, top_k)
        # Fill in the recorded gold_root_cause for T002 from live ticket data,
        # rather than hard-coding the string twice.
        t002 = next((t for t in tickets if t["id"] == "T002"), None)
        if t002 is not None:
            WATCHED_CASES["T002"]["gold_root_cause"] = t002.get("gold_root_cause")

    if do_runbooks and do_memory:
        print_side_by_side(runbook_stats, memory_stats)

    print_watched_cases(runbook_rows, memory_rows)

    # --- verdict ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(
        "\nSample sizes are small (15 tickets over 6 runbooks; 13 tickets with a gold root cause\n"
        "over 8 incidents / 6 root causes). This is far too small a sample to justify moving\n"
        "either SCORE_THRESHOLD to any new precise-sounding value -- any such number would be\n"
        "overfit to these data points and could not be trusted to generalize. What follows is\n"
        "only what this specific run's numbers show, not a recommendation to hard-code a new\n"
        "constant from them."
    )

    if do_runbooks:
        print("\n--- Runbooks (rag/retrieve.py) ---")
        if runbook_stats["overlap"]:
            print(
                "The CORRECT and WRONG top-1 score distributions OVERLAP in this sample, which\n"
                "means no single threshold value can perfectly separate confident-correct\n"
                f"retrievals from confident-wrong ones here. {RUNBOOK_SCORE_THRESHOLD} is therefore\n"
                "NOT cleanly validated by this data -- it is an assumption that happens to not be\n"
                "contradicted (or is contradicted, depending on the sweep numbers above), not a\n"
                "value derived from measurement."
            )
        else:
            print(
                "In this specific sample, CORRECT and WRONG top-1 scores do not overlap, which is\n"
                "weak positive evidence that some threshold between them is workable -- but with\n"
                f"this few tickets it is not strong enough to promote {RUNBOOK_SCORE_THRESHOLD} (or\n"
                "any other specific value) from 'unmeasured guess' to 'validated constant'."
            )

    if do_memory:
        print("\n--- Memory (memory/store.py) ---")
        print(
            f"The live {MEMORY_SCORE_THRESHOLD} gate is already lower than the runbook layer's\n"
            f"{RUNBOOK_SCORE_THRESHOLD} precisely because mirroring {RUNBOOK_SCORE_THRESHOLD} onto\n"
            "memory was measured to reject several correct top-1 matches (status=no_confident_match)\n"
            "whose retrieved incident's resolved_root_cause is exactly the ticket's gold_root_cause --\n"
            "see the CORRECT vs WRONG score separation and the rejected-by-live-gate count above.\n"
            "But CORRECT and WRONG score ranges overlap here too -- a WRONG top-1 can score above the\n"
            "CORRECT median (e.g. a wrong top-1 at 0.5981 sitting above a correct median of 0.5719 in\n"
            "the reference measurement this script's threshold sweep should reproduce). So no\n"
            "threshold cleanly separates correct from wrong on this collection either; picking a\n"
            "lower cutoff trades false rejections for false admissions rather than eliminating the\n"
            "tradeoff.\n"
            "\n"
            "The practical implication is not 'pick a better number' -- it is that the gate alone\n"
            "cannot be the safeguard against a wrong memory match. The real safeguard is the\n"
            "retrieval contract's requirement to verify any suggested root cause independently\n"
            "via query_logs/query_metrics before adopting it, exactly as memory/store.py's own\n"
            "summary text already states. This script does not recommend a single precise\n"
            "replacement value for either SCORE_THRESHOLD."
        )

    print(
        "\nRecommendation: treat both SCORE_THRESHOLD constants as still unvalidated as precise\n"
        "cutoffs. If either needs tightening or loosening, do so only after collecting a\n"
        "materially larger and more diverse ticket/incident set -- do not hand-tune it to fit\n"
        "these rows, and do not assume the two constants should share a value just because they\n"
        "share a name and default."
    )
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnostic: measure retrieval quality against data/tickets.json to "
        "evaluate (not change) rag/retrieve.py's and memory/store.py's SCORE_THRESHOLD."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k for the 'official' reported search and Recall@top-k (default: 3).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force build_index() before querying, for a reproducible run from a clean checkout. "
        "Rebuilds both indexes when --layer is 'both' (the default).",
    )
    parser.add_argument(
        "--layer",
        choices=["runbooks", "memory", "both"],
        default="both",
        help="Which retrieval layer(s) to calibrate (default: both).",
    )
    args = parser.parse_args()
    main(top_k=args.top_k, rebuild=args.rebuild, layer=args.layer)
