"""Diagnostic script for WHY tickets that expected resolve_with_approval
ended up escalated instead.

This is a pure-analysis diagnostic. It makes NO API calls (no LLM, no
Chroma, nothing) and computes NO eval metrics (no recall, no accuracy,
no scoring against gold_* fields -- that is Part B / eval/report.py, not
this script). It only reads the raw per-ticket JSON files already written
by eval/run_benchmark.py to eval/results/raw/*.json and reconstructs the
TaskState for each one to evaluate, condition by condition, the SIX-part
stop-success gate in agent.orchestrator._can_resolve.

It does not reimplement that gate's logic or its thresholds -- it imports
the real CONFIDENCE_THRESHOLD, MIN_EVIDENCE_SOURCES, OBSERVATIONAL_TOOLS,
_observed_runbook_doc_ids, and _can_resolve from agent.orchestrator and
evaluates the same six conditions _can_resolve checks, independently and
without short-circuiting, against a TaskState reconstructed from each raw
result's `state` dict via TaskState.model_validate(...).

Note: TaskState.called_tool_signatures is `exclude=True` in the model dump,
so it is never present in the raw JSON and the reconstructed TaskState will
have it at its default (empty set). This is fine: _can_resolve only reads
state.confidence, state.evidence_sources, state.citations, and (via
_observed_runbook_doc_ids) state.trajectory -- it never reads
called_tool_signatures, so the missing field cannot affect the gate result
being diagnosed here.

Usage (from repo root, using the repo's venv):

    venv/bin/python -m eval.diagnose_underresolution
    venv/bin/python -m eval.diagnose_underresolution --raw-dir eval/results/raw
    venv/bin/python -m eval.diagnose_underresolution --examples 5
    venv/bin/python -m eval.diagnose_underresolution --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from agent.orchestrator import (
    CONFIDENCE_THRESHOLD,
    MIN_EVIDENCE_SOURCES,
    OBSERVATIONAL_TOOLS,
    _can_resolve,
    _observed_runbook_doc_ids,
)
from agent.state import TaskState

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "eval" / "results" / "raw"

CONDITION_NAMES = [
    "confidence",
    "evidence_count",
    "observational_source",
    "runbook_credited",
    "has_citations",
    "citations_grounded",
]


def _load_raw_files(raw_dir: Path) -> list[dict]:
    files = sorted(raw_dir.glob("*.json"))
    results = []
    for path in files:
        with open(path) as f:
            results.append(json.load(f))
    return results


def _eval_conditions(state: TaskState) -> dict[str, bool]:
    """Evaluate each of the six _can_resolve conditions independently,
    never short-circuiting on an earlier failure."""
    observed_doc_ids = _observed_runbook_doc_ids(state, [])
    return {
        "confidence": state.confidence >= CONFIDENCE_THRESHOLD,
        "evidence_count": len(state.evidence_sources) >= MIN_EVIDENCE_SOURCES,
        "observational_source": any(
            name in OBSERVATIONAL_TOOLS for name in state.evidence_sources
        ),
        "runbook_credited": "search_runbooks" in state.evidence_sources,
        "has_citations": len(state.citations) > 0,
        "citations_grounded": all(
            doc_id in observed_doc_ids for doc_id in state.citations
        ),
    }


def _search_runbooks_called_but_not_credited(state: TaskState) -> bool:
    """True if a search_runbooks tool_call appears anywhere in the
    trajectory but "search_runbooks" was never credited into
    evidence_sources -- the "no_confident_match" path, since
    _credit_evidence only credits observations with status "ok" and
    search_runbooks reports "no_confident_match" (not "ok") when it has
    no confident hit."""
    if "search_runbooks" in state.evidence_sources:
        return False
    for entry in state.trajectory:
        tool_call = entry.get("tool_call")
        if tool_call is not None and tool_call.get("name") == "search_runbooks":
            return True
    return False


def analyze(raw_dir: Path) -> dict:
    raw_results = _load_raw_files(raw_dir)

    crashed_count = 0
    skipped_not_selected = 0
    analyzed: list[dict] = []
    disagreement_count = 0
    disagreements: list[str] = []

    for result in raw_results:
        ticket_id = result.get("ticket_id")
        run_info = result.get("run") or {}
        state_dump = result.get("state")
        ticket_meta = result.get("ticket") or {}

        if run_info.get("runner_error") is not None or state_dump is None:
            crashed_count += 1
            continue

        if ticket_meta.get("expected_behavior") != "resolve_with_approval":
            skipped_not_selected += 1
            continue
        if state_dump.get("status") != "escalated":
            skipped_not_selected += 1
            continue

        state = TaskState.model_validate(state_dump)
        conditions = _eval_conditions(state)
        all_pass = all(conditions.values())
        actual = _can_resolve(state)

        if actual != all_pass:
            disagreement_count += 1
            disagreements.append(ticket_id)
            print(
                f"!!! LOUD WARNING: ticket {ticket_id} disagreement between "
                f"reconstructed six-condition check (all_pass={all_pass}) and "
                f"real _can_resolve(state)={actual}. This indicates either a "
                f"reconstruction bug in this script or drift between this "
                f"script and agent.orchestrator._can_resolve.",
                file=sys.stderr,
            )

        failed = [name for name, ok in conditions.items() if not ok]

        analyzed.append(
            {
                "ticket_id": ticket_id,
                "category": ticket_meta.get("category"),
                "confidence": state.confidence,
                "evidence_sources": list(state.evidence_sources),
                "evidence_sources_count": len(state.evidence_sources),
                "citations": list(state.citations),
                "iteration": state.iteration,
                "conditions": conditions,
                "failed_conditions": failed,
                "failure_signature": tuple(failed),
                "can_resolve_actual": actual,
                "all_pass_reconstructed": all_pass,
                "runbook_called_not_credited": _search_runbooks_called_but_not_credited(
                    state
                ),
            }
        )

    per_condition_fail_counts = {name: 0 for name in CONDITION_NAMES}
    for entry in analyzed:
        for name in entry["failed_conditions"]:
            per_condition_fail_counts[name] += 1

    exactly_one_failure = [e for e in analyzed if len(e["failed_conditions"]) == 1]
    exactly_one_breakdown = Counter(e["failed_conditions"][0] for e in exactly_one_failure)

    confidence_alone = sum(
        1 for e in analyzed if e["failed_conditions"] == ["confidence"]
    )
    evidence_count_alone = sum(
        1 for e in analyzed if e["failed_conditions"] == ["evidence_count"]
    )
    both_confidence_and_evidence = sum(
        1
        for e in analyzed
        if set(e["failed_conditions"]) == {"confidence", "evidence_count"}
    )
    observational_requirement_failures = sum(
        1 for e in analyzed if "observational_source" in e["failed_conditions"]
    )

    signature_counter = Counter(e["failure_signature"] for e in analyzed)
    signature_ranked = signature_counter.most_common()

    confidences = [e["confidence"] for e in analyzed]
    if confidences:
        sorted_conf = sorted(confidences)
        confidence_distribution = {
            "min": sorted_conf[0],
            "p25": statistics.quantiles(sorted_conf, n=4)[0]
            if len(sorted_conf) >= 2
            else sorted_conf[0],
            "median": statistics.median(sorted_conf),
            "p75": statistics.quantiles(sorted_conf, n=4)[2]
            if len(sorted_conf) >= 2
            else sorted_conf[0],
            "max": sorted_conf[-1],
        }
    else:
        confidence_distribution = None

    runbook_called_not_credited_count = sum(
        1 for e in analyzed if e["runbook_called_not_credited"]
    )

    summary = {
        "total_raw_files": len(raw_results),
        "crashed_count": crashed_count,
        "skipped_not_selected": skipped_not_selected,
        "analyzed_count": len(analyzed),
        "disagreement_count": disagreement_count,
        "disagreements": disagreements,
        "per_condition_fail_counts": per_condition_fail_counts,
        "exactly_one_failure_count": len(exactly_one_failure),
        "exactly_one_failure_breakdown": dict(exactly_one_breakdown),
        "confidence_alone": confidence_alone,
        "evidence_count_alone": evidence_count_alone,
        "both_confidence_and_evidence": both_confidence_and_evidence,
        "observational_requirement_failures": observational_requirement_failures,
        "failure_signature_ranked": [
            {"signature": list(sig), "count": count} for sig, count in signature_ranked
        ],
        "confidence_distribution": confidence_distribution,
        "runbook_called_not_credited_count": runbook_called_not_credited_count,
    }

    return {"analyzed": analyzed, "summary": summary}


def _print_ticket_detail(entry: dict) -> None:
    print(f"  ticket_id: {entry['ticket_id']}")
    print(f"  category: {entry['category']}")
    print(f"  confidence: {entry['confidence']}")
    print(
        f"  evidence_sources ({entry['evidence_sources_count']}): "
        f"{entry['evidence_sources']}"
    )
    print(f"  citations: {entry['citations']}")
    print(f"  iteration at escalation: {entry['iteration']}")
    print(f"  runbook_called_not_credited: {entry['runbook_called_not_credited']}")
    print("  conditions:")
    for name in CONDITION_NAMES:
        status = "PASS" if entry["conditions"][name] else "FAIL"
        print(f"    {name}: {status}")
    print()


def print_report(result: dict, examples: int) -> None:
    summary = result["summary"]
    analyzed = result["analyzed"]

    print("=" * 72)
    print("Under-resolution diagnosis: resolve_with_approval tickets that escalated")
    print("=" * 72)
    print(f"Total raw files: {summary['total_raw_files']}")
    print(f"Crashed tickets (skipped, but counted): {summary['crashed_count']}")
    print(f"Skipped (not resolve_with_approval+escalated): {summary['skipped_not_selected']}")
    print(f"Analyzed tickets: {summary['analyzed_count']}")
    print(f"Cross-check disagreements: {summary['disagreement_count']}")
    if summary["disagreements"]:
        print(f"  -> {summary['disagreements']}")
    print()

    print("Per-condition failure counts (a ticket can appear in multiple):")
    for name in CONDITION_NAMES:
        print(f"  {name}: {summary['per_condition_fail_counts'][name]}")
    print()

    print(f"Tickets failing exactly one condition: {summary['exactly_one_failure_count']}")
    for name, count in summary["exactly_one_failure_breakdown"].items():
        print(f"  {name}: {count}")
    print()

    print("Co-occurrence counts:")
    print(f"  confidence-alone: {summary['confidence_alone']}")
    print(f"  evidence-count-alone: {summary['evidence_count_alone']}")
    print(f"  both confidence+evidence-count: {summary['both_confidence_and_evidence']}")
    print(f"  observational-requirement failures (any): {summary['observational_requirement_failures']}")
    print()

    print("Failure signature tally (sorted by frequency):")
    for row in summary["failure_signature_ranked"]:
        sig = row["signature"] or ["(none -- all pass, unexpected)"]
        print(f"  {row['count']:3d}  {sig}")
    print()

    cd = summary["confidence_distribution"]
    if cd:
        print("Confidence distribution among failures:")
        print(
            f"  min={cd['min']:.3f} p25={cd['p25']:.3f} median={cd['median']:.3f} "
            f"p75={cd['p75']:.3f} max={cd['max']:.3f}"
        )
    else:
        print("Confidence distribution: no analyzed tickets")
    print()

    print(
        "search_runbooks called but NOT credited (no_confident_match path): "
        f"{summary['runbook_called_not_credited_count']}"
    )
    print()

    if summary["failure_signature_ranked"]:
        top_signature = tuple(summary["failure_signature_ranked"][0]["signature"])
        top_bucket = [e for e in analyzed if e["failure_signature"] == top_signature]
        print(
            f"Detail for up to {examples} ticket(s) from the largest failure-"
            f"signature bucket {list(top_signature)} (n={len(top_bucket)}):"
        )
        print("-" * 72)
        for entry in top_bucket[:examples]:
            _print_ticket_detail(entry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory of raw per-ticket result JSON files (default: eval/results/raw)",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of example tickets to print in full detail (default: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout instead of the table",
    )
    args = parser.parse_args(argv)

    result = analyze(args.raw_dir)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result, args.examples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
