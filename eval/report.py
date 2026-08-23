"""Aggregate + render the offline benchmark's raw per-ticket results.

Standalone script:

    venv/bin/python -m eval.report
    venv/bin/python -m eval.report --raw-dir eval/results/raw --out-dir eval/results
    venv/bin/python -m eval.report --with-corpus-recall

This module computes NO scoring logic of its own -- every per-ticket
judgement (task success, known-issue bucketing, tool-use, state-tracking,
RAG metrics, cost) is delegated to the pure functions in eval/metrics.py and
the KnownIssue registry in eval/known_issues.py. This file only: loads raw
result files + data/tickets.json, calls those functions per ticket,
aggregates across tickets, runs the safety-suite subprocess, and renders
eval/results/report.md + eval/results/report.json.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

from eval import metrics
from eval.known_issues import find_known_issue

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "eval" / "results" / "raw"
DEFAULT_OUT_DIR = REPO_ROOT / "eval" / "results"
TICKETS_PATH = REPO_ROOT / "data" / "tickets.json"

# The denormalized-ticket fields written into each raw file by
# eval/run_benchmark.py -- checked against the live data/tickets.json entry
# to catch a raw sweep scored against stale/changed gold labels.
DENORM_FIELDS = [
    "category",
    "gold_root_cause",
    "gold_runbook_id",
    "required_tools",
    "expected_behavior",
    "min_confidence_evidence_sources",
]

CATEGORIES = ["easy", "multi_step", "tool_heavy", "rag_heavy", "ambiguous"]

# Rate cards this report knows about, keyed by a lowercase substring to look
# for in LLM_BASE_URL. Groq is the only provider this project currently has a
# real, on-file rate for; Baseten (and anything else) has NO on-file rate, so
# cost is rendered as UNCOMPUTED rather than silently reusing Groq's numbers
# -- see docs/design.md's efficiency spec and PROGRESS.md's cost-figure
# correction entry for why that distinction matters.
KNOWN_RATE_CARDS = {
    "groq": ("Groq (openai/gpt-oss-120b)", metrics.RateCard(input_per_mtok=0.15, output_per_mtok=0.60)),
    # Confirmed by the project owner on 2026-08-23: Baseten bills at the same
    # per-Mtok rates as the original Groq card. Do not assume this for any
    # other provider without a similar explicit confirmation -- see
    # docs/design.md's Session 9a efficiency note.
    "baseten": (
        "Baseten (openai/gpt-oss-120b, rates confirmed == Groq card by project owner on 2026-08-23)",
        metrics.RateCard(input_per_mtok=0.15, output_per_mtok=0.60),
    ),
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_tickets(tickets_path: Path = TICKETS_PATH) -> dict:
    tickets = json.loads(tickets_path.read_text())
    return {t["id"]: t for t in tickets}


def load_raw_results(raw_dir: Path) -> list[dict]:
    results = []
    for path in sorted(raw_dir.glob("*.json")):
        results.append(json.loads(path.read_text()))
    return results


def check_stale_denorm(raw: dict, live_ticket: dict | None) -> list[str]:
    """Return a list of human-readable mismatches between the raw file's
    denormalized `ticket` block and the live data/tickets.json entry for the
    same ticket id. Empty list if they agree (or live_ticket is missing,
    which is reported separately by the caller)."""
    if live_ticket is None:
        return []
    denorm = raw.get("ticket") or {}
    mismatches = []
    for field in DENORM_FIELDS:
        raw_val = denorm.get(field)
        live_val = live_ticket.get(field)
        if raw_val != live_val:
            mismatches.append(f"{field}: raw={raw_val!r} live={live_val!r}")
    return mismatches


# --------------------------------------------------------------------------
# Rate card selection
# --------------------------------------------------------------------------


def select_rate_card(base_url: str | None) -> tuple[str, object | None]:
    """Return (provider_label, RateCard-or-None) for the given LLM_BASE_URL.

    None means "no on-file rate card for this provider" -- callers must
    render cost as UNCOMPUTED, never fall back to a different provider's
    numbers."""
    if base_url:
        lowered = base_url.lower()
        for key, (label, card) in KNOWN_RATE_CARDS.items():
            if key in lowered:
                return label, card
    if not base_url:
        return "unknown (LLM_BASE_URL not set)", None
    return f"unknown provider (LLM_BASE_URL={base_url})", None


# --------------------------------------------------------------------------
# Safety gate
# --------------------------------------------------------------------------

_SUMMARY_LINE_RE = re.compile(
    r"(?:(?P<failed>\d+) failed)?"
    r"(?:.*?(?P<passed>\d+) passed)?"
    r"(?:.*?(?P<skipped>\d+) skipped)?"
)


def _parse_pytest_summary(output: str) -> dict:
    """Parse the last non-empty line of pytest -q output for pass/fail/skip
    counts. Returns zeros for any category not mentioned (pytest omits
    zero-count categories from its summary line)."""
    lines = [line for line in output.strip().splitlines() if line.strip()]
    last_line = lines[-1] if lines else ""

    def _count(word: str) -> int:
        m = re.search(rf"(\d+) {word}", last_line)
        return int(m.group(1)) if m else 0

    return {
        "passed": _count("passed"),
        "failed": _count("failed"),
        "skipped": _count("skipped"),
        "error": _count("error"),
        "summary_line": last_line,
    }


def run_safety_gate(python: str = "venv/bin/python") -> dict:
    """Run tests/test_safety.py as a subprocess and return a structured
    verdict. NEVER assumes/hardcodes a result -- if the subprocess itself
    cannot be run, returns status="UNKNOWN" with the reason."""
    try:
        proc = subprocess.run(
            [python, "-m", "pytest", "tests/test_safety.py", "-q"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "UNKNOWN",
            "reason": f"could not run the safety suite subprocess: {exc}",
            "passed": None,
            "failed": None,
            "skipped": None,
            "exit_code": None,
            "unauthorized_write_block_rate": "UNKNOWN",
            "skipped_note": None,
            "injection_block_rate": "NOT COMPUTED (adversarial ticket set not yet built -- see PROGRESS.md)",
        }

    counts = _parse_pytest_summary(proc.stdout + "\n" + proc.stderr)
    # Skips are NEVER folded into the enforced denominator -- a skipped test
    # asserts nothing, so counting it as either a pass or a fail would
    # misrepresent what was actually enforced. The enforced half is strictly
    # passed-over-(passed+failed); skips are reported separately, as their
    # own line item, and never move the status toward FAIL by themselves.
    enforced_total = counts["passed"] + counts["failed"]
    total_run = enforced_total + counts["skipped"] + counts["error"]

    if counts["failed"] > 0 or counts["error"] > 0:
        status = "FAIL"
    else:
        # unauthorized_write_block_rate is measured (this suite); injection
        # is not -- so status can never reach PASS until both halves exist.
        # Skips alone (e.g. a placeholder injection test) never cause FAIL.
        status = "PARTIAL"

    if counts["skipped"] > 0:
        skipped_note = f"{counts['skipped']} test skipped: injection placeholder (adversarial ticket set not built)"
    else:
        skipped_note = None

    return {
        "status": status,
        "reason": None,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "exit_code": proc.returncode,
        "total_run": total_run,
        "enforced_total": enforced_total,
        "unauthorized_write_block_rate": (
            f"{counts['passed']}/{enforced_total} enforced tests passed ({counts['failed']} failed)"
        ),
        "skipped_note": skipped_note,
        "injection_block_rate": "NOT COMPUTED (adversarial ticket set not yet built -- see PROGRESS.md)",
    }


# --------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round(pct * (len(ordered) - 1))
    return ordered[idx]


def _stat_block(values: list[float]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "p50": None, "p95": None, "n": 0}
    return {
        "mean": statistics.mean(clean),
        "p50": _percentile(clean, 0.50),
        "p95": _percentile(clean, 0.95),
        "n": len(clean),
    }


def _rate(numer: int, denom: int) -> float | None:
    if denom == 0:
        return None
    return numer / denom


# --------------------------------------------------------------------------
# Core aggregation
# --------------------------------------------------------------------------


def build_report(
    raw_dir: Path,
    tickets_by_id: dict,
    safety: dict,
    with_corpus_recall: bool = False,
    rate_card_provider: str | None = None,
) -> dict:
    raw_results = load_raw_results(raw_dir)

    rate_card_label, rate_card = select_rate_card(rate_card_provider)

    per_ticket = []
    stale_warnings = []
    missing_gold = []

    for raw in raw_results:
        ticket_id = raw.get("ticket_id")
        live_ticket = tickets_by_id.get(ticket_id)
        if live_ticket is None:
            missing_gold.append(ticket_id)
            live_ticket = raw.get("ticket") or {}
        else:
            mismatches = check_stale_denorm(raw, live_ticket)
            if mismatches:
                stale_warnings.append({"ticket_id": ticket_id, "mismatches": mismatches})

        state = raw.get("state")
        run = raw.get("run") or {}
        usage = raw.get("usage") or {}

        known_issue = find_known_issue(ticket_id)
        outcome = metrics.classify_outcome(raw, live_ticket, known_issue)

        success_status_only = metrics.task_success_status_only(state, live_ticket)
        success_strict = metrics.task_success_strict_lexical(state, live_ticket)

        tool_call_count = len(metrics.loop_tool_calls(state))
        cost = metrics.estimated_cost_usd(usage, rate_card) if rate_card is not None else None

        per_ticket.append(
            {
                "ticket_id": ticket_id,
                "category": live_ticket.get("category"),
                "started_at": run.get("started_at"),
                "crashed": bool(run.get("runner_error")),
                "runner_error": run.get("runner_error"),
                "bucket": outcome["bucket"],
                "known_issue_id": outcome["known_issue_id"],
                "documented_cause": outcome["documented_cause"],
                "task_success_status_only": success_status_only,
                "task_success_strict_lexical": success_strict,
                "tool_selection_accuracy": metrics.tool_selection_accuracy(state, live_ticket),
                "unnecessary_tool_calls": metrics.unnecessary_tool_calls(state, live_ticket),
                "parameter_validity": metrics.parameter_validity(state),
                "state_consistency": metrics.state_consistency(state),
                "write_gate_appended_correctly": metrics.write_gate_appended_correctly(state),
                "retrieval_recall_at_3_observed": metrics.retrieval_recall_at_3_observed(state, live_ticket),
                "citation_presence": metrics.citation_presence(state),
                "efficiency": {
                    "llm_call_count": usage.get("llm_call_count"),
                    "tool_call_count": tool_call_count,
                    "total_tokens_in": usage.get("total_tokens_in"),
                    "total_tokens_out": usage.get("total_tokens_out"),
                    "wall_clock_seconds": run.get("wall_clock_seconds"),
                    "estimated_cost_usd": cost,
                },
            }
        )

    n_total = len(per_ticket)
    n_crashed = sum(1 for t in per_ticket if t["crashed"])
    crashed_tickets = [t for t in per_ticket if t["crashed"]]

    # Task success -- over ALL tickets, crashed included (crashed counts as
    # failure for both measures, never silently excluded).
    n_status_success = sum(1 for t in per_ticket if t["task_success_status_only"])
    n_strict_success = sum(1 for t in per_ticket if t["task_success_strict_lexical"])
    status_rate = _rate(n_status_success, n_total)
    strict_rate = _rate(n_strict_success, n_total)
    gap_n = n_status_success - n_strict_success

    task_success = {
        "task_success_status_only": {"n_success": n_status_success, "n_total": n_total, "rate": status_rate},
        "task_success_strict_lexical": {"n_success": n_strict_success, "n_total": n_total, "rate": strict_rate},
        "hypothesis_semantic": "PENDING (Session 9b)",
        "gap_explanation": (
            f"{gap_n} ticket(s) reached the status-only success bar (status==resolved or "
            "correctly escalated) but failed the strict-lexical check -- their hypothesis text "
            "did not literally contain every significant token of gold_root_cause (see "
            "hypothesis_matches_gold's documented false-negative classes). This produces a "
            f"{(status_rate - strict_rate) * 100:.1f}-point gap between the two measures "
            f"({n_status_success}/{n_total} vs {n_strict_success}/{n_total})."
            if n_total and status_rate is not None and strict_rate is not None
            else "No tickets scored -- gap cannot be computed."
        ),
    }

    # Category breakdown.
    category_breakdown = {}
    for cat in CATEGORIES:
        cat_tickets = [t for t in per_ticket if t["category"] == cat]
        n_cat = len(cat_tickets)
        n_cat_status = sum(1 for t in cat_tickets if t["task_success_status_only"])
        n_cat_strict = sum(1 for t in cat_tickets if t["task_success_strict_lexical"])
        category_breakdown[cat] = {
            "n": n_cat,
            "task_success_status_only": {"n_success": n_cat_status, "rate": _rate(n_cat_status, n_cat)},
            "task_success_strict_lexical": {"n_success": n_cat_strict, "rate": _rate(n_cat_strict, n_cat)},
        }

    # Known issues / stale known issues / unexplained failures.
    known_issue_bucket = [t for t in per_ticket if t["bucket"] == "known_issue"]
    stale_known_issue_bucket = [t for t in per_ticket if t["bucket"] == "stale_known_issue"]
    unexplained_bucket = [t for t in per_ticket if t["bucket"] == "unexplained_failure"]

    # Efficiency aggregates.
    def _field_values(name: str) -> list[float]:
        return [t["efficiency"][name] for t in per_ticket if t["efficiency"][name] is not None]

    efficiency_agg = {
        field: _stat_block(_field_values(field))
        for field in (
            "llm_call_count",
            "tool_call_count",
            "total_tokens_in",
            "total_tokens_out",
            "wall_clock_seconds",
            "estimated_cost_usd",
        )
    }

    # Tool-use / state / RAG aggregates.
    tsa_values = [t["tool_selection_accuracy"] for t in per_ticket if t["tool_selection_accuracy"] is not None]
    unnecessary_total = sum(t["unnecessary_tool_calls"] for t in per_ticket)

    param_observed_failures = sum(t["parameter_validity"]["observed_failures"] for t in per_ticket)
    param_total_calls = sum(t["parameter_validity"]["total_calls"] for t in per_ticket)

    state_consistency_values = [t["state_consistency"] for t in per_ticket if not t["crashed"]]
    write_gate_values = [t["write_gate_appended_correctly"] for t in per_ticket if t["write_gate_appended_correctly"] is not None]

    retrieval_values = [t["retrieval_recall_at_3_observed"] for t in per_ticket if t["retrieval_recall_at_3_observed"] is not None]
    citation_values = [t["citation_presence"] for t in per_ticket if t["citation_presence"] is not None]

    tool_use = {
        "tool_selection_accuracy": {
            "mean": statistics.mean(tsa_values) if tsa_values else None,
            "n": len(tsa_values),
        },
        "unnecessary_tool_calls_total": unnecessary_total,
        "parameter_validity": {
            "observed_failures": param_observed_failures,
            "total_calls": param_total_calls,
            "measurable": False,
            "note": (
                "measurable=False by design (see eval/metrics.parameter_validity docstring) -- "
                "this is a raw failure count, NEVER a percentage, and is NOT a validated 100% "
                "figure."
            ),
        },
    }

    state_tracking = {
        "state_consistency": {
            "n_true": sum(1 for v in state_consistency_values if v),
            "n": len(state_consistency_values),
            "rate": _rate(sum(1 for v in state_consistency_values if v), len(state_consistency_values)),
        },
        "write_gate_appended_correctly": {
            "n_true": sum(1 for v in write_gate_values if v),
            "n": len(write_gate_values),
            "rate": _rate(sum(1 for v in write_gate_values if v), len(write_gate_values)),
            "note": "N/A tickets (never reached the write gate) excluded from n/rate.",
        },
    }

    rag = {
        "retrieval_recall_at_3_observed": {
            "n_true": sum(1 for v in retrieval_values if v),
            "n": len(retrieval_values),
            "rate": _rate(sum(1 for v in retrieval_values if v), len(retrieval_values)),
        },
        "citation_presence_rate": {
            "n_true": sum(1 for v in citation_values if v),
            "n": len(citation_values),
            "rate": _rate(sum(1 for v in citation_values if v), len(citation_values)),
        },
        "retrieval_recall_at_3_corpus": None,
    }

    if with_corpus_recall:
        from eval.calibrate_retrieval import run_runbook_calibration

        live_tickets = [tickets_by_id[t["ticket_id"]] for t in per_ticket if t["ticket_id"] in tickets_by_id]
        rows = run_runbook_calibration(live_tickets, top_k=3)
        n_corpus = len(rows)
        n_corpus_hit = sum(1 for r in rows if r["in_topk"])
        rag["retrieval_recall_at_3_corpus"] = {
            "n_true": n_corpus_hit,
            "n": n_corpus,
            "rate": _rate(n_corpus_hit, n_corpus),
            "note": (
                "queries with the ticket's own ticket_text (calibrate_retrieval's corpus view) -- "
                "measures a DIFFERENT thing than retrieval_recall_at_3_observed, which reflects "
                "the agent's own log-derived search_runbooks queries during the actual run."
            ),
        }

    # Provenance.
    started_ats = [t["started_at"] for t in per_ticket if t["started_at"]]
    provenance = {
        "n_tickets_scored": n_total,
        "earliest_started_at": min(started_ats) if started_ats else None,
        "latest_started_at": max(started_ats) if started_ats else None,
    }

    # Generated summary prose -- entirely interpolated from the numbers above.
    status_pct = f"{status_rate * 100:.1f}%" if status_rate is not None else "n/a"
    strict_pct = f"{strict_rate * 100:.1f}%" if strict_rate is not None else "n/a"
    summary = (
        f"This report scores {n_total} raw ticket result(s). task_success_status_only is "
        f"{n_status_success}/{n_total} ({status_pct}); task_success_strict_lexical is "
        f"{n_strict_success}/{n_total} ({strict_pct}). {len(known_issue_bucket)} ticket(s) fall "
        f"into documented known-issue buckets, {len(stale_known_issue_bucket)} known-issue "
        f"entries no longer reproduce (stale), and {len(unexplained_bucket)} failure(s) are "
        f"unexplained. {n_crashed} ticket(s) crashed. Safety gate status: {safety['status']}."
    )

    report = {
        "schema_version": 1,
        "provenance": provenance,
        "rate_card": {"provider": rate_card_label, "used": rate_card is not None},
        "safety": safety,
        "summary": summary,
        "task_success": task_success,
        "category_breakdown": category_breakdown,
        "known_issues": [
            {
                "ticket_id": t["ticket_id"],
                "known_issue_id": t["known_issue_id"],
                "documented_cause": t["documented_cause"],
            }
            for t in known_issue_bucket
        ],
        "stale_known_issues": [
            {
                "ticket_id": t["ticket_id"],
                "known_issue_id": t["known_issue_id"],
                "documented_cause": t["documented_cause"],
            }
            for t in stale_known_issue_bucket
        ],
        "unexplained_failures": [t["ticket_id"] for t in unexplained_bucket],
        "crashed": {
            "n": n_crashed,
            "tickets": [
                {"ticket_id": t["ticket_id"], "error_type": (t["runner_error"] or {}).get("type")}
                for t in crashed_tickets
            ],
        },
        "efficiency": efficiency_agg,
        "tool_use": tool_use,
        "state_tracking": state_tracking,
        "rag": rag,
        "stale_gold_warnings": stale_warnings,
        "missing_gold_tickets": missing_gold,
        "per_ticket": per_ticket,
    }
    return report


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fmt_rate(entry: dict) -> str:
    if entry.get("rate") is None:
        return "n/a"
    return f"{entry['n_success' if 'n_success' in entry else 'n_true']}/{entry.get('n_total', entry.get('n'))} ({entry['rate'] * 100:.1f}%)"


def _fmt_stat_block(block: dict, unit: str = "") -> str:
    if block["n"] == 0:
        return "n/a (no data)"
    return f"mean={block['mean']:.3f}{unit} p50={block['p50']:.3f}{unit} p95={block['p95']:.3f}{unit} (n={block['n']})"


def render_markdown(report: dict) -> str:
    lines = []
    lines.append("# IRCA Offline Benchmark Report")
    lines.append("")

    safety = report["safety"]
    lines.append(f"## ⚠️ SAFETY GATE: {safety['status']}")
    lines.append(f"- unauthorized_write_block_rate: {safety['unauthorized_write_block_rate']}")
    if safety.get("skipped_note"):
        lines.append(f"- {safety['skipped_note']}")
    lines.append(f"- injection_block_rate: {safety['injection_block_rate']}")
    if safety.get("reason"):
        lines.append(f"- reason: {safety['reason']}")
    lines.append("Do not report safety as fully passing until both halves are measured.")
    lines.append("")

    lines.append("## Summary")
    lines.append(report["summary"])
    lines.append("")

    prov = report["provenance"]
    lines.append("## Provenance")
    lines.append(f"- tickets scored: {prov['n_tickets_scored']}")
    lines.append(f"- earliest run.started_at: {prov['earliest_started_at']}")
    lines.append(f"- latest run.started_at: {prov['latest_started_at']}")
    lines.append(
        f"- rate card used: {report['rate_card']['provider']}"
        + ("" if report["rate_card"]["used"] else " -- cost figures UNCOMPUTED (no on-file rate card for this provider)")
    )
    if report["stale_gold_warnings"]:
        lines.append("")
        lines.append(
            f"**WARNING: {len(report['stale_gold_warnings'])} ticket(s) have a denormalized "
            "`ticket` block that disagrees with the live data/tickets.json labels:**"
        )
        for w in report["stale_gold_warnings"]:
            lines.append(f"  - {w['ticket_id']}: {'; '.join(w['mismatches'])}")
    if report["missing_gold_tickets"]:
        lines.append("")
        lines.append(
            f"**WARNING: {len(report['missing_gold_tickets'])} ticket(s) in the raw set have no "
            f"matching entry in data/tickets.json: {report['missing_gold_tickets']}**"
        )
    lines.append("")

    ts = report["task_success"]
    lines.append("## Task Success")
    a = ts["task_success_status_only"]
    b = ts["task_success_strict_lexical"]
    lines.append(f"- task_success_status_only: {a['n_success']}/{a['n_total']} ({(a['rate'] or 0) * 100:.1f}%)")
    lines.append(f"- task_success_strict_lexical: {b['n_success']}/{b['n_total']} ({(b['rate'] or 0) * 100:.1f}%)")
    lines.append(f"- hypothesis_semantic: {ts['hypothesis_semantic']}")
    lines.append(f"- gap: {ts['gap_explanation']}")
    lines.append("")

    lines.append("## Category Breakdown")
    lines.append("| category | n | status_only | strict_lexical |")
    lines.append("|---|---|---|---|")
    for cat in CATEGORIES:
        c = report["category_breakdown"][cat]
        so = c["task_success_status_only"]
        sl = c["task_success_strict_lexical"]
        so_str = "n/a" if so["rate"] is None else f"{so['n_success']}/{c['n']} ({so['rate'] * 100:.1f}%)"
        sl_str = "n/a" if sl["rate"] is None else f"{sl['n_success']}/{c['n']} ({sl['rate'] * 100:.1f}%)"
        lines.append(f"| {cat} | {c['n']} | {so_str} | {sl_str} |")
    lines.append("")

    lines.append("## Known Issues (documented)")
    if report["known_issues"]:
        for k in report["known_issues"]:
            lines.append(f"- {k['ticket_id']} [{k['known_issue_id']}]: {k['documented_cause']}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Stale Known Issues (no longer reproducing -- prune from eval/known_issues.py)")
    if report["stale_known_issues"]:
        for k in report["stale_known_issues"]:
            lines.append(f"- {k['ticket_id']} [{k['known_issue_id']}]: {k['documented_cause']}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## Unexplained Failures")
    if report["unexplained_failures"]:
        for tid in report["unexplained_failures"]:
            lines.append(f"- {tid}")
    else:
        lines.append("(none)")
    lines.append("")

    crashed = report["crashed"]
    lines.append("## Crashed Tickets")
    lines.append(f"- count: {crashed['n']}")
    if crashed["n"] > 0:
        for t in crashed["tickets"]:
            lines.append(f"  - {t['ticket_id']}: {t['error_type']}")
    lines.append("")

    eff = report["efficiency"]
    lines.append("## Efficiency")
    lines.append(f"(rate card: {report['rate_card']['provider']})")
    lines.append(f"- llm_call_count: {_fmt_stat_block(eff['llm_call_count'])}")
    lines.append(f"- tool_call_count: {_fmt_stat_block(eff['tool_call_count'])}")
    lines.append(f"- total_tokens_in: {_fmt_stat_block(eff['total_tokens_in'])}")
    lines.append(f"- total_tokens_out: {_fmt_stat_block(eff['total_tokens_out'])}")
    lines.append(f"- wall_clock_seconds: {_fmt_stat_block(eff['wall_clock_seconds'], 's')}")
    if report["rate_card"]["used"]:
        lines.append(f"- estimated_cost_usd: {_fmt_stat_block(eff['estimated_cost_usd'], '$')}")
    else:
        lines.append("- estimated_cost_usd: UNCOMPUTED (no on-file rate card for the serving provider)")
    lines.append("")

    tu = report["tool_use"]
    lines.append("## Tool Use")
    tsa = tu["tool_selection_accuracy"]
    tsa_str = "n/a" if tsa["mean"] is None else f"{tsa['mean'] * 100:.1f}% (n={tsa['n']})"
    lines.append(f"- tool_selection_accuracy (mean): {tsa_str}")
    lines.append(f"- unnecessary_tool_calls (total): {tu['unnecessary_tool_calls_total']}")
    pv = tu["parameter_validity"]
    lines.append(
        f"- parameter_validity: observed_failures={pv['observed_failures']} / "
        f"total_calls={pv['total_calls']} -- {pv['note']}"
    )
    lines.append("")

    st = report["state_tracking"]
    lines.append("## State Tracking")
    lines.append(f"- state_consistency: {_fmt_rate({**st['state_consistency'], 'n_total': st['state_consistency']['n']})}")
    lines.append(
        f"- write_gate_appended_correctly: "
        f"{_fmt_rate({**st['write_gate_appended_correctly'], 'n_total': st['write_gate_appended_correctly']['n']})} "
        f"({st['write_gate_appended_correctly']['note']})"
    )
    lines.append("")

    rag = report["rag"]
    lines.append("## RAG")
    lines.append(
        f"- retrieval_recall_at_3_observed: "
        f"{_fmt_rate({**rag['retrieval_recall_at_3_observed'], 'n_total': rag['retrieval_recall_at_3_observed']['n']})}"
    )
    lines.append(
        f"- citation_presence_rate: "
        f"{_fmt_rate({**rag['citation_presence_rate'], 'n_total': rag['citation_presence_rate']['n']})}"
    )
    if rag["retrieval_recall_at_3_corpus"] is None:
        lines.append("- retrieval_recall_at_3_corpus: NOT COMPUTED (run with --with-corpus-recall)")
    else:
        c = rag["retrieval_recall_at_3_corpus"]
        lines.append(
            f"- retrieval_recall_at_3_corpus: {_fmt_rate({**c, 'n_total': c['n']})} -- {c['note']}"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import os

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Aggregate eval/results/raw/*.json into report.md + report.json."
    )
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--with-corpus-recall",
        action="store_true",
        help="Also compute retrieval_recall_at_3_corpus via eval.calibrate_retrieval "
        "(requires a chroma index; slow). Default OFF.",
    )
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists() or not any(raw_dir.glob("*.json")):
        print(f"Error: no raw result files found in {raw_dir}", file=sys.stderr)
        return 2

    tickets_by_id = load_tickets()
    safety = run_safety_gate()
    base_url = os.environ.get("LLM_BASE_URL")

    report = build_report(
        raw_dir,
        tickets_by_id,
        safety=safety,
        with_corpus_recall=args.with_corpus_recall,
        rate_card_provider=base_url,
    )

    md_path = out_dir / "report.md"
    json_path = out_dir / "report.json"
    md_path.write_text(render_markdown(report))
    json_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"Wrote {md_path}", file=sys.stderr)
    print(f"Wrote {json_path}", file=sys.stderr)

    if safety["status"] == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
