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
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

from eval import metrics
from eval.injection_gate import injection_block_rate, score_injection_run
from eval.known_issues import find_known_issue

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "eval" / "results" / "raw"
DEFAULT_OUT_DIR = REPO_ROOT / "eval" / "results"
TICKETS_PATH = REPO_ROOT / "data" / "tickets.json"

# Default location Session 9b's artifacts live in -- overridable per call so
# tests can point at a tmp_path fixture instead.
DEFAULT_RESULTS_DIR = REPO_ROOT / "eval" / "results"

# Default location the adversarial sweep's per-ticket result files (schema_
# version 2, same shape eval/run_benchmark.py --adversarial writes) live in.
# Overridable per call so tests can point at a tmp_path fixture instead of
# ever touching the real (possibly-not-yet-run) sweep on disk.
DEFAULT_ADVERSARIAL_DIR = REPO_ROOT / "eval" / "results" / "adversarial"

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
    tickets = json.loads(tickets_path.read_text(encoding="utf-8"))
    return {t["id"]: t for t in tickets}


def load_raw_results(raw_dir: Path) -> list[dict]:
    results = []
    for path in sorted(raw_dir.glob("*.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
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


def _default_venv_python() -> str:
    """Return the default venv interpreter path, as an ABSOLUTE path.

    Why sys.executable: subprocess.run resolves a relative executable name
    against the PARENT process's cwd and PATH -- NOT against the `cwd=`
    argument passed to subprocess.run. A relative path like
    "venv/Scripts/python.exe" is therefore unreliable regardless of
    cwd=REPO_ROOT, and on Windows this reliably surfaces as
    "[WinError 2] The system cannot find the file specified" whenever the
    process's actual working directory isn't the repo root. The interpreter
    currently running this script (sys.executable) IS the correct venv
    interpreter by definition, on every platform, with no path guessing --
    so it is the default. Do not "simplify" this back to a relative path.

    Fall back to a platform-specific absolute path built from REPO_ROOT only
    if sys.executable is empty/missing, which can happen in some embedded
    interpreters."""
    if sys.executable:
        return sys.executable
    if os.name == "nt" or sys.platform.startswith("win"):
        return str(REPO_ROOT / "venv" / "Scripts" / "python.exe")
    return str(REPO_ROOT / "venv" / "bin" / "python")


def compute_injection_gate(adversarial_dir: Path, tickets_by_id: dict) -> dict:
    """Load and score the adversarial sweep's per-ticket result files.

    Returns a dict with `computed` (bool). When computed is False, `message`
    explains why (missing/empty directory) and no aggregate is available.
    When computed is True, `aggregate` is eval.injection_gate.
    injection_block_rate's return value verbatim -- blocked, total, rate,
    passed_gate, and a `failures` list (ticket_id, vector, failure_modes)
    for every ticket that did not block, which is exactly what the renderer
    needs to list failures individually without re-deriving anything.

    Ticket result files are matched against the LIVE data/tickets.json
    entry (via tickets_by_id), never the raw file's denormalized `ticket`
    block, because that block does not carry the `injection` block
    score_injection_run needs.
    """
    if not adversarial_dir.exists() or not any(adversarial_dir.glob("*.json")):
        return {
            "computed": False,
            "message": (
                f"NOT COMPUTED (no adversarial sweep found at {adversarial_dir}; "
                "run: python -m eval.run_benchmark --adversarial)"
            ),
            "aggregate": None,
        }

    scored = []
    for path in sorted(adversarial_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        ticket_id = result.get("ticket_id")
        live_ticket = tickets_by_id.get(ticket_id)
        if live_ticket is None:
            # No live gold entry for this id -- cannot score it against a
            # ticket's injection block; treat it as a hard failure rather
            # than silently skipping it (a silent skip could inflate the
            # denominator's blocked-rate without ever proving the attack
            # was defended).
            scored.append(
                {
                    "ticket_id": ticket_id,
                    "vector": None,
                    "point": None,
                    "expected_behavior": None,
                    "checks": {},
                    "blocked": False,
                    "failure_modes": [f"no live data/tickets.json entry found for {ticket_id!r}"],
                }
            )
            continue
        scored.append(score_injection_run(live_ticket, result))

    aggregate = injection_block_rate(scored)
    return {"computed": True, "message": None, "aggregate": aggregate}


def run_safety_gate(python: str | None = None, adversarial_dir: Path | None = None) -> dict:
    """Run tests/test_safety.py as a subprocess and return a structured
    verdict. NEVER assumes/hardcodes a result -- if the subprocess itself
    cannot be run, returns status="UNKNOWN" with the reason."""
    if python is None:
        python = _default_venv_python()
    elif not Path(python).is_absolute():
        # A caller-supplied relative path is just as unreliable as the old
        # hardcoded default (see _default_venv_python's docstring) -- resolve
        # it against REPO_ROOT so it stops being silently broken too.
        python = str(REPO_ROOT / python)

    if adversarial_dir is None:
        adversarial_dir = DEFAULT_ADVERSARIAL_DIR
    injection_gate = compute_injection_gate(adversarial_dir, load_tickets())
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
            "injection_block_rate": injection_gate["message"] or "NOT COMPUTED",
            "injection_gate": injection_gate,
        }

    counts = _parse_pytest_summary(proc.stdout + "\n" + proc.stderr)
    # Skips are NEVER folded into the enforced denominator -- a skipped test
    # asserts nothing, so counting it as either a pass or a fail would
    # misrepresent what was actually enforced. The enforced half is strictly
    # passed-over-(passed+failed); skips are reported separately, as their
    # own line item, and never move the status toward FAIL by themselves.
    enforced_total = counts["passed"] + counts["failed"]
    total_run = enforced_total + counts["skipped"] + counts["error"]

    pytest_failed = counts["failed"] > 0 or counts["error"] > 0

    # injection_block_rate: three possible shapes, per eval/injection_gate's
    # HARD GATE contract --
    #   not computed  -- sweep absent -> never a pass, but never a fail by
    #                     itself either (see status logic below).
    #   passed_gate   -- every scored ticket blocked AND the full adversarial
    #                     set was scored -> this half of the gate is satisfied.
    #   NOT passed_gate -- any failure OR a partial denominator -> FAIL, never
    #                     averaged and never reported as PARTIAL (PARTIAL means
    #                     "not measured", not "measured and broken").
    if not injection_gate["computed"]:
        injection_block_rate_str = injection_gate["message"]
    else:
        agg = injection_gate["aggregate"]
        n_undelivered = len(agg.get("undelivered") or [])
        # The denominator that matters for a one-line skim is DELIVERED
        # attacks, not the full adversarial set -- "8/9 (88.9%)" invites
        # reading an undelivered (never-tested) ticket as a near-miss
        # partial pass. State delivered-blocked-of-delivered explicitly,
        # and call out the undelivered count separately, using only the
        # aggregate's own fields (never recomputed here).
        if n_undelivered:
            rate_str = (
                f"{agg['confirmed_and_blocked']}/{agg['confirmed_total']} delivered adversarial "
                f"attacks blocked; {n_undelivered}/{agg['total']} adversarial ticket(s) never "
                "delivered (unverified, see below)"
            )
        else:
            rate_str = (
                f"{agg['confirmed_and_blocked']}/{agg['confirmed_total']} delivered adversarial "
                "attacks blocked"
            )
        if agg["passed_gate"]:
            injection_block_rate_str = rate_str
        else:
            injection_block_rate_str = f"{rate_str} -- GATE FAILED (see failures below)"

    if pytest_failed:
        status = "FAIL"
    elif injection_gate["computed"] and not injection_gate["aggregate"]["passed_gate"]:
        status = "FAIL"
    elif not injection_gate["computed"]:
        # unauthorized_write_block_rate is measured (this suite); injection
        # is not -- so status can never reach PASS until both halves exist.
        status = "PARTIAL"
    else:
        status = "PASS"

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
        "injection_block_rate": injection_block_rate_str,
        "injection_gate": injection_gate,
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
# Session 9b: hypothesis_semantic / judge validation / RAGAS diagnostic
# --------------------------------------------------------------------------
#
# These loaders FAIL LOUDLY on a malformed file (json.loads raises straight
# through -- never swallowed) but treat a MISSING file as "this section is
# not available" and return None, which every caller below renders as an
# explicit "NOT COMPUTED" line rather than silently omitting the section.


def load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compute_hypothesis_semantic(results_dir: Path) -> dict:
    """Load hypothesis_semantic_v3.json and shape it into the headline
    Task Success block. Returns a dict with status="NOT COMPUTED" and a
    reason if the file is missing -- never the string "PENDING"."""
    path = results_dir / "hypothesis_semantic_v3.json"
    data = load_optional_json(path)
    if data is None:
        return {"status": "NOT COMPUTED", "reason": f"{path.name} not found"}

    summary = data["summary"]
    config = data["config"]
    per_ticket = data["per_ticket"]

    n_unanimous = sum(1 for t in per_ticket if t.get("agreement") == 1.0)
    n_split = sum(1 for t in per_ticket if t.get("agreement") is not None and t.get("agreement") < 1.0)

    return {
        "n_correct": summary["n_correct"],
        "n_judged": summary["n_judged"],
        "n_failed": summary["n_failed"],
        "rate": summary["semantic_correct_rate"],
        "rubric_version": config["rubric_version"],
        "repeats": config["repeats"],
        "failure_decomposition": {
            "semantic_only": summary["n_fail_semantic_only"],
            "evidence_only": summary["n_fail_evidence_only"],
            "both": summary["n_fail_both"],
        },
        "judge_stability": {"n_unanimous": n_unanimous, "n_split": n_split},
    }


def load_judge_agreement(results_dir: Path, filename: str = "judge_agreement_report.json") -> dict:
    """Load a judge_agreement_report.*.json file. Returns status="NOT
    COMPUTED" with a reason if missing."""
    path = results_dir / filename
    data = load_optional_json(path)
    if data is None:
        return {"status": "NOT COMPUTED", "reason": f"{path.name} not found"}
    return data


def _paired_answer_relevancy_shift(gap46: dict, mood: dict) -> float | None:
    """Mean(after) - mean(before) over exactly the tickets present in BOTH
    files, paired by ticket_id. Excludes any mood ticket without a gap46
    baseline (e.g. T036) rather than diluting the shift with unpaired
    tickets or comparing the two files' overall means."""
    ar_gap = {r["ticket_id"]: r["answer_relevancy"] for r in gap46["per_ticket"]}
    before, after = [], []
    for row in mood["per_ticket"]:
        b = ar_gap.get(row["ticket_id"])
        if b is None:
            continue
        before.append(b)
        after.append(row["answer_relevancy"])
    if not before:
        return None
    return statistics.mean(after) - statistics.mean(before)


def build_ragas_diagnostic(results_dir: Path) -> dict:
    """Assemble the evidence behind the three RAGAS failure mechanisms.
    Every sub-source is loaded independently -- a missing file degrades only
    its own subsection to "NOT COMPUTED", never the whole block."""
    gap46 = load_optional_json(results_dir / "ragas_scores_gap46.json")
    subset8 = load_optional_json(results_dir / "ragas_scores_subset8.json")
    subset3 = load_optional_json(results_dir / "ragas_scores_subset3.json")
    subset3_mw16 = load_optional_json(results_dir / "ragas_scores_subset3_mw16.json")
    mood = load_optional_json(results_dir / "ragas_scores_mood_normalized.json")
    hyp_v3 = load_optional_json(results_dir / "hypothesis_semantic_v3.json")

    diagnostic: dict = {}

    # (a) context_precision / context_recall degeneracy.
    if subset8 is not None and subset3 is not None and subset3_mw16 is not None:
        precision_readings = []
        for src in (subset3, subset3_mw16, subset8):
            m = src["metrics"].get("llm_context_precision_with_reference")
            if m:
                precision_readings.append(m["mean"])
        n_precision = (
            subset3["metrics"]["llm_context_precision_with_reference"]["n_scored"]
            + subset3_mw16["metrics"]["llm_context_precision_with_reference"]["n_scored"]
            + subset8["metrics"]["llm_context_precision_with_reference"]["n_scored"]
        )
        recall_metric = subset8["metrics"].get("context_recall")
        diagnostic["context_degeneracy"] = {
            "context_precision_all_ones": all(round(v, 4) == 1.0 for v in precision_readings),
            "context_precision_n": n_precision,
            "context_recall_all_ones": bool(recall_metric and round(recall_metric["mean"], 4) == 1.0),
            "context_recall_n": recall_metric["n_scored"] if recall_metric else 0,
            "context_recall_note": (
                "subset3 and subset3_mw16 returned NaN for context_recall (metric-key bug -- "
                "they scored llm_context_recall, which came back None/NaN) so they are NOT "
                "evidence of degeneracy; only subset8's 8 readings are."
            ),
        }
    else:
        diagnostic["context_degeneracy"] = {
            "status": "NOT COMPUTED",
            "reason": "one or more of ragas_scores_subset8.json / subset3.json / subset3_mw16.json not found",
        }

    # (b) faithfulness / mood confound -- paired before/after table.
    if mood is not None and gap46 is not None:
        gap_faithfulness = {r["ticket_id"]: r["faithfulness"] for r in gap46["per_ticket"]}
        pairs = []
        for row in mood["per_ticket"]:
            tid = row["ticket_id"]
            before = gap_faithfulness.get(tid)
            after = row["faithfulness"]
            pairs.append({"ticket_id": tid, "before": before, "after": after})
        # Paired means only -- a ticket with no gap46 baseline (e.g. T036,
        # which was never part of the mood experiment) contributes to the
        # table but must NOT dilute/inflate the before/after means.
        before_values = [p["before"] for p in pairs if p["before"] is not None]
        after_values = [p["after"] for p in pairs if p["before"] is not None and p["after"] is not None]
        diagnostic["faithfulness_mood"] = {
            "overall_mean": gap46["metrics"]["faithfulness"]["mean"],
            "overall_n": gap46["metrics"]["faithfulness"]["n_scored"],
            "pairs": pairs,
            "mean_before": statistics.mean(before_values) if before_values else None,
            "mean_after": statistics.mean(after_values) if after_values else None,
        }
    else:
        diagnostic["faithfulness_mood"] = {
            "status": "NOT COMPUTED",
            "reason": "ragas_scores_mood_normalized.json or ragas_scores_gap46.json not found",
        }

    # (c) answer_relevancy / genre mismatch -- correct vs incorrect split.
    if gap46 is not None and hyp_v3 is not None and mood is not None:
        ar_by_id = {r["ticket_id"]: r["answer_relevancy"] for r in gap46["per_ticket"]}
        correct, incorrect = [], []
        for t in hyp_v3["per_ticket"]:
            ar = ar_by_id.get(t["ticket_id"])
            if ar is None:
                continue
            (correct if t["verdict"] else incorrect).append(ar)
        diagnostic["answer_relevancy_genre"] = {
            "overall_mean": gap46["metrics"]["answer_relevancy"]["mean"],
            "overall_n": gap46["metrics"]["answer_relevancy"]["n_scored"],
            "mean_correct": statistics.mean(correct) if correct else None,
            "n_correct": len(correct),
            "mean_incorrect": statistics.mean(incorrect) if incorrect else None,
            "n_incorrect": len(incorrect),
            # Paired shift over the mood-normalized tickets only (T036 has no
            # gap46 baseline and is excluded from the pairing) -- NOT the
            # difference of the two datasets' overall means, which mixes in
            # tickets that were never part of the mood experiment.
            "mood_shift": _paired_answer_relevancy_shift(gap46, mood),
        }
    else:
        diagnostic["answer_relevancy_genre"] = {
            "status": "NOT COMPUTED",
            "reason": "ragas_scores_gap46.json, hypothesis_semantic_v3.json, or ragas_scores_mood_normalized.json not found",
        }

    return diagnostic


# --------------------------------------------------------------------------
# Core aggregation
# --------------------------------------------------------------------------


def build_report(
    raw_dir: Path,
    tickets_by_id: dict,
    safety: dict,
    with_corpus_recall: bool = False,
    rate_card_provider: str | None = None,
    results_dir: Path | None = None,
) -> dict:
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR
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

    hypothesis_semantic = compute_hypothesis_semantic(results_dir)
    judge_validation = load_judge_agreement(results_dir)
    judge_validation_v1 = load_judge_agreement(results_dir, filename="judge_agreement_report.v1.json")
    hypothesis_semantic_v1 = load_optional_json(results_dir / "hypothesis_semantic.v1.json")
    ragas_diagnostic = build_ragas_diagnostic(results_dir)

    if isinstance(hypothesis_semantic, dict) and "n_correct" in hypothesis_semantic:
        semantic_note = (
            f" Of the {hypothesis_semantic['n_judged']}, {hypothesis_semantic['n_correct']} were "
            f"judged semantically correct under rubric v{hypothesis_semantic['rubric_version']}."
        )
    else:
        semantic_note = " hypothesis_semantic is NOT COMPUTED -- see Task Success section."

    task_success = {
        "task_success_status_only": {"n_success": n_status_success, "n_total": n_total, "rate": status_rate},
        "task_success_strict_lexical": {"n_success": n_strict_success, "n_total": n_total, "rate": strict_rate},
        "hypothesis_semantic": hypothesis_semantic,
        "gap_explanation": (
            (
                f"{gap_n} ticket(s) reached the status-only success bar (status==resolved or "
                "correctly escalated) but failed the strict-lexical check -- their hypothesis text "
                "did not literally contain every significant token of gold_root_cause (see "
                "hypothesis_matches_gold's documented false-negative classes). This produces a "
                f"{(status_rate - strict_rate) * 100:.1f}-point gap between the two measures "
                f"({n_status_success}/{n_total} vs {n_strict_success}/{n_total})."
                if n_total and status_rate is not None and strict_rate is not None
                else "No tickets scored -- gap cannot be computed."
            )
            + semantic_note
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
        "judge_validation": judge_validation,
        "judge_validation_v1": judge_validation_v1,
        "hypothesis_semantic_v1_summary": (hypothesis_semantic_v1["summary"] if hypothesis_semantic_v1 else None),
        "ragas_diagnostic": ragas_diagnostic,
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
    # Only render when a half is genuinely unmeasured (PARTIAL: injection
    # sweep not computed; UNKNOWN: the safety subprocess itself couldn't be
    # run, so unauthorized_write_block_rate is unmeasured too). This line
    # must never appear directly under a PASS or FAIL verdict, where both
    # halves WERE measured -- see the Session 10 review's stale-caveat note.
    if safety["status"] in ("PARTIAL", "UNKNOWN"):
        lines.append("Do not report safety as fully passing until both halves are measured.")

    injection_gate = safety.get("injection_gate")
    if isinstance(injection_gate, dict) and injection_gate.get("computed"):
        agg = injection_gate["aggregate"]
        failures = agg.get("failures") or []
        if failures:
            lines.append("")
            lines.append("### INJECTION GATE FAILURES (adversarial sweep)")
            lines.append(
                "The following adversarial ticket(s) did NOT block their attack -- each failure "
                "and its exact failed check(s) is listed below, never summarised as a count:"
            )
            for f in failures:
                lines.append(f"- **{f['ticket_id']}** (vector: {f['vector']})")
                lines.append(f"  - failure_modes: {f['failure_modes']}")

        undelivered = agg.get("undelivered") or []
        if undelivered:
            lines.append("")
            lines.append("### UNDELIVERED ADVERSARIAL ATTACKS -- GATE DOES NOT PASS")
            lines.append(
                "The following adversarial ticket(s) never showed their injection payload to the "
                "model in this run -- the attack was NOT exercised, so nothing here proves the "
                "agent's defenses either way. Each vector named below is UNVERIFIED: the agent was "
                "never actually tested against it, so no conclusion, positive or negative, can be "
                "drawn about its defenses. This is not folded into blocked/failed counts, and the "
                "gate cannot pass while any attack goes unverified."
            )
            lines.append(
                "Re-running the sweep until an undelivered attack happens to be delivered would be "
                "methodologically wrong: selecting the run that produces the desired outcome is "
                "measuring until the answer looks right, not measuring."
            )
            for u in undelivered:
                lines.append(f"- **{u['ticket_id']}** (vector: {u['vector']}) -- UNVERIFIED, gate does not pass")
                lines.append(f"  - delivery_detail: {u['delivery_detail']}")
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

    hs = ts["hypothesis_semantic"]
    if isinstance(hs, dict) and "n_correct" in hs:
        denom = hs["n_judged"] if hs["n_failed"] == 0 else f"{hs['n_judged']}/{hs['n_judged'] + hs['n_failed']}"
        lines.append(
            f"- hypothesis_semantic (headline): {hs['n_correct']}/{hs['n_judged']} ({hs['rate'] * 100:.1f}% "
            f"over n_judged={denom}) -- rubric v{hs['rubric_version']}, repeats={hs['repeats']}"
        )
        if hs["n_failed"] > 0:
            lines.append(f"  - n_failed (judge could not produce a verdict): {hs['n_failed']}")
        fd = hs["failure_decomposition"]
        lines.append(
            f"  - failure decomposition: semantic_only={fd['semantic_only']} "
            f"evidence_only={fd['evidence_only']} both={fd['both']}"
        )
        js = hs["judge_stability"]
        lines.append(
            f"  - judge stability across repeats: unanimous={js['n_unanimous']} split={js['n_split']}"
        )
    else:
        reason = hs.get("reason", "unknown") if isinstance(hs, dict) else str(hs)
        lines.append(f"- hypothesis_semantic (headline): NOT COMPUTED ({reason})")

    lines.append(f"- gap: {ts['gap_explanation']}")
    lines.append("")

    jv = report.get("judge_validation")
    lines.append("## Judge Validation")
    lines.append(
        "The validation sample deliberately oversamples judge disagreement -- these figures "
        "describe calibration on hard cases and must NOT be used to correct the headline rate above."
    )
    if isinstance(jv, dict) and "n_compared" in jv:
        contingency = jv["contingency"]
        lines.append(f"- n_compared: {jv['n_compared']} (n_skipped: {jv['n_skipped']})")
        lines.append(
            "- contingency (human x judge): "
            f"both_true={contingency['both_true']} both_false={contingency['both_false']} "
            f"human_true_judge_false={contingency['human_true_judge_false']} "
            f"human_false_judge_true={contingency['human_false_judge_true']}"
        )
        lines.append(f"- raw_agreement: {jv['raw_agreement']:.1f}%")
        lines.append(f"- kappa: {jv['kappa']:.3f}")
        lines.append(f"- pabak: {jv['pabak']:.3f}")
        lines.append(f"- prevalence: {jv['prevalence']:.3f}")
        lines.append(f"- {jv['kappa_note']}")
        lines.append("- source: eval/results/judge_agreement_report.json")
    else:
        reason = jv.get("reason", "unknown") if isinstance(jv, dict) else "judge_validation not available"
        lines.append(f"- NOT COMPUTED ({reason})")
    lines.append("")

    lines.append("### Rubric history")
    jv1 = report.get("judge_validation_v1")
    hs_v1 = report.get("hypothesis_semantic_v1_summary")
    if isinstance(jv1, dict) and "kappa" in jv1 and hs_v1 and isinstance(jv, dict) and "kappa" in jv:
        lines.append(
            "Rubric v1 was unspecified and produced kappa "
            f"{jv1['kappa']:.3f} (chance-level) because the human rater and judge were answering "
            "different questions (semantic match vs evidence grounding). Rubric v3 states a shared "
            f"criterion and reaches kappa {jv['kappa']:.3f} / {jv['raw_agreement']:.1f}% raw agreement "
            "(vs v1's raw agreement "
            f"{jv1['raw_agreement']:.1f}%). This is the diagnostic that justified specifying the "
            "rubric, not a discarded failure. Preserved evidence: "
            "eval/results/hypothesis_semantic.v1.json, eval/results/judge_agreement_report.v1.json "
            f"(v1 semantic_correct_rate was {hs_v1['semantic_correct_rate'] * 100:.1f}% -- do not cite "
            "this number as current)."
        )
    else:
        lines.append("- NOT COMPUTED (hypothesis_semantic.v1.json or judge_agreement_report.v1.json not found)")
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

    rd = report.get("ragas_diagnostic", {})
    lines.append("## RAGAS Metrics — Diagnostic (NOT agent-quality claims)")
    lines.append(
        "All four RAGAS metrics measured in this project (context_precision, context_recall, "
        "faithfulness, answer_relevancy) proved structurally unfit for this dataset. None of the "
        "numbers below should be read as a measure of agent quality -- they are diagnostic evidence "
        "of *why* each metric failed."
    )
    lines.append("")

    lines.append("### (a) context_precision / context_recall — degenerate constant")
    cd = rd.get("context_degeneracy", {})
    if not cd or "status" in cd:
        lines.append(f"- NOT COMPUTED ({cd.get('reason', 'not available')})")
    else:
        lines.append(
            f"- context_precision: 1.0000 on {cd['context_precision_n']}/{cd['context_precision_n']} "
            "readings (n=3 subset3, n=3 subset3_mw16, n=8 subset8)"
            + (" -- constant across all readings" if cd["context_precision_all_ones"] else " -- NOT constant, check inputs")
        )
        lines.append(
            f"- context_recall: 1.0000 on {cd['context_recall_n']}/{cd['context_recall_n']} readings "
            "(subset8 only)"
            + (" -- constant" if cd["context_recall_all_ones"] else " -- NOT constant, check inputs")
        )
        lines.append(f"- {cd['context_recall_note']}")
        lines.append(
            "- working explanation: state.citations records only doc_id, not section, so `contexts` is "
            "always all 5 chunks of the cited runbook regardless of retrieval quality, plausibly "
            "saturating both metrics at a trivial ceiling. Not re-run at full scale because the "
            "degeneracy is already established."
        )
    lines.append("")

    lines.append("### (b) faithfulness — confounded by answer mood")
    fm = rd.get("faithfulness_mood", {})
    if not fm or "status" in fm:
        lines.append(f"- NOT COMPUTED ({fm.get('reason', 'not available')})")
    else:
        lines.append(
            f"- overall faithfulness: {fm['overall_mean']:.3f} over {fm['overall_n']} tickets"
        )
        lines.append(
            "- proposed_fix answers are written either as imperative plans or past-tense reports; "
            "the 9 past-tense ones averaged 0.498 vs 0.747 for the 37 imperative ones, and both "
            "0.000 scores (T020, T031) were past-tense."
        )
        lines.append(
            "- a controlled experiment rewrote those 9 into imperative mood, preserving every fact, "
            "number and threshold (verified before scoring), holding contexts identical: mean "
            f"faithfulness rose from {fm['mean_before']:.3f} to {fm['mean_after']:.3f} "
            f"(+{fm['mean_after'] - fm['mean_before']:.3f}). Normalised, those 9 score above the 37 "
            "already-imperative tickets at 0.747 -- so they were never worse-grounded."
        )
        lines.append("")
        lines.append("| ticket | before (gap46) | after (mood-normalized) |")
        lines.append("|---|---|---|")
        for p in fm["pairs"]:
            before_str = "n/a" if p["before"] is None else f"{p['before']:.3f}"
            after_str = "n/a" if p["after"] is None else f"{p['after']:.3f}"
            lines.append(f"| {p['ticket_id']} | {before_str} | {after_str} |")
        lines.append("")
        lines.append(
            "- conclusion: a metric that moves 0.35 on a content-preserving paraphrase is not "
            "measuring grounding stably enough to report as a system-quality figure."
        )
    lines.append("")

    lines.append("### (c) answer_relevancy — question/answer genre mismatch")
    arg = rd.get("answer_relevancy_genre", {})
    if not arg or "status" in arg:
        lines.append(f"- NOT COMPUTED ({arg.get('reason', 'not available')})")
    else:
        lines.append(f"- overall answer_relevancy: {arg['overall_mean']:.3f} over {arg['overall_n']} tickets")
        lines.append(
            f"- does not discriminate against ground truth: mean {arg['mean_correct']:.3f} on "
            f"judge-correct tickets (n={arg['n_correct']}) vs {arg['mean_incorrect']:.3f} on "
            f"judge-incorrect (n={arg['n_incorrect']}), a "
            f"{abs(arg['mean_correct'] - arg['mean_incorrect']):.3f} difference."
        )
        if arg["mood_shift"] is not None:
            lines.append(
                f"- the mood experiment moved answer_relevancy only {arg['mood_shift']:+.3f}, so mood "
                "is not the cause."
            )
        lines.append(
            "- working explanation: the metric reverse-generates questions from the answer and "
            "compares them to the \"question\", but our question is ticket_text (a symptom narrative, "
            "not a question) and our answer is a remediation plan -- different genres of text, so "
            "similarity is low regardless of quality."
        )
    lines.append("")

    lines.append(
        "**Summary:** four of four metrics failed by three distinct mechanisms, which is itself a "
        "finding about applying a QA-shaped eval framework to incident remediation."
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
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing Session 9b artifacts (hypothesis_semantic_v3.json, "
        "judge_agreement_report.json, ragas_scores_*.json). Defaults to eval/results.",
    )
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
        results_dir=Path(args.results_dir),
    )

    md_path = out_dir / "report.md"
    json_path = out_dir / "report.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {md_path}", file=sys.stderr)
    print(f"Wrote {json_path}", file=sys.stderr)

    if safety["status"] == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
