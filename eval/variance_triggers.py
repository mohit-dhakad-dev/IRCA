"""Paired A/B harness for the memory-consultation triggers in
agent/orchestrator.py's _maybe_trigger_memory (see IRCA_MEMORY_TRIGGERS,
agent/orchestrator.py's _memory_triggers_enabled).

Context (measured, not re-derived here): in the full 63-ticket main sweep,
the nudge fired on exactly 5 tickets (T004, T014, T019, T059, T062 -- the
loop-guard tickets), 3 of 5 responded by consulting memory, and the
deterministic fallback never fired. That effect is real but too small to
separate from run-to-run variance in a single sweep. This harness runs a
paired A/B ("triggers_on" vs "triggers_off") with repeats over the same
ticket set, so variance is visible instead of averaged away.

This REUSES eval.run_benchmark.run_one for the actual run/persist logic
(never reimplements it) and eval.metrics for scoring (never reimplements
task-success logic). It writes ONLY under
eval/results/variance_triggers/<arm>/rep<N>/<TICKET>.json and
eval/results/variance_triggers/summary.json -- it must NEVER write into
eval/results/raw/, which is the main sweep's output directory.

This hits the REAL LLM API (via run_one -> run_agent_loop) once per
(arm, repeat, ticket) triple, and is therefore gated behind --live, exactly
like eval/run_benchmark.py.

Usage (from repo root, using the repo's venv):

    venv/bin/python -m eval.variance_triggers --live
    venv/bin/python -m eval.variance_triggers --live --tickets T004,T014 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import eval.run_benchmark as run_benchmark
from eval.metrics import loop_tool_calls, task_success_status_only

REPO_ROOT = Path(__file__).resolve().parent.parent
TICKETS_PATH = REPO_ROOT / "data" / "tickets.json"
VARIANCE_ROOT = REPO_ROOT / "eval" / "results" / "variance_triggers"

ARMS = ("triggers_on", "triggers_off")

DEFAULT_TICKETS = ["T004", "T014", "T019", "T059", "T062"]
DEFAULT_REPEATS = 5

_ENV_VAR = "IRCA_MEMORY_TRIGGERS"

# Arm -> value written to IRCA_MEMORY_TRIGGERS for the duration of that
# arm's runs. "triggers_on" clears the var entirely (relying on the
# documented default, which is ON), rather than setting an explicit "on"
# value, so the control arm exercises the actual default codepath.
_ARM_ENV_VALUE = {
    "triggers_on": None,
    "triggers_off": "0",
}


def _load_tickets_by_id() -> dict[str, dict]:
    tickets = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))
    return {t["id"]: t for t in tickets}


def _arm_out_dir(arm: str, repeat: int, out_root: Path) -> Path:
    return out_root / arm / f"rep{repeat}"


def _set_arm_env(arm: str) -> str | None:
    """Set IRCA_MEMORY_TRIGGERS for `arm` and return the PREVIOUS value (or
    None if it was unset), so the caller can restore it afterward."""
    previous = os.environ.get(_ENV_VAR)
    value = _ARM_ENV_VALUE[arm]
    if value is None:
        os.environ.pop(_ENV_VAR, None)
    else:
        os.environ[_ENV_VAR] = value
    return previous


def _restore_env(previous: str | None) -> None:
    if previous is None:
        os.environ.pop(_ENV_VAR, None)
    else:
        os.environ[_ENV_VAR] = previous


def _initiated_by_counts(state: dict | None) -> dict[str, int]:
    """search_past_incidents call counts this run, split by
    tool_call["initiated_by"] ("model" / "loop" / "unknown")."""
    counts = {"model": 0, "loop": 0, "unknown": 0}
    for entry in loop_tool_calls(state):
        tool_call = entry["tool_call"]
        if tool_call.get("name") != "search_past_incidents":
            continue
        provenance = tool_call.get("initiated_by")
        if provenance not in ("model", "loop"):
            counts["unknown"] += 1
        else:
            counts[provenance] += 1
    return counts


def run_ab(
    ticket_ids: list[str], repeats: int, out_root: Path = VARIANCE_ROOT
) -> dict:
    """Run every (arm, repeat, ticket) triple, persist each raw result via
    run_benchmark.run_one, and return the summary dict (also written to
    <out_root>/summary.json by the caller).

    Per-ticket per-repeat raw results land at
    <out_root>/<arm>/rep<N>/<TICKET>.json -- never eval/results/raw/.
    """
    tickets_by_id = _load_tickets_by_id()
    missing = [tid for tid in ticket_ids if tid not in tickets_by_id]
    if missing:
        raise ValueError(f"Unknown ticket id(s): {missing}")

    # per_ticket[arm][ticket_id] -> list of per-repeat records.
    per_ticket: dict[str, dict[str, list[dict]]] = {
        arm: {tid: [] for tid in ticket_ids} for arm in ARMS
    }

    for arm in ARMS:
        previous = _set_arm_env(arm)
        try:
            for repeat in range(1, repeats + 1):
                out_dir = _arm_out_dir(arm, repeat, out_root)
                out_dir.mkdir(parents=True, exist_ok=True)
                for tid in ticket_ids:
                    ticket = tickets_by_id[tid]
                    run_benchmark.run_one(ticket, out_dir)
                    raw = json.loads((out_dir / f"{tid}.json").read_text(encoding="utf-8"))
                    state = raw.get("state")
                    counts = _initiated_by_counts(state)
                    per_ticket[arm][tid].append(
                        {
                            "repeat": repeat,
                            "status": state.get("status") if state else None,
                            "memory_nudge_issued": bool(state.get("memory_nudge_issued")) if state else False,
                            "memory_autoconsulted": bool(state.get("memory_autoconsulted")) if state else False,
                            "consulted_by_model": counts["model"] > 0,
                            "consulted_by_loop": counts["loop"] > 0,
                            "consulted_any": (counts["model"] + counts["loop"] + counts["unknown"]) > 0,
                            "task_success_status_only": task_success_status_only(state, ticket),
                        }
                    )
        finally:
            _restore_env(previous)

    summary: dict = {"schema_version": 1, "repeats": repeats, "tickets": ticket_ids, "arms": {}}
    for arm in ARMS:
        arm_summary: dict = {"n_repeats": repeats, "per_ticket": {}}
        for tid in ticket_ids:
            records = per_ticket[arm][tid]
            status_counts: dict[str, int] = defaultdict(int)
            for r in records:
                status_counts[str(r["status"])] += 1
            arm_summary["per_ticket"][tid] = {
                "n_repeats": len(records),
                "memory_nudge_issued_count": sum(1 for r in records if r["memory_nudge_issued"]),
                "memory_autoconsulted_count": sum(1 for r in records if r["memory_autoconsulted"]),
                "consulted_by_model_count": sum(1 for r in records if r["consulted_by_model"]),
                "consulted_by_loop_count": sum(1 for r in records if r["consulted_by_loop"]),
                "consulted_any_count": sum(1 for r in records if r["consulted_any"]),
                "status_distribution": dict(status_counts),
                "task_success_status_only_count": sum(
                    1 for r in records if r["task_success_status_only"]
                ),
                "runs": records,
            }
        summary["arms"][arm] = arm_summary

    return summary


def compare_arms(summary: dict, warrant: dict | None = None) -> dict:
    """Side-by-side observed counts for the two arms, computed from the
    summary produced by run_ab / persisted at summary.json.

    Reports, per arm, across ALL repeats and tickets combined:
      - memory-consultation rate ("k of n runs" that consulted memory at all)
      - warranted-call count, IF `warrant` (eval/results/memory_warrant.json,
        already loaded) is supplied -- joined on ticket id's classification
        being "warranted". None if `warrant` is not supplied.
      - hazard-call count, same join, classification == "hazard". None if
        `warrant` is not supplied.
      - task-success count ("k of n runs")

    Deliberately reports ONLY observed counts, each as "k of n" with n
    (the total repeat*ticket run count) always visible alongside k -- NO
    p-value, no significance claim. With 5 tickets and a handful of repeats
    there is no statistical power for that, and this project does not dress
    up weak evidence as strong.
    """
    classification_by_id: dict[str, str] = {}
    if warrant is not None:
        for row in warrant.get("tickets", []):
            classification_by_id[row["id"]] = row["classification"]

    result: dict = {"repeats": summary["repeats"], "tickets": summary["tickets"], "arms": {}}
    for arm, arm_summary in summary["arms"].items():
        per_ticket = arm_summary["per_ticket"]
        n_total = sum(t["n_repeats"] for t in per_ticket.values())
        consulted_total = sum(t["consulted_any_count"] for t in per_ticket.values())
        success_total = sum(t["task_success_status_only_count"] for t in per_ticket.values())

        warranted_calls = None
        hazard_calls = None
        if warrant is not None:
            warranted_calls = sum(
                t["consulted_any_count"]
                for tid, t in per_ticket.items()
                if classification_by_id.get(tid) == "warranted"
            )
            hazard_calls = sum(
                t["consulted_any_count"]
                for tid, t in per_ticket.items()
                if classification_by_id.get(tid) == "hazard"
            )

        result["arms"][arm] = {
            "n_total_runs": n_total,
            "memory_consultation": f"{consulted_total} of {n_total}",
            "warranted_calls": (f"{warranted_calls} of {n_total}" if warranted_calls is not None else None),
            "hazard_calls": (f"{hazard_calls} of {n_total}" if hazard_calls is not None else None),
            "task_success": f"{success_total} of {n_total}",
        }

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Paired A/B harness (triggers on vs off) with repeats, "
        "for the memory-consultation triggers in agent/orchestrator.py."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Required to actually run. Hits the real LLM API for every (arm, repeat, ticket).",
    )
    parser.add_argument(
        "--tickets",
        type=str,
        default=",".join(DEFAULT_TICKETS),
        help=f"Comma-separated ticket ids (default: {','.join(DEFAULT_TICKETS)}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of repeats per arm (default: {DEFAULT_REPEATS}).",
    )
    args = parser.parse_args(argv)

    if not args.live:
        print(
            "Refusing to run: this hits the real LLM API for every (arm, repeat, ticket) "
            "triple and costs real tokens. Rerun with --live to proceed.",
            file=sys.stderr,
        )
        return 2

    ticket_ids = [t.strip() for t in args.tickets.split(",") if t.strip()]

    try:
        summary = run_ab(ticket_ids, args.repeats)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    VARIANCE_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = VARIANCE_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
