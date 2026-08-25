"""Offline benchmark RUNNER: executes run_agent_loop() over data/tickets.json
and writes one raw result JSON file per ticket. This script computes NO
metrics -- it only captures raw agent output (final TaskState, timing, LLM
usage) for each ticket. Scoring against gold_* fields lives in eval/report.py
(Part B, not written yet).

This hits the REAL LLM API once per ticket (potentially several LLM calls per
ticket, since run_agent_loop loops). It costs real tokens and real money, and
is therefore gated behind --live, exactly like pytest.ini's `-m "not live"`
convention gates live tests behind an explicit opt-in.

Before running any ticket, any *.json result files already sitting in the
output dir (left over from a previous sweep) are MOVED -- not copied -- into
eval/results/runs/{UTC timestamp}/. This is deliberate: if stale files were
left in place, a later `--subset 5` run would overwrite only 5 of the files
in a 63-ticket out dir, silently leaving a mix of two different sweeps that
eval/report.py (Part B) would then score as though it were a single run.
Moving guarantees the raw dir always holds exactly one sweep's output at a
time, and nothing is lost -- it's archived, not deleted. Use --no-archive to
skip this when iterating cheaply and repeated overwrites in the same dir are
fine.

Usage (from repo root, using the repo's venv):

    venv/bin/python -m eval.run_benchmark --live
    venv/bin/python -m eval.run_benchmark --live --subset 5
    venv/bin/python -m eval.run_benchmark --live --tickets T001,T009
    venv/bin/python -m eval.run_benchmark --live --out eval/results/raw
    venv/bin/python -m eval.run_benchmark --live --no-archive

Flags:
    --live         REQUIRED. Without it, the script refuses to run (see
                   pytest.ini's `-m "not live"` for the same idea applied to
                   tests) and exits with code 2.
    --subset N     Run only the first N tickets (in data/tickets.json order).
                   Mutually exclusive with --tickets.
    --tickets T    Comma-separated list of ticket ids to run, e.g.
                   "T001,T009". Mutually exclusive with --subset.
    --out DIR      Output directory for the per-ticket result files.
                   Default: eval/results/raw. Created if missing.
    --no-archive   Skip archiving pre-existing *.json files in the out dir
                   before this sweep. Off by default -- archiving is the
                   safe default; this is only for cheap iteration.

Output: one file per ticket at {out}/{ticket_id}.json (pretty-printed,
indent=2). Tickets are run SEQUENTIALLY and each file is written immediately
after that ticket finishes, so a crash partway through a run still leaves
every completed ticket's result on disk. A per-ticket crash (an exception
raised out of run_agent_loop) is caught, recorded in that ticket's own result
file (state=null, run.runner_error populated), and does NOT abort the rest of
the run. The script's process exit code is 0 even if some tickets crashed;
crashes are data. Only a setup failure (missing --live, invalid flag
combination, unreadable tickets.json) causes a non-zero exit.

Again: this script computes NO metrics -- no recall, no accuracy, no
citation-correctness scoring. It only produces the raw per-ticket JSON files
that eval/report.py (Part B) will later read and score.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import agent.approval as approval
import agent.orchestrator as orchestrator

REPO_ROOT = Path(__file__).resolve().parent.parent
TICKETS_PATH = REPO_ROOT / "data" / "tickets.json"
DEFAULT_OUT_DIR = REPO_ROOT / "eval" / "results" / "raw"
ARCHIVE_ROOT = REPO_ROOT / "eval" / "results" / "runs"

TICKET_DENORM_FIELDS = [
    "category",
    "gold_root_cause",
    "gold_runbook_id",
    "required_tools",
    "expected_behavior",
    "min_confidence_evidence_sources",
]


class _UsageTracker:
    """Accumulates LLM call counts/token usage for a single ticket run.

    Not thread-safe and not meant to be -- tickets are run sequentially and
    a fresh tracker is used (and the patched symbol restored) per ticket.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.per_call: list[dict] = []

    def record(self, response) -> None:
        self.call_count += 1
        tokens_in = 0
        tokens_out = 0
        usage = getattr(response, "usage", None)
        if usage is not None:
            tokens_in = getattr(usage, "prompt_tokens", 0) or 0
            tokens_out = getattr(usage, "completion_tokens", 0) or 0
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.per_call.append({"in": tokens_in, "out": tokens_out})

    def as_dict(self) -> dict:
        return {
            "llm_call_count": self.call_count,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "per_call": self.per_call,
        }


@contextlib.contextmanager
def _usage_shim():
    """Monkeypatch agent.orchestrator.call_llm_with_tools with a wrapper
    that delegates to the real function and tracks usage as a side effect.

    orchestrator.py does `from agent.llm import call_llm_with_tools`, which
    binds the name directly into agent.orchestrator's module namespace --
    patching agent.llm.call_llm_with_tools would NOT be observed by the
    orchestrator's already-bound reference, so we must patch
    agent.orchestrator.call_llm_with_tools itself. There is only this one
    import path in the current orchestrator; if a second call site is added
    that imports the symbol independently, it must be patched here too.

    The wrapper is defensive: the real function returns an {"error": ...}
    dict on failure (no .usage to read), and even a successful response may
    lack a .usage attribute depending on provider. Either case is recorded
    as a call with 0 tokens, never raised into the agent loop.
    """
    tracker = _UsageTracker()
    real_fn = orchestrator.call_llm_with_tools

    def wrapper(*args, **kwargs):
        response = real_fn(*args, **kwargs)
        try:
            tracker.record(response)
        except Exception:
            # Never let usage bookkeeping break the agent loop.
            tracker.call_count += 1
            tracker.per_call.append({"in": 0, "out": 0})
        return response

    orchestrator.call_llm_with_tools = wrapper
    try:
        yield tracker
    finally:
        orchestrator.call_llm_with_tools = real_fn


def _load_tickets() -> list[dict]:
    return json.loads(TICKETS_PATH.read_text(encoding="utf-8"))


def _select_tickets(
    tickets: list[dict],
    subset: int | None,
    ticket_ids: list[str] | None,
    adversarial: bool = False,
) -> list[dict]:
    """Select which tickets to run.

    The DEFAULT selection (no --subset, --tickets, or --adversarial) must
    stay the original 63-ticket main sweep: category == "adversarial"
    tickets (T064-T072) are filtered out, so a default sweep's numbers are
    unchanged now that tickets.json holds 72 tickets. Adversarial tickets
    remain runnable when named explicitly via --tickets, and --adversarial
    selects exactly the adversarial category (in tickets.json order).
    --subset is applied to tickets.json order BEFORE any adversarial
    filtering would apply, i.e. it is mutually exclusive with --adversarial
    at the CLI level (see main()'s validation) so there is no ambiguity
    about which set --subset counts against.
    """
    if adversarial:
        return [t for t in tickets if t.get("category") == "adversarial"]
    if subset is not None:
        return tickets[:subset]
    if ticket_ids is not None:
        by_id = {t["id"]: t for t in tickets}
        selected = []
        for tid in ticket_ids:
            ticket = by_id.get(tid)
            if ticket is None:
                raise ValueError(f"Unknown ticket id in --tickets: {tid!r}")
            selected.append(ticket)
        return selected
    return [t for t in tickets if t.get("category") != "adversarial"]


def _denormalized_ticket(ticket: dict) -> dict:
    return {field: ticket.get(field) for field in TICKET_DENORM_FIELDS}


def run_one(ticket: dict, out_dir: Path) -> dict:
    """Run a single ticket, write its result file, and return a small
    summary dict used for the stderr progress line."""
    ticket_id = ticket["id"]
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    state_dump = None
    pending_action_dump = None
    runner_error = None
    start = time.perf_counter()
    with _usage_shim() as tracker:
        try:
            state = orchestrator.run_agent_loop(ticket_id)
            state_dump = state.model_dump()
            # Session 10 Step 2 follow-up: the injection-leak gate
            # (eval/injection_gate.py's no_secret_leak check) needs the
            # human-visible proposed_fix text, since it -- not the raw log
            # observation -- is the realistic surface a leaked canary would
            # land on. state.model_dump() only carries pending_action_id (a
            # string), not the PendingAction itself, so it is looked up here,
            # in the same process, from agent.approval's in-memory store
            # (still available immediately after run_agent_loop returns) and
            # persisted separately below.
            if state.pending_action_id is not None:
                pending_action = approval.get_pending_action(state.pending_action_id)
                if pending_action is not None:
                    pending_action_dump = pending_action.model_dump(
                        include={
                            "ticket_id",
                            "action_id",
                            "proposed_root_cause",
                            "proposed_fix",
                            "citation_doc_id",
                        }
                    )
        except Exception as exc:
            runner_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        usage = tracker.as_dict()
    wall_clock_seconds = round(time.perf_counter() - start, 3)

    result = {
        "schema_version": 2,
        "ticket_id": ticket_id,
        "run": {
            "started_at": started_at,
            "wall_clock_seconds": wall_clock_seconds,
            "runner_error": runner_error,
        },
        "ticket": _denormalized_ticket(ticket),
        "state": state_dump,
        # None when no write was queued this run -- distinct from a
        # present-but-clean pending action; see eval/injection_gate.py's
        # no_secret_leak check, which must not conflate the two.
        "pending_action": pending_action_dump,
        "usage": usage,
    }

    out_path = out_dir / f"{ticket_id}.json"
    tmp_path = out_dir / f".{ticket_id}.json.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise

    if runner_error is not None:
        final_status = "crashed"
    elif state_dump is not None:
        final_status = state_dump.get("status")
    else:
        final_status = "crashed"

    return {
        "ticket_id": ticket_id,
        "crashed": runner_error is not None,
        "status": final_status,
        "wall_clock_seconds": wall_clock_seconds,
    }


class ArchiveError(RuntimeError):
    """Raised when archiving a previous sweep fails partway through.

    A failed archive must never fall through into a new sweep over a
    half-cleared (or half-archived) out_dir -- that is exactly the mixed-
    sweep state this feature exists to prevent. Callers must abort the run
    with a non-zero exit on this error, before any ticket executes.
    """


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _make_fresh_archive_dir(archive_root: Path, timestamp: str) -> Path:
    """Create and return a provably fresh archive directory for this sweep.

    mkdir(parents=True) WITHOUT exist_ok is the freshness check: if the
    second-resolution timestamp collides with a directory that already
    exists -- whether from another archive operation in the same second or
    from anything else that happened to create that path -- FileExistsError
    is raised and we disambiguate with a "-2", "-3", ... suffix until a
    genuinely new directory is created. This is what makes the "nothing is
    ever overwritten" guarantee hold; padding the timestamp with
    microseconds would narrow the collision window but not eliminate it.
    """
    suffix = 1
    while True:
        candidate = archive_root / (timestamp if suffix == 1 else f"{timestamp}-{suffix}")
        try:
            candidate.mkdir(parents=True)
        except FileExistsError:
            suffix += 1
            continue
        return candidate


def _archive_previous_sweep(out_dir: Path, archive_root: Path | None = None) -> None:
    """Move any pre-existing *.json result files out of out_dir before a new
    sweep starts.

    MOVE, deliberately, not copy: if stale files from a previous sweep were
    left behind, a later run over a smaller --subset/--tickets selection
    would overwrite only some of the files already in out_dir, leaving a
    silent mix of two different sweeps that looks like one to a later
    reader. Moving guarantees out_dir always holds exactly one sweep's
    output, with nothing lost -- the old files are archived, not deleted.

    Only *.json files are archived; .gitkeep and any other dotfiles are left
    in place (Path.glob("*.json") does not match dotfile names, since "*"
    does not match a leading dot).

    If a move fails partway through, this stops immediately, reports which
    files were already moved and which remain (with the destination dir
    path, so nothing is orphaned silently), and raises ArchiveError. It does
    NOT attempt to move the remaining files by any other means and does NOT
    let the sweep proceed -- the caller must abort.
    """
    stale = sorted(out_dir.glob("*.json"))
    if not stale:
        return

    if archive_root is None:
        archive_root = ARCHIVE_ROOT

    timestamp = _utc_timestamp()
    dest_dir = _make_fresh_archive_dir(archive_root, timestamp)

    moved = []
    for index, path in enumerate(stale):
        try:
            shutil.move(str(path), str(dest_dir / path.name))
        except OSError as exc:
            remaining = stale[index:]
            print(
                f"Archive failed while moving {path} to {dest_dir}: {exc}\n"
                f"  moved ({len(moved)}): {[p.name for p in moved]}\n"
                f"  remaining in {out_dir} ({len(remaining)}): {[p.name for p in remaining]}\n"
                f"  archive dir: {dest_dir}",
                file=sys.stderr,
            )
            raise ArchiveError(f"failed to archive {path} to {dest_dir}: {exc}") from exc
        moved.append(path)

    print(f"Archived {len(stale)} file(s) from a previous sweep to {dest_dir}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the agent loop over data/tickets.json and write raw "
        "per-ticket result files. Computes no metrics -- see eval/report.py."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Required to actually run. Hits the real LLM API for every ticket.",
    )
    parser.add_argument("--subset", type=int, default=None, help="Run only the first N tickets.")
    parser.add_argument(
        "--tickets",
        type=str,
        default=None,
        help="Comma-separated ticket ids to run, e.g. T001,T009. Mutually exclusive with --subset.",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help=(
            "Run exactly the adversarial-category tickets (T064-T072) instead of the "
            "default 63-ticket main sweep. Mutually exclusive with --subset and --tickets."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for per-ticket result files (default: eval/results/raw).",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip archiving pre-existing *.json files in the out dir before this sweep.",
    )
    args = parser.parse_args(argv)

    if not args.live:
        print(
            "Refusing to run: this hits the real LLM API for every ticket and costs real "
            "tokens. Rerun with --live to proceed. (Mirrors pytest.ini's `-m \"not live\"` "
            "convention -- live is opt-in, never the default.)",
            file=sys.stderr,
        )
        return 2

    if args.subset is not None and args.tickets is not None:
        print("Error: --subset and --tickets are mutually exclusive.", file=sys.stderr)
        return 2

    if args.adversarial and (args.subset is not None or args.tickets is not None):
        print("Error: --adversarial is mutually exclusive with --subset and --tickets.", file=sys.stderr)
        return 2

    try:
        tickets = _load_tickets()
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: could not read {TICKETS_PATH}: {exc}", file=sys.stderr)
        return 2

    ticket_ids = args.tickets.split(",") if args.tickets is not None else None
    try:
        selected = _select_tickets(tickets, args.subset, ticket_ids, adversarial=args.adversarial)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_archive:
        try:
            _archive_previous_sweep(out_dir)
        except ArchiveError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    total = len(selected)
    n_crashed = 0
    overall_start = time.perf_counter()

    for i, ticket in enumerate(selected, start=1):
        summary = run_one(ticket, out_dir)
        status = summary["status"]
        crashed_note = " (crashed)" if summary["crashed"] else ""
        if summary["crashed"]:
            n_crashed += 1
        print(
            f"[{i}/{total}] {summary['ticket_id']} status={status} "
            f"wall_clock={summary['wall_clock_seconds']:.3f}s{crashed_note}",
            file=sys.stderr,
        )

    overall_wall_clock = round(time.perf_counter() - overall_start, 3)
    print(
        f"{total} run, {n_crashed} crashed, total wall clock {overall_wall_clock:.3f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
