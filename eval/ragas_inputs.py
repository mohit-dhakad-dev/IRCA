"""Reconstruct the four RAGAS input fields (question, answer, contexts,
ground_truth) for each RESOLVED ticket, purely from already-recorded
artifacts on disk (eval/results/raw/*.json, data/tickets.json,
data/runbooks/*.md via rag/ingest.py).

This module NEVER re-runs the agent and makes NO network/LLM calls -- it is
a pure reconstruction over what a prior benchmark sweep already wrote out.

The critical contract here is how `answer` is identified. A raw result's
trajectory can contain MULTIPLE `update_ticket` tool_call entries at the
SAME iteration: the write gate rejects a proposed fix that violates a
runbook constraint (status "verification_failed", action_id None in the
observation), the agent revises, and the accepted retry is what actually
gets queued for approval and tracked as `state.pending_action_id`. "The
last update_ticket call" and "disambiguate by iteration index" both fail on
tickets like T011 and T024, which have exactly this rejected-then-accepted
pair at one iteration. The only correct identifier is the write gate's own
action_id: the trajectory entry whose tool_call.name == "update_ticket" AND
whose observation's data.action_id equals state.pending_action_id is, by
construction, the one specific action the state is tracking as the
terminal, accepted proposal. If that match count is ever not exactly 1, the
raw record is treated as broken and this module raises rather than guessing.

Standalone script:

    venv/bin/python -m eval.ragas_inputs
    venv/bin/python -m eval.ragas_inputs --only T009,T038
    venv/bin/python -m eval.ragas_inputs --json eval/results/ragas_inputs.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.ingest import CHUNK_SECTIONS, Chunk, load_chunks

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "eval" / "results" / "raw"
TICKETS_PATH = REPO_ROOT / "data" / "tickets.json"


@dataclass(frozen=True)
class RagasInput:
    ticket_id: str
    category: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    citations: list[str]
    gold_runbook_id: str
    action_id: str


def load_chunk_index(runbooks_dir: Path | None = None) -> dict[str, dict[str, Chunk]]:
    """Build {doc_id: {section: Chunk}} once from the runbook corpus, so
    build_all() doesn't re-parse every runbook file per ticket."""
    kwargs = {} if runbooks_dir is None else {"runbooks_dir": runbooks_dir}
    chunks = load_chunks(**kwargs)
    index: dict[str, dict[str, Chunk]] = {}
    for chunk in chunks:
        index.setdefault(chunk.doc_id, {})[chunk.section] = chunk
    return index


def load_tickets(tickets_path: Path = TICKETS_PATH) -> dict[str, dict]:
    tickets = json.loads(tickets_path.read_text(encoding="utf-8"))
    return {t["id"]: t for t in tickets}


def _find_terminal_answer(ticket_id: str, state: dict) -> tuple[str, str]:
    """Return (answer, action_id) for the single trajectory entry whose
    update_ticket observation's action_id matches state.pending_action_id.
    See module docstring for why this is the only valid disambiguator."""
    pending_action_id = state.get("pending_action_id")
    matches = []
    for entry in state.get("trajectory", []):
        tool_call = entry.get("tool_call") or {}
        if tool_call.get("name") != "update_ticket":
            continue
        observation = entry.get("observation") or {}
        data = observation.get("data") or {}
        if data.get("action_id") == pending_action_id:
            arguments = tool_call.get("arguments") or {}
            proposed_fix = arguments.get("proposed_fix")
            action_id = data.get("action_id")
            if proposed_fix is None or not proposed_fix.strip():
                raise ValueError(
                    f"{ticket_id}: terminal update_ticket call (action_id={action_id!r}) "
                    f"has a missing or empty proposed_fix"
                )
            matches.append((proposed_fix, action_id))

    if len(matches) != 1:
        raise ValueError(
            f"{ticket_id}: expected exactly 1 terminal update_ticket call matching "
            f"pending_action_id={pending_action_id!r}, found {len(matches)}"
        )
    return matches[0]


def _contexts_for_citations(ticket_id: str, citations: list[str], chunk_index: dict[str, dict[str, Chunk]]) -> list[str]:
    """For each cited doc_id, in citation order, emit the .text of all 5 of
    that doc's chunks in CHUNK_SECTIONS order. Citations record only a
    doc_id, not which section was retrieved, so the whole document's chunk
    set is the honest reconstruction of what evidence that citation could
    have drawn from."""
    contexts: list[str] = []
    for doc_id in citations:
        sections = chunk_index.get(doc_id)
        if not sections:
            raise ValueError(f"{ticket_id}: cited doc_id {doc_id!r} has no chunks in the runbook corpus")
        for section in CHUNK_SECTIONS:
            contexts.append(sections[section].text)
    return contexts


def build_ragas_input(
    ticket_id: str,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    tickets: dict[str, dict],
    chunk_index: dict[str, dict[str, Chunk]],
) -> RagasInput:
    raw = json.loads((Path(raw_dir) / f"{ticket_id}.json").read_text(encoding="utf-8"))
    state = raw["state"]
    ticket_meta = raw["ticket"]

    if ticket_id not in tickets:
        raise ValueError(f"{ticket_id}: not found in data/tickets.json")
    question = tickets[ticket_id]["ticket_text"]

    answer, action_id = _find_terminal_answer(ticket_id, state)

    # A RESOLVED ticket that cites no runbook is a project invariant
    # violation (verification requires citing evidence), not a legitimate
    # zero-citation answer -- treat it as corrupt raw data and raise loudly
    # rather than silently producing contexts=[] and tanking ragas scores
    # with no signal.
    citations = state.get("citations")
    if not citations:
        raise ValueError(f"{ticket_id}: resolved ticket has no citations in state")
    contexts = _contexts_for_citations(ticket_id, citations, chunk_index)

    gold_runbook_id = ticket_meta["gold_runbook_id"]
    gold_doc_id = gold_runbook_id.removesuffix(".md")
    gold_sections = chunk_index.get(gold_doc_id)
    if not gold_sections:
        raise ValueError(f"{ticket_id}: gold runbook {gold_doc_id!r} has no chunks in the runbook corpus")
    ground_truth = gold_sections["Fix"].body

    return RagasInput(
        ticket_id=ticket_id,
        category=ticket_meta["category"],
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        citations=citations,
        gold_runbook_id=gold_runbook_id,
        action_id=action_id,
    )


def build_all(
    raw_dir: Path = DEFAULT_RAW_DIR,
    only: list[str] | None = None,
) -> list[RagasInput]:
    """Build RagasInput rows for every RESOLVED ticket under raw_dir.
    Escalated tickets are skipped -- they never produced an accepted
    update_ticket call, so there is no `answer` to reconstruct."""
    raw_dir = Path(raw_dir)
    tickets = load_tickets()
    chunk_index = load_chunk_index()

    ticket_ids = sorted(p.stem for p in raw_dir.glob("*.json"))
    if only is not None:
        wanted = set(only)
        ticket_ids = [t for t in ticket_ids if t in wanted]

    rows: list[RagasInput] = []
    for ticket_id in ticket_ids:
        raw = json.loads((raw_dir / f"{ticket_id}.json").read_text(encoding="utf-8"))
        if raw["state"].get("status") != "resolved":
            continue
        rows.append(
            build_ragas_input(ticket_id, raw_dir=raw_dir, tickets=tickets, chunk_index=chunk_index)
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", type=str, default=None, help="Comma-separated ticket_ids to restrict to")
    parser.add_argument("--json", type=str, default=None, help="Path to write the full rows as JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    only = args.only.split(",") if args.only else None

    all_ids = sorted(p.stem for p in DEFAULT_RAW_DIR.glob("*.json"))
    if only is not None:
        wanted = set(only)
        all_ids = [t for t in all_ids if t in wanted]
    escalated_count = sum(
        1 for t in all_ids
        if json.loads((DEFAULT_RAW_DIR / f"{t}.json").read_text(encoding="utf-8"))["state"].get("status") != "resolved"
    )

    rows = build_all(only=only)

    print(f"Built {len(rows)} RAGAS input rows, skipped {escalated_count} escalated tickets.")
    for row in rows:
        print(
            f"  {row.ticket_id}: answer={len(row.answer)} chars, "
            f"contexts={len(row.contexts)}, ground_truth={len(row.ground_truth)} chars"
        )

    if args.json:
        out_path = Path(args.json)
        out_path.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
        print(f"Wrote {len(rows)} rows to {out_path}")
