"""LLM-as-judge for semantic correctness of agent hypotheses.

Produces ``hypothesis_semantic_verdict`` — a per-ticket judgment of whether
the agent's free-text hypothesis identifies the same underlying root cause
as the gold ``ticket.gold_root_cause`` slug, independent of wording,
verbosity, or added detail.

The judge is deliberately blind to any prior task-success metric: it is
never told about ``task_success_strict_lexical`` or
``task_success_status_only`` outcomes, since it exists to explain the gap
between them.

Run under the project venv (``venv/bin/python``), not ``venv-ragas``. Uses
``agent.llm.call_llm_with_tools`` for provider plumbing — see
``agent/llm.py`` for retry/backoff behaviour and env var configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.llm import MODEL, call_llm_with_tools
TICKETS_PATH = REPO_ROOT / "data" / "tickets.json"
RAW_DIR = REPO_ROOT / "eval" / "results" / "raw"
GAP_SET_PATH = REPO_ROOT / "eval" / "results" / "gap_set.json"
DEFAULT_OUT_PATH = REPO_ROOT / "eval" / "results" / "hypothesis_semantic.json"

RECORD_VERDICT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "record_verdict",
        "description": (
            "Record whether the agent's hypothesis identifies the same "
            "underlying root cause as the gold root cause."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "semantically_correct": {
                    "type": "boolean",
                    "description": (
                        "True if the hypothesis identifies the same "
                        "underlying root cause as the gold slug, allowing "
                        "for differences in wording, verbosity, or added "
                        "detail. False if it identifies a different "
                        "mechanism."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": "1-3 sentences explaining the verdict.",
                },
            },
            "required": ["semantically_correct", "reasoning"],
        },
    },
}


def build_prompt(ticket_text: str, category: str, hypothesis: str, gold_root_cause: str) -> str:
    return (
        "You are judging whether an IT-incident diagnosis agent's hypothesis "
        "identifies the same underlying root cause as a gold root cause "
        "label.\n\n"
        f"Ticket category: {category}\n"
        f"Ticket description: {ticket_text}\n\n"
        f"Agent's hypothesis: {hypothesis}\n\n"
        f"Gold root cause (slug): {gold_root_cause}\n\n"
        "Question: does the agent's hypothesis identify the same underlying "
        "root cause as the gold root cause slug?\n\n"
        "Differences in wording, verbosity, or added detail do NOT make the "
        "hypothesis incorrect. Identifying a different underlying mechanism "
        "DOES make it incorrect.\n\n"
        "Call record_verdict with your boolean verdict first "
        "(semantically_correct), then a 1-3 sentence reasoning."
    )


def _extract_verdict(resp) -> tuple[bool | None, str | None, str | None]:
    """Returns (verdict, reasoning, error). Exactly one of (verdict is not
    None) or (error is not None) should hold on failure; on success verdict
    is a bool and error is None.
    """
    if isinstance(resp, dict) and "error" in resp:
        return None, None, resp["error"]

    try:
        message = resp.choices[0].message
        tool_calls = message.tool_calls
    except (AttributeError, IndexError):
        return None, None, "malformed LLM response: no choices/message"

    if not tool_calls:
        return None, None, "no tool call returned by the model"

    call = tool_calls[0]
    try:
        args = json.loads(call.function.arguments)
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        return None, None, f"failed to parse tool call arguments: {exc}"

    if "semantically_correct" not in args or not isinstance(
        args["semantically_correct"], bool
    ):
        return None, None, "tool call arguments missing/invalid 'semantically_correct'"

    reasoning = args.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return args["semantically_correct"], reasoning, None


def judge_once(ticket_text: str, category: str, hypothesis: str, gold_root_cause: str):
    prompt = build_prompt(ticket_text, category, hypothesis, gold_root_cause)
    messages = [{"role": "user", "content": prompt}]
    resp = call_llm_with_tools(
        messages,
        [RECORD_VERDICT_SCHEMA],
        temperature=0.0,
        tool_choice={"type": "function", "function": {"name": "record_verdict"}},
    )
    return _extract_verdict(resp)


def _majority(verdicts: list[bool | None]) -> tuple[bool | None, float]:
    successful = [v for v in verdicts if v is not None]
    if not successful:
        return None, 0.0
    counts = Counter(successful)
    top_value, top_count = counts.most_common(1)[0]
    # tie => no majority
    if len(counts) > 1 and list(counts.values()).count(top_count) > 1:
        return None, top_count / len(successful)
    return top_value, top_count / len(successful)


def judge_ticket(ticket_text: str, category: str, hypothesis: str, gold_root_cause: str, repeats: int):
    verdicts: list[bool | None] = []
    reasonings: list[str | None] = []
    errors: list[str | None] = []
    for _ in range(repeats):
        v, r, e = judge_once(ticket_text, category, hypothesis, gold_root_cause)
        verdicts.append(v)
        reasonings.append(r)
        errors.append(e)

    verdict, agreement = _majority(verdicts)
    error = None
    if all(v is None for v in verdicts):
        error = "; ".join(e for e in errors if e) or "all judge calls failed"

    return {
        "verdict": verdict,
        "verdicts": verdicts,
        "agreement": agreement,
        "reasonings": reasonings,
        "error": error,
    }


def load_tickets() -> dict[str, dict]:
    with open(TICKETS_PATH) as f:
        tickets = json.load(f)
    return {t["id"]: t for t in tickets}


def load_gap_set_ids(gap_set_path: str | Path = GAP_SET_PATH) -> list[str]:
    with open(gap_set_path) as f:
        data = json.load(f)
    return [t["ticket_id"] for t in data["tickets"]]


def load_gap_set_records(gap_set_path: str | Path = GAP_SET_PATH) -> dict[str, dict]:
    """Ticket rows from the committed gap_set.json, keyed by ticket_id.

    Used as a fallback source when eval/results/raw/{ticket_id}.json is not
    present (that directory is gitignored and empty on a fresh checkout).
    """
    with open(gap_set_path) as f:
        data = json.load(f)
    return {t["ticket_id"]: t for t in data["tickets"]}


def resolve_ticket_input(
    ticket_id: str, gap_set_records: dict[str, dict]
) -> tuple[str | None, str | None, str | None]:
    """Returns (hypothesis, category, source).

    Prefers eval/results/raw/{ticket_id}.json when present ("raw"); falls
    back to the matching row in gap_set.json ("gap_set"). source is None if
    neither source yields a hypothesis.
    """
    hypothesis = load_raw_hypothesis(ticket_id)
    category = load_raw_category(ticket_id)
    if hypothesis is not None:
        return hypothesis, category, "raw"

    gap_row = gap_set_records.get(ticket_id)
    if gap_row is not None and gap_row.get("hypothesis") is not None:
        return gap_row.get("hypothesis"), category or gap_row.get("category"), "gap_set"

    return None, category, None


def load_raw_hypothesis(ticket_id: str) -> str | None:
    path = RAW_DIR / f"{ticket_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return raw.get("state", {}).get("hypothesis")


def load_raw_category(ticket_id: str) -> str | None:
    path = RAW_DIR / f"{ticket_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return raw.get("ticket", {}).get("category")


def resolve_ticket_ids(args) -> list[str]:
    gap_set_path = getattr(args, "gap_set_file", None) or GAP_SET_PATH
    if args.only:
        return [t.strip() for t in args.only.split(",") if t.strip()]
    if args.gap_set:
        ids = load_gap_set_ids(gap_set_path)
    else:
        all_tickets = load_tickets()
        ids = sorted(all_tickets.keys())
    if args.subset is not None:
        ids = sorted(ids)[: args.subset]
    return ids


def run(args) -> dict:
    tickets_by_id = load_tickets()
    ticket_ids = resolve_ticket_ids(args)
    gap_set_path = getattr(args, "gap_set_file", None) or GAP_SET_PATH
    gap_set_records = load_gap_set_records(gap_set_path)

    per_ticket = []
    for ticket_id in ticket_ids:
        ticket = tickets_by_id.get(ticket_id)
        ticket_text = ticket.get("ticket_text") if ticket else None
        gold_root_cause = ticket.get("gold_root_cause") if ticket else None
        hypothesis, category, source = resolve_ticket_input(ticket_id, gap_set_records)
        category = category or (ticket.get("category") if ticket else None)

        if ticket is None or hypothesis is None or gold_root_cause is None:
            per_ticket.append(
                {
                    "ticket_id": ticket_id,
                    "category": category,
                    "gold_root_cause": gold_root_cause,
                    "hypothesis": hypothesis,
                    "source": source,
                    "verdict": None,
                    "verdicts": [],
                    "agreement": 0.0,
                    "reasonings": [],
                    "error": "missing ticket text, hypothesis, or gold root cause",
                }
            )
            continue

        result = judge_ticket(ticket_text, category, hypothesis, gold_root_cause, args.repeats)
        per_ticket.append(
            {
                "ticket_id": ticket_id,
                "category": category,
                "gold_root_cause": gold_root_cause,
                "hypothesis": hypothesis,
                "source": source,
                **result,
            }
        )

    n_judged = sum(1 for t in per_ticket if t["verdict"] is not None)
    n_correct = sum(1 for t in per_ticket if t["verdict"] is True)
    n_incorrect = sum(1 for t in per_ticket if t["verdict"] is False)
    n_failed = sum(1 for t in per_ticket if t["verdict"] is None)
    semantic_correct_rate = (n_correct / n_judged) if n_judged > 0 else None
    sources = Counter(t["source"] for t in per_ticket if t["source"] is not None)

    return {
        "schema_version": 1,
        "provider": {
            "model": MODEL,
            "base_url": os.environ.get(
                "LLM_BASE_URL", "https://api.groq.com/openai/v1"
            ).strip(),
        },
        "config": {
            "n_tickets": len(ticket_ids),
            "repeats": args.repeats,
            "gap_set": bool(args.gap_set),
            "gap_set_file": str(gap_set_path),
            "sources": {"raw": sources.get("raw", 0), "gap_set": sources.get("gap_set", 0)},
        },
        "summary": {
            "n_judged": n_judged,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_failed": n_failed,
            "semantic_correct_rate": semantic_correct_rate,
        },
        "per_ticket": per_ticket,
    }


def print_summary(report: dict) -> None:
    summary = report["summary"]
    config = report["config"]
    n_total = config["n_tickets"]
    rate = summary["semantic_correct_rate"]
    rate_str = f"{rate:.3f}" if rate is not None else "N/A"
    if summary["n_failed"] > 0:
        print(
            f"hypothesis_semantic_verdict: rate={rate_str} "
            f"over {summary['n_judged']}/{n_total} judged "
            f"(n_correct={summary['n_correct']}, n_incorrect={summary['n_incorrect']}, "
            f"n_failed={summary['n_failed']})"
        )
    else:
        print(
            f"hypothesis_semantic_verdict: rate={rate_str} "
            f"over {summary['n_judged']}/{n_total} judged "
            f"(n_correct={summary['n_correct']}, n_incorrect={summary['n_incorrect']})"
        )
    sources = config.get("sources", {})
    print(
        f"input sources: raw={sources.get('raw', 0)} gap_set={sources.get('gap_set', 0)} "
        f"(gap_set_file={config.get('gap_set_file')})"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-set", action="store_true", default=True)
    parser.add_argument("--gap-set-file", type=str, default=str(GAP_SET_PATH))
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.only:
        args.gap_set = False

    ticket_ids = resolve_ticket_ids(args)

    if args.dry_run:
        n_calls = len(ticket_ids) * args.repeats
        gap_set_records = load_gap_set_records(args.gap_set_file)
        raw_available = sum(1 for tid in ticket_ids if (RAW_DIR / f"{tid}.json").exists())
        gap_set_available = sum(
            1
            for tid in ticket_ids
            if not (RAW_DIR / f"{tid}.json").exists() and tid in gap_set_records
        )
        print(f"ticket_count={len(ticket_ids)} repeats={args.repeats} projected_calls={n_calls}")
        print(
            f"projected input sources: raw={raw_available} gap_set={gap_set_available} "
            f"(gap_set_file={args.gap_set_file})"
        )
        return 0

    try:
        report = run(args)
    except Exception as exc:
        print(f"hard failure: {exc}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
