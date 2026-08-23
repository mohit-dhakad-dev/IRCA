"""Constraint-parsing completeness dump for agent/approval.py.

This is a standalone diagnostic SCRIPT, not a pytest test. It makes NO API
calls, asserts NOTHING, and modifies NOTHING -- it is a dump for human
reading. Its purpose is to show 100% of what the constraint parser produces
against 100% of what actually exists in the runbook corpus, since the parser
has previously been "fixed" from hand-picked cases and both times missed
defects a full-corpus view would have caught.

It loads all six runbooks via the EXISTING loader
(`rag.ingest.RUNBOOKS_DIR` / `rag.ingest.parse_runbook`), selects each
runbook's "Constraints" section chunk, splits it into its bullets (every
"- ..." line), and for each bullet calls the real internal helper
`agent.approval._extract_bounds_from_bullet_with_param` (which in turn calls
`agent.approval._extract_bounds_from_clause` per ';'-separated clause) to
extract every (op, value, unit, subject_tokens, bullet_text, parameter)
bound. This is deliberately the SAME function `verify_against_constraints`
uses to check proposed fixes, so this dump can never drift from what the
system actually verifies. Every bullet is printed, even ones that yield zero
bounds (marked "(NO BOUNDS PARSED)") -- nothing is hand-picked or filtered.

Usage (from repo root, using the repo's venv):

    venv/bin/python -m eval.verify_constraint_parsing
    venv/bin/python -m eval.verify_constraint_parsing --json
"""

from __future__ import annotations

import argparse
import json

from agent.approval import _extract_bounds_from_bullet_with_param
from rag.ingest import RUNBOOKS_DIR, parse_runbook


def _bullets_for_doc(path) -> list[str]:
    chunks = parse_runbook(path)
    constraints_chunk = next((c for c in chunks if c.section == "Constraints"), None)
    if constraints_chunk is None:
        return []
    return [
        line.strip()
        for line in constraints_chunk.body.splitlines()
        if line.strip().startswith("-")
    ]


def _truncate(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def collect() -> list[dict]:
    """One dict per runbook: doc_id, path, and a list of bullet records.
    Each bullet record has the source text and its parsed bounds (possibly
    empty)."""
    results: list[dict] = []
    for path in sorted(RUNBOOKS_DIR.glob("*.md")):
        doc_id = path.stem
        bullets = _bullets_for_doc(path)
        bullet_records = []
        for bullet in bullets:
            bounds = _extract_bounds_from_bullet_with_param(bullet)
            bound_records = [
                {
                    "direction": op,
                    "value": value,
                    "unit": unit,
                    "subject_tokens": sorted(subject),
                    "parameter": param,
                }
                for op, value, unit, subject, _bullet_text, param in bounds
            ]
            bullet_records.append(
                {
                    "source_text": bullet,
                    "bounds": bound_records,
                }
            )
        results.append(
            {
                "doc_id": doc_id,
                "path": str(path),
                "bullets": bullet_records,
            }
        )
    return results


def print_report(results: list[dict]) -> None:
    total_bullets = 0
    total_bounds = 0
    bullets_with_no_bounds = 0
    bounds_with_unknown_parameter = 0

    for doc in results:
        print("\n" + "=" * 100)
        print(f"RUNBOOK: {doc['doc_id']}  ({doc['path']})")
        print("=" * 100)

        if not doc["bullets"]:
            print("  (no Constraints section / no bullets found)")
            continue

        for bullet in doc["bullets"]:
            total_bullets += 1
            print("-" * 100)
            print(f"  source: {_truncate(bullet['source_text'])}")
            if not bullet["bounds"]:
                bullets_with_no_bounds += 1
                print("    (NO BOUNDS PARSED)")
                continue
            for bound in bullet["bounds"]:
                total_bounds += 1
                parameter = bound["parameter"]
                if not parameter:
                    bounds_with_unknown_parameter += 1
                    parameter_disp = "(EMPTY/UNKNOWN)"
                else:
                    parameter_disp = parameter
                print(
                    f"    doc_id={doc['doc_id']:<14} direction={bound['direction']:<4} "
                    f"value={bound['value']!s:<8} unit={(bound['unit'] or '(no unit)'):<10} "
                    f"parameter=[{parameter_disp}]"
                )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total bullets:                          {total_bullets}")
    print(f"Total bounds parsed:                    {total_bounds}")
    print(f"Bullets producing NO bounds:             {bullets_with_no_bounds}")
    print(f"Bounds with empty/unknown parameter:     {bounds_with_unknown_parameter}")


def main(as_json: bool) -> None:
    results = collect()
    if as_json:
        print(json.dumps(results, indent=2))
        return
    print_report(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnostic: dump every constraint bullet in the runbook corpus "
        "alongside every bound agent.approval's parser extracts from it, for full-corpus "
        "human review before touching the parser again."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the same data as machine-readable JSON instead of the printed report.",
    )
    args = parser.parse_args()
    main(as_json=args.json)
