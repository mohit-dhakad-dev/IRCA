"""Human-vs-judge agreement for the ``semantically_correct`` label.

Validates the LLM judge in ``eval/llm_judge.py`` against the project owner's
own hand labels on the 46-ticket gap set. The owner hand-labels
``semantically_correct`` BLIND to the judge's output; this module then
computes agreement statistics (Cohen's kappa, PABAK, raw agreement) between
the two label sets.

Two subcommands:

- ``template``: emit a blind labeling template (no judge output, no gold
  reasoning) for a sample of tickets from the 46-ticket gap set.
- ``compare``: compute agreement statistics between a filled-in human
  template and the judge's ``hypothesis_semantic.json`` output.

Sampling note: the originally-planned second selection axis for the
template sample was low ``context_precision``, but that metric is
structurally degenerate on this data (constant at 1.0000 across 14
readings), so it carries no selection signal. Low-faithfulness
oversampling (within a proportional category spread) was used initially,
but measurement subsequently showed faithfulness does not predict the
judge's verdict (and is mildly inverted): the ``faithfulness`` strategy
is preserved and reachable via ``--strategy faithfulness`` for
reproducibility, but the default is now ``disagreement``, which
oversamples the judge's own uncertain/incorrect cases (verdict False,
and non-unanimous repeats) instead. See the ``--strategy`` flag on the
``template`` subcommand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.llm_judge import (
    OBSERVATIONS_CHAR_CAP,
    RUBRIC_TEXT,
    RUBRIC_VERSION,
    load_observations,
    render_observations,
)

TICKETS_PATH = REPO_ROOT / "data" / "tickets.json"
RAW_DIR = REPO_ROOT / "eval" / "results" / "raw"
GAP_SET_PATH = REPO_ROOT / "eval" / "results" / "gap_set.json"
DEFAULT_RAGAS_PATH = REPO_ROOT / "eval" / "results" / "ragas_scores_gap46.json"
DEFAULT_TEMPLATE_OUT = REPO_ROOT / "eval" / "results" / "human_label_template.json"
DEFAULT_JUDGE_PATH = REPO_ROOT / "eval" / "results" / "hypothesis_semantic.json"
DEFAULT_REPORT_OUT = REPO_ROOT / "eval" / "results" / "judge_agreement_report.json"

CATEGORY_COUNTS = {
    "multi_step": 17,
    "easy": 12,
    "tool_heavy": 10,
    "rag_heavy": 7,
}


# ---------------------------------------------------------------------------
# template
# ---------------------------------------------------------------------------


def _largest_remainder_allocation(counts: dict[str, int], n: int) -> dict[str, int]:
    """Allocate n slots proportionally across categories using
    largest-remainder rounding, so slots sum exactly to n."""
    total = sum(counts.values())
    if total == 0 or n <= 0:
        return {cat: 0 for cat in counts}

    exact = {cat: (counts[cat] / total) * n for cat in counts}
    floors = {cat: int(exact[cat]) for cat in counts}
    allocated = sum(floors.values())
    remainder = n - allocated

    # sort by fractional remainder desc, tie-break by category name asc
    # for determinism
    remainders = sorted(
        counts.keys(),
        key=lambda cat: (-(exact[cat] - floors[cat]), cat),
    )
    for cat in remainders[:remainder]:
        floors[cat] += 1

    return floors


def _load_gap_set(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["tickets"]


def _load_ragas_faithfulness(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"RAGAS results file not found at {path}. Run the RAGAS eval "
            "first (eval/ragas_eval.py) to produce faithfulness scores "
            "before building the human labeling template — sampling "
            "without faithfulness scores would silently change the "
            "documented methodology."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    scores = {}
    for row in data.get("per_ticket", []):
        tid = row.get("ticket_id")
        faith = row.get("faithfulness")
        if tid is not None and faith is not None:
            scores[tid] = faith
    return scores


def _load_tickets_index(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        tickets = json.load(f)
    return {t["id"]: t for t in tickets}


def _load_raw_hypothesis(ticket_id: str) -> str | None:
    raw_path = RAW_DIR / f"{ticket_id}.json"
    if not raw_path.exists():
        return None
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("state", {}).get("hypothesis")


def select_sample(
    gap_tickets: list[dict],
    faithfulness: dict[str, float],
    n: int,
) -> list[dict]:
    """Select n tickets from the gap set: proportional category spread
    (largest-remainder rounding), lowest faithfulness first within each
    category, deterministic ties broken by ticket_id ascending."""
    by_category: dict[str, list[dict]] = {}
    for t in gap_tickets:
        by_category.setdefault(t["category"], []).append(t)

    counts = {cat: len(tickets) for cat, tickets in by_category.items()}
    allocation = _largest_remainder_allocation(counts, n)

    selected: list[dict] = []
    for cat, slots in allocation.items():
        if slots <= 0:
            continue
        candidates = by_category.get(cat, [])
        # sort by faithfulness ascending (missing scores sort last), then
        # ticket_id ascending for determinism
        ranked = sorted(
            candidates,
            key=lambda t: (
                faithfulness.get(t["ticket_id"], float("inf")),
                t["ticket_id"],
            ),
        )
        selected.extend(ranked[:slots])

    selected.sort(key=lambda t: t["ticket_id"])
    return selected


def _load_judge_per_ticket(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Judge results file not found at {path}. The disagreement "
            "sampling strategy requires the judge's per-ticket verdicts "
            "(and agreement values) to know which tickets are "
            "disagreement/uncertain cases — pass --judge pointing at "
            "hypothesis_semantic.json, or run the semantic judge first."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("per_ticket", [])


def select_sample_disagreement(
    gap_tickets: list[dict],
    judge_rows: list[dict],
    n: int,
) -> tuple[list[dict], dict]:
    """Select tickets weighted toward judge disagreement/uncertainty.

    Mandatory set: all judge verdict==False tickets, plus all tickets with
    agreement < 1.0 (deduplicated). Remaining slots up to n are filled with
    verdict==True tickets, allocated proportionally by category using
    largest-remainder rounding, ticket_id ascending for determinism.

    Returns (selected_tickets, provenance_dict). If the mandatory set
    exceeds n, all mandatory tickets are still included (never truncated)
    and provenance notes the overage; the caller is responsible for
    printing a warning.
    """
    by_id = {t["ticket_id"]: t for t in gap_tickets}

    false_ids = [
        r["ticket_id"] for r in judge_rows if r.get("verdict") is False
    ]
    nonunanimous_ids = [
        r["ticket_id"]
        for r in judge_rows
        if r.get("agreement") is not None and r.get("agreement") < 1.0
    ]

    mandatory_ids = list(dict.fromkeys(false_ids + nonunanimous_ids))
    mandatory_ids.sort()

    true_ids = sorted(
        r["ticket_id"] for r in judge_rows
        if r.get("verdict") is True and r["ticket_id"] not in set(mandatory_ids)
    )

    n_mandatory = len(mandatory_ids)
    n_exceeded = max(0, n_mandatory - n)
    remaining = max(0, n - n_mandatory)

    true_tickets = [by_id[tid] for tid in true_ids if tid in by_id]
    by_category: dict[str, list[dict]] = {}
    for t in true_tickets:
        by_category.setdefault(t["category"], []).append(t)
    counts = {cat: len(tickets) for cat, tickets in by_category.items()}
    allocation = _largest_remainder_allocation(counts, min(remaining, sum(counts.values())))

    filled_true: list[dict] = []
    for cat, slots in allocation.items():
        if slots <= 0:
            continue
        candidates = sorted(by_category.get(cat, []), key=lambda t: t["ticket_id"])
        filled_true.extend(candidates[:slots])

    filled_true.sort(key=lambda t: t["ticket_id"])

    selected = [by_id[tid] for tid in mandatory_ids if tid in by_id] + filled_true
    selected.sort(key=lambda t: t["ticket_id"])

    category_counts: dict[str, int] = {}
    for t in selected:
        category_counts[t["category"]] = category_counts.get(t["category"], 0) + 1

    provenance = {
        "strategy": "disagreement",
        "n_requested": n,
        "n_selected": len(selected),
        "n_mandatory_disagreement": n_mandatory,
        "n_filled_true": len(filled_true),
        "category_counts": category_counts,
        "n_exceeded": n_exceeded,
    }

    return selected, provenance


def build_template(
    gap_set_path: Path,
    ragas_path: Path,
    tickets_path: Path,
    n: int,
    strategy: str = "faithfulness",
    judge_path: Path | None = None,
) -> list[dict]:
    """Build the blind labeling template. Defaults to the legacy
    ``faithfulness`` strategy for backward compatibility with existing
    callers/tests that omit ``strategy``; the CLI's own default is
    ``disagreement`` (see ``main()``)."""
    return build_template_with_provenance(
        gap_set_path, ragas_path, tickets_path, n, strategy, judge_path
    )[0]


def build_template_with_provenance(
    gap_set_path: Path,
    ragas_path: Path,
    tickets_path: Path,
    n: int,
    strategy: str = "faithfulness",
    judge_path: Path | None = None,
) -> tuple[list[dict], dict | None]:
    gap_tickets = _load_gap_set(gap_set_path)
    tickets_index = _load_tickets_index(tickets_path)

    provenance = None

    if strategy == "faithfulness":
        faithfulness = _load_ragas_faithfulness(ragas_path)
        sample = select_sample(gap_tickets, faithfulness, n)
    elif strategy == "disagreement":
        if judge_path is None:
            judge_path = DEFAULT_JUDGE_PATH
        judge_rows = _load_judge_per_ticket(Path(judge_path))
        sample, provenance = select_sample_disagreement(gap_tickets, judge_rows, n)
    elif strategy == "verdict-stratified":
        raise NotImplementedError(
            "verdict-stratified strategy is declared but not yet implemented"
        )
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    template = []
    for t in sample:
        tid = t["ticket_id"]
        ticket = tickets_index.get(tid, {})
        hypothesis = t.get("hypothesis")
        if hypothesis is None:
            hypothesis = _load_raw_hypothesis(tid)

        observations = load_observations(tid, RAW_DIR)
        if observations is None:
            observations_text = (
                f"[no raw trajectory record found for ticket {tid} at "
                f"{RAW_DIR / f'{tid}.json'} — evidence grounding (clause b) "
                "cannot be assessed for this ticket]"
            )
        else:
            observations_text, _truncated = render_observations(observations, OBSERVATIONS_CHAR_CAP)

        template.append(
            {
                "ticket_id": tid,
                "category": t.get("category"),
                "gold_root_cause": t.get("gold_root_cause"),
                "hypothesis": hypothesis,
                "ticket_text": ticket.get("ticket_text"),
                "observations": observations_text,
                "semantically_correct": None,
                "notes": "",
            }
        )
    return template, provenance


def run_template(args: argparse.Namespace) -> None:
    template, provenance = build_template_with_provenance(
        gap_set_path=Path(args.gap_set),
        ragas_path=Path(args.ragas),
        tickets_path=Path(args.tickets),
        n=args.n,
        strategy=args.strategy,
        judge_path=Path(args.judge) if args.judge else None,
    )

    out = {
        "n": len(template),
        "rubric": {
            "version": RUBRIC_VERSION,
            "text": RUBRIC_TEXT,
        },
        "tickets": template,
    }
    out["INSTRUCTIONS"] = (
        "Judge whether the hypothesis satisfies BOTH clauses of the rubric "
        "above: clause (a) semantic match — the hypothesis identifies a "
        "root-cause condition consistent with the gold root cause's "
        "causal chain (a more specific instance of the gold mechanism "
        "still counts; a different, non-overlapping mechanism does not) "
        "— AND clause (b) evidence grounding — the hypothesis is "
        "supported by evidence the agent actually observed in its tool "
        "outputs during its run, not merely asserted without observed "
        "support. Use the 'observations' field on each ticket (the "
        "agent's own recorded tool outputs for that run) to judge clause "
        "(b) — do not assume the hypothesis is grounded just because it "
        "sounds plausible. Mark semantically_correct True only if BOTH "
        "clauses hold; False if either clause fails."
    )

    if provenance is not None:
        n_exceeded = provenance.pop("n_exceeded", 0)
        if n_exceeded > 0:
            print(
                f"WARNING: mandatory disagreement set ({provenance['n_mandatory_disagreement']}) "
                f"exceeds --n ({provenance['n_requested']}) by {n_exceeded}. "
                "All disagreement tickets are included anyway; nothing was dropped."
            )
        out["sampling"] = provenance
        out["WARNING"] = (
            "This sample is deliberately NOT representative of the 46-ticket "
            "gap set. It oversamples judge disagreement/uncertainty (verdict "
            "False and non-unanimous repeats), so the resulting human/judge "
            "agreement statistics describe judge calibration on hard cases "
            "and must NOT be used to correct or re-estimate the overall "
            "semantic-correct rate across the gap set."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(template)} tickets to {out_path}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _load_human_labels(path: Path) -> dict[str, bool | None]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    labels = {}
    for row in data.get("tickets", []):
        labels[row["ticket_id"]] = row.get("semantically_correct")
    return labels


def _load_human_rubric_version(path: Path) -> int | None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rubric", {}).get("version")


def _load_judge_labels(path: Path) -> dict[str, bool | None]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    labels = {}
    for row in data.get("per_ticket", []):
        labels[row["ticket_id"]] = row.get("verdict")
    return labels


def _load_judge_rubric_version(path: Path) -> int | None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("config", {}).get("rubric_version")


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> tuple[float | None, float, str]:
    """Compute Cohen's kappa and observed agreement p_o for a list of
    (rater_a, rater_b) boolean label pairs.

    Returns (kappa, p_o, note). kappa is None when expected agreement
    p_e == 1 (both raters gave the same single label to everything) —
    kappa is undefined in that case, not 0.0.
    """
    n = len(pairs)
    if n == 0:
        return None, 0.0, "no pairs to compare"

    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n

    a_true = sum(1 for a, _ in pairs if a is True) / n
    a_false = 1 - a_true
    b_true = sum(1 for _, b in pairs if b is True) / n
    b_false = 1 - b_true

    p_e = a_true * b_true + a_false * b_false

    prevalence = a_true  # fraction of human (rater a) labels that are True

    pabak = 2 * p_o - 1

    if p_e >= 1.0:
        note = (
            "kappa is undefined: expected agreement p_e == 1 (both raters "
            "gave the same single label to everything, prevalence "
            f"{prevalence:.3f}). PABAK ({pabak:.3f}) is the interpretable "
            "figure here since it does not depend on marginal variance."
        )
        return None, p_o, note

    kappa = (p_o - p_e) / (1 - p_e)

    if p_o > 0.7 and (kappa is None or kappa < 0.3):
        note = (
            "raw agreement is high but kappa is low — this is the "
            "prevalence problem: one class dominates the labels, so "
            f"expected chance agreement (p_e={p_e:.3f}) is already high "
            "and chance-corrected agreement (kappa) becomes unstable. "
            f"PABAK ({pabak:.3f}), which adjusts for prevalence and bias, "
            "is the more interpretable figure at this prevalence "
            f"({prevalence:.3f})."
        )
    else:
        note = (
            f"kappa={kappa:.3f}, PABAK={pabak:.3f}, prevalence={prevalence:.3f}. "
            "PABAK equals 2*p_o - 1 and does not depend on the raters' "
            "marginal label distributions; kappa is chance-corrected "
            "against those marginals and can diverge from PABAK when "
            "prevalence is far from 0.5."
        )

    return kappa, p_o, note


def run_compare(args: argparse.Namespace) -> None:
    human_path = Path(args.human)
    judge_path = Path(args.judge)

    human_labels = _load_human_labels(human_path)
    judge_labels = _load_judge_labels(judge_path)

    human_rubric_version = _load_human_rubric_version(human_path)
    judge_rubric_version = _load_judge_rubric_version(judge_path)

    rubric_warning = None
    if human_rubric_version is None or judge_rubric_version is None:
        rubric_warning = (
            "one or both files lack a rubric_version (human="
            f"{human_rubric_version!r}, judge={judge_rubric_version!r}). "
            "This comparison spans rubric versions and is NOT interpretable."
        )
        print(f"WARNING: {rubric_warning}")
    elif human_rubric_version != judge_rubric_version:
        raise ValueError(
            "rubric version mismatch: human file has rubric_version="
            f"{human_rubric_version!r}, judge file has rubric_version="
            f"{judge_rubric_version!r}. Comparing labels made under "
            "different rubrics is invalid — this is exactly what produced "
            "the chance-level kappa in the v1 run."
        )

    all_ids = sorted(set(human_labels) | set(judge_labels))

    n_skipped = {
        "human_unlabeled": 0,
        "judge_failed": 0,
        "not_in_both": 0,
    }

    pairs: list[tuple[bool, bool]] = []
    per_ticket = []
    contingency = {
        "both_true": 0,
        "both_false": 0,
        "human_true_judge_false": 0,
        "human_false_judge_true": 0,
    }

    for tid in all_ids:
        in_human = tid in human_labels
        in_judge = tid in judge_labels
        human_val = human_labels.get(tid)
        judge_val = judge_labels.get(tid)

        if not (in_human and in_judge):
            n_skipped["not_in_both"] += 1
            per_ticket.append(
                {
                    "ticket_id": tid,
                    "human": human_val if in_human else None,
                    "judge": judge_val if in_judge else None,
                    "compared": False,
                    "skip_reason": "not_in_both",
                }
            )
            continue

        if human_val is None:
            n_skipped["human_unlabeled"] += 1
            per_ticket.append(
                {
                    "ticket_id": tid,
                    "human": human_val,
                    "judge": judge_val,
                    "compared": False,
                    "skip_reason": "human_unlabeled",
                }
            )
            continue

        if judge_val is None:
            n_skipped["judge_failed"] += 1
            per_ticket.append(
                {
                    "ticket_id": tid,
                    "human": human_val,
                    "judge": judge_val,
                    "compared": False,
                    "skip_reason": "judge_failed",
                }
            )
            continue

        pairs.append((human_val, judge_val))
        agreed = human_val == judge_val
        per_ticket.append(
            {
                "ticket_id": tid,
                "human": human_val,
                "judge": judge_val,
                "compared": True,
                "agree": agreed,
            }
        )

        if human_val and judge_val:
            contingency["both_true"] += 1
        elif not human_val and not judge_val:
            contingency["both_false"] += 1
        elif human_val and not judge_val:
            contingency["human_true_judge_false"] += 1
        else:
            contingency["human_false_judge_true"] += 1

    n_compared = len(pairs)
    raw_agreement = (
        100.0 * sum(1 for a, b in pairs if a == b) / n_compared
        if n_compared
        else 0.0
    )

    kappa, p_o, kappa_note = cohens_kappa(pairs)
    pabak = 2 * p_o - 1 if n_compared else None
    prevalence = (
        sum(1 for a, _ in pairs if a is True) / n_compared if n_compared else None
    )

    report = {
        "rubric_versions": {
            "human": human_rubric_version,
            "judge": judge_rubric_version,
        },
        "rubric_warning": rubric_warning,
        "n_compared": n_compared,
        "n_skipped": n_skipped,
        "contingency": contingency,
        "raw_agreement": raw_agreement,
        "kappa": kappa,
        "pabak": pabak,
        "prevalence": prevalence,
        "kappa_note": kappa_note,
        "per_ticket": per_ticket,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"n_compared={n_compared}  n_skipped={n_skipped}")
    print("Contingency table:")
    print(f"  both_true              = {contingency['both_true']}")
    print(f"  both_false             = {contingency['both_false']}")
    print(f"  human_true_judge_false = {contingency['human_true_judge_false']}")
    print(f"  human_false_judge_true = {contingency['human_false_judge_true']}")
    print(f"raw_agreement = {raw_agreement:.1f}%")
    kappa_str = f"{kappa:.3f}" if kappa is not None else "None"
    pabak_str = f"{pabak:.3f}" if pabak is not None else "None"
    print(f"kappa = {kappa_str}   pabak = {pabak_str}")
    print(f"prevalence (human True fraction) = {prevalence}")
    print(f"note: {kappa_note}")
    print(f"Wrote report to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser(
        "template", help="emit a blind labeling template"
    )
    template_parser.add_argument("--n", type=int, default=25)
    template_parser.add_argument("--ragas", default=str(DEFAULT_RAGAS_PATH))
    template_parser.add_argument("--gap-set", default=str(GAP_SET_PATH))
    template_parser.add_argument("--tickets", default=str(TICKETS_PATH))
    template_parser.add_argument("--out", default=str(DEFAULT_TEMPLATE_OUT))
    template_parser.add_argument(
        "--strategy",
        choices=["disagreement", "faithfulness", "verdict-stratified"],
        default="disagreement",
    )
    template_parser.add_argument("--judge", default=str(DEFAULT_JUDGE_PATH))
    template_parser.set_defaults(func=run_template)

    compare_parser = subparsers.add_parser(
        "compare", help="compute human/judge agreement"
    )
    compare_parser.add_argument("--human", required=True)
    compare_parser.add_argument("--judge", default=str(DEFAULT_JUDGE_PATH))
    compare_parser.add_argument("--out", default=str(DEFAULT_REPORT_OUT))
    compare_parser.set_defaults(func=run_compare)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
