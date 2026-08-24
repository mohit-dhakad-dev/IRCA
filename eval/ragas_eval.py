"""RAGAS evaluation over the reconstructed ticket answers in
``eval/results/ragas_inputs.json``.

This module MUST run under the separate ``venv-ragas`` interpreter, never
under the project ``venv``:

    venv-ragas/bin/python eval/ragas_eval.py --gap-set --dry-run
    venv-ragas/bin/python eval/ragas_eval.py --gap-set --subset 3

RAGAS cannot be installed in the project venv -- it drags in openai 3.3.1,
which downgrades the ``openai`` SDK that ``agent/llm.py`` depends on. See
``eval/requirements-ragas.txt`` for exactly what is installed in
``venv-ragas`` and why.

Consequently this module imports NOTHING from ``agent/``, ``rag/``,
``tools/``, ``vectorstore.py``, or ``eval/report.py`` -- none of those are
importable under venv-ragas. It reads its input only from the JSON file
produced by ``eval/ragas_inputs.py`` (a pure, network-free reconstruction
step that already ran under the project venv).

The pure helpers below (input loading/filtering, NaN accounting, the
dry-run call/token estimator) are importable without ``ragas`` installed --
all ``ragas``/``langchain`` imports live inside ``run_evaluation``, which is
only ever called from ``main()`` under venv-ragas. This is what lets
``tests/test_ragas_eval.py`` run under the project venv, where ragas is
absent, and still exercise real logic instead of skipping outright.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUTS_PATH = REPO_ROOT / "eval" / "results" / "ragas_inputs.json"
DEFAULT_GAP_SET_PATH = REPO_ROOT / "eval" / "results" / "gap_set.json"
DEFAULT_OUT_PATH = REPO_ROOT / "eval" / "results" / "ragas_scores.json"
ENV_PATH = REPO_ROOT / ".env"

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]

# Per-sample LLM-call counts used by --dry-run. These are ESTIMATES based on
# each metric's documented algorithm, not measurements -- the actual per-run
# call count is captured (where possible) from the real evaluate() run and
# reported separately in the "usage" block. faithfulness: 1 call to extract
# claims + 1 call to verify them against contexts = 2. answer_relevancy: with
# strictness=1 (see below), 1 generation call. context_recall: 1 call
# (classifies the reference sentence-by-sentence against contexts in a single
# prompt). context_precision (LLMContextPrecisionWithReference): 1 call per
# retrieved context, since it scores each context's relevance individually.
#
# Note on key names: these are the canonical output keys this module uses
# (matching METRIC_NAMES), NOT necessarily the ragas class names -- e.g.
# ragas's LLMContextRecall metric class reports its score under
# `.name == "context_recall"`, not "llm_context_recall". run_evaluation()
# derives the actual lookup key from each live metric instance's `.name`
# rather than hardcoding it a second time here, so this dict is only ever
# used for the ragas-free dry-run estimate, never for reading real results.
CALLS_PER_SAMPLE = {
    "faithfulness": 2,
    "answer_relevancy": 1,
    "context_recall": 1,
    "llm_context_precision_with_reference": None,  # depends on len(contexts)
}

# AnswerRelevancy is pinned to strictness=1 rather than the ragas default of
# 3. Default strictness=3 requests n=3 chat completions in a single request
# (via the `n` parameter) to average multiple paraphrase-question samples.
# Our Baseten-hosted openai/gpt-oss-120b endpoint does not honour `n` and
# only ever returns 1 generation, logging "LLM returned 1 generations
# instead of requested 3. Proceeding with 1 generations." -- so the default
# already silently degrades to n=1 while claiming (via its own docstring and
# the "3" in the code) to average three. Pinning strictness=1 makes that
# degradation explicit and honest instead of hidden behind a default that
# lies about what it computed.
ANSWER_RELEVANCY_STRICTNESS = 1

# RunConfig defaults. The probe observed ~57s per metric-job per sample for
# LLMContextPrecisionWithReference, which issues one call per context and we
# always have 5 contexts per sample -- so a single sample's context-precision
# job alone can approach RAGAS's old 180s default timeout, producn NaN cells
# from spurious TimeoutErrors rather than genuine model failures. 600s gives
# ample headroom; max_workers=4 balances wall-clock time against rate limits
# on the shared LLM endpoint.
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_WORKERS = 4


# --------------------------------------------------------------------------
# Pure helpers -- no ragas/langchain imports here, must work under the
# project venv (ragas absent) as well as venv-ragas.
# --------------------------------------------------------------------------


def load_inputs(path: Path = DEFAULT_INPUTS_PATH) -> list[dict[str, Any]]:
    """Load the list of ragas-input rows written by eval/ragas_inputs.py."""
    return json.loads(Path(path).read_text())


def load_gap_set_ids(path: Path = DEFAULT_GAP_SET_PATH) -> set[str]:
    """Load the ticket_ids named in eval/results/gap_set.json's tickets[]."""
    data = json.loads(Path(path).read_text())
    return {t["ticket_id"] for t in data["tickets"]}


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    only: list[str] | None = None,
    gap_set_ids: set[str] | None = None,
    subset: int | None = None,
) -> list[dict[str, Any]]:
    """Apply --only / --gap-set / --subset filtering, in that order, with a
    deterministic ticket_id sort so --subset is reproducible."""
    result = sorted(rows, key=lambda r: r["ticket_id"])
    if only is not None:
        wanted = set(only)
        result = [r for r in result if r["ticket_id"] in wanted]
    if gap_set_ids is not None:
        result = [r for r in result if r["ticket_id"] in gap_set_ids]
    if subset is not None:
        result = result[:subset]
    return result


def _is_nan(value: Any) -> bool:
    """True for NaN floats AND for anything that isn't a real float/int
    score (None, missing, non-numeric) -- both count as "not scored"."""
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return isinstance(value, float) and math.isnan(value)
    return True


def summarize_metric(values: list[Any]) -> dict[str, Any]:
    """Explicit NaN accounting for one metric's per-sample scores.

    Never returns a bare mean: mean is computed only over the real
    (non-NaN) values, and is None (not 0.0) when there are zero of them.
    """
    n_total = len(values)
    real_values = [float(v) for v in values if not _is_nan(v)]
    n_scored = len(real_values)
    n_nan = n_total - n_scored
    mean = sum(real_values) / n_scored if n_scored > 0 else None
    return {
        "mean": mean,
        "n_scored": n_scored,
        "n_total": n_total,
        "n_nan": n_nan,
        "complete": n_nan == 0,
    }


def metric_summary_text(name: str, summary: dict[str, Any]) -> str:
    """Human-readable one-line summary, flagging incompleteness."""
    if summary["n_scored"] == 0:
        return f"{name}: mean=None (0/{summary['n_total']} scored)"
    text = f"{name}: mean={summary['mean']:.4f} over {summary['n_scored']}/{summary['n_total']}"
    if not summary["complete"]:
        text += f" -- INCOMPLETE, {summary['n_nan']} NaN"
    return text


def estimate_calls_for_row(row: dict[str, Any], metric_names: list[str] = METRIC_NAMES) -> dict[str, int]:
    """Projected LLM-call count per metric for one row, per CALLS_PER_SAMPLE,
    restricted to ``metric_names``. context precision issues one call per
    retrieved context."""
    n_contexts = len(row.get("contexts") or [])
    all_calls = {
        "faithfulness": CALLS_PER_SAMPLE["faithfulness"],
        "answer_relevancy": CALLS_PER_SAMPLE["answer_relevancy"],
        "context_recall": CALLS_PER_SAMPLE["context_recall"],
        "llm_context_precision_with_reference": n_contexts,
    }
    return {name: all_calls[name] for name in metric_names}


def estimate_dry_run(rows: list[dict[str, Any]], metric_names: list[str] = METRIC_NAMES) -> dict[str, Any]:
    """Projected call counts and prompt-token volume for --dry-run, over
    only ``metric_names`` (default: all of METRIC_NAMES). Token volume is
    estimated from actual input character lengths (chars/4), a coarse
    standard heuristic, not a tokenizer measurement."""
    per_metric_calls: dict[str, int] = {name: 0 for name in metric_names}
    total_chars = 0
    for row in rows:
        calls = estimate_calls_for_row(row, metric_names)
        for name, n in calls.items():
            per_metric_calls[name] += n
        total_chars += (
            len(row.get("question") or "")
            + len(row.get("answer") or "")
            + len(row.get("ground_truth") or "")
            + sum(len(c) for c in row.get("contexts") or [])
        )
    total_calls = sum(per_metric_calls.values())
    return {
        "n_tickets": len(rows),
        "per_metric_calls": per_metric_calls,
        "total_calls": total_calls,
        "estimated_prompt_tokens": total_chars // 4,
    }


def parse_metrics_arg(value: str | None) -> list[str]:
    """Parse the --metrics flag into a list of canonical metric keys, in
    canonical METRIC_NAMES order (not the order the user typed them).

    ``value`` is None or empty -> all of METRIC_NAMES (unchanged default
    behaviour). Otherwise it is a comma-separated list of canonical keys;
    any key not in METRIC_NAMES raises ValueError naming the bad key and
    the full list of valid options -- an unknown key is never silently
    ignored.
    """
    if not value:
        return list(METRIC_NAMES)
    requested = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in requested if s not in METRIC_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown metric key(s) {unknown} passed to --metrics. "
            f"Valid keys are: {METRIC_NAMES}."
        )
    selected = set(requested)
    return [name for name in METRIC_NAMES if name in selected]


def dropped_metrics(selected: list[str]) -> list[str]:
    """The canonical keys in METRIC_NAMES that are NOT in ``selected``, in
    canonical order -- the complement recorded as config.metrics_dropped."""
    selected_set = set(selected)
    return [name for name in METRIC_NAMES if name not in selected_set]


def validate_pairing(pairs: list[tuple[str, str]], expected_keys: list[str]) -> None:
    """Guard against the canonical-key -> metric mapping drifting from
    METRIC_NAMES.

    ``pairs`` is a list of (canonical_key, metric_name) tuples -- the same
    shape as ``[(key, metric.name) for key, metric in METRIC_PAIRS]`` in
    ``run_evaluation``, but taking plain strings so this can be exercised
    with fakes and stays importable without ragas.

    Raises RuntimeError if the canonical keys don't cover ``expected_keys``
    exactly, or if a canonical key is duplicated (which would silently drop
    one metric's mapping in favour of the other).
    """
    seen: dict[str, str] = {}
    duplicates = []
    for key, name in pairs:
        if key in seen:
            duplicates.append(key)
        seen[key] = name

    if duplicates:
        raise RuntimeError(
            f"Duplicate canonical key(s) {sorted(set(duplicates))} in metric pairing -- "
            f"each canonical key must map to exactly one metric."
        )

    actual_keys = set(seen)
    expected = set(expected_keys)
    if actual_keys != expected:
        missing = sorted(expected - actual_keys)
        extra = sorted(actual_keys - expected)
        raise RuntimeError(
            f"Metric pairing keys do not match METRIC_NAMES -- "
            f"missing={missing}, extra={extra}."
        )


def validate_metric_keys(available_keys: Any, expected_names: list[str]) -> None:
    """Guard against a metric-key lookup drifting from what ragas actually
    emits (the exact bug this function exists to prevent recurring: ragas's
    LLMContextRecall class reports under `.name == "context_recall"`, not
    "llm_context_recall", and a hardcoded lookup silently discarded every
    score as NaN instead of failing loudly).

    ``available_keys`` is normally one row of ``result.scores`` (a dict);
    any object supporting ``in`` / iteration of keys works for testing.
    Raises RuntimeError naming exactly which expected key(s) are missing
    and what keys were actually available, rather than letting a missing
    key silently degrade into an all-NaN metric.
    """
    missing = [name for name in expected_names if name not in available_keys]
    if missing:
        available = sorted(available_keys) if hasattr(available_keys, "__iter__") else available_keys
        raise RuntimeError(
            f"ragas result.scores is missing expected metric key(s) {missing}. "
            f"This means a metric's `.name` no longer matches what this module "
            f"expects -- available keys were: {available}."
        )


def _require_env(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(
            f"{name} is not set after loading {ENV_PATH} -- refusing to proceed "
            f"with a None/empty value. Set it in .env (see .env.example)."
        )
    return value


def load_provider_config() -> dict[str, str]:
    """Load LLM_MODEL / LLM_BASE_URL / LLM_API_KEY from the repo-root .env,
    resolved explicitly by path (load_dotenv() alone resolves relative to
    the CALLING SCRIPT's cwd, which silently returned LLM_MODEL=None when
    this was run as a standalone script from elsewhere). Fails loudly if
    anything required is missing or empty -- never proceeds with None."""
    import os

    from dotenv import load_dotenv

    load_dotenv(dotenv_path=ENV_PATH)

    model = _require_env("LLM_MODEL", os.environ.get("LLM_MODEL"))
    base_url = _require_env("LLM_BASE_URL", os.environ.get("LLM_BASE_URL"))
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
    api_key = _require_env("LLM_API_KEY (or GROQ_API_KEY)", api_key)

    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    return {"model": model, "base_url": base_url, "api_key": api_key}


# --------------------------------------------------------------------------
# The actual evaluation -- ragas/langchain imports are local to this
# function so the module stays importable without ragas installed.
# --------------------------------------------------------------------------


def run_evaluation(
    rows: list[dict[str, Any]],
    *,
    metric_names: list[str] = METRIC_NAMES,
    timeout: int = DEFAULT_TIMEOUT,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    """Run the four RAGAS metrics over rows and return the full output dict
    described in the module's spec (schema_version, provider, config,
    metrics, usage, per_ticket). Makes real network calls to the LLM."""
    from ragas import RunConfig, SingleTurnSample, EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    provider = load_provider_config()

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=provider["model"],
            base_url=provider["base_url"],
            api_key=provider["api_key"],
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # Single authoritative structure pairing each canonical output key with
    # the ragas metric instance that produces it. `metrics` and the
    # name->canonical mapping are BOTH derived from this one list, so the
    # two can never drift apart.
    #
    # A positional `zip(metric_names, METRIC_NAMES)` was rejected here: it
    # relies on two separately-maintained lists staying in the same order,
    # and if they ever drift (reorder, insertion) it fails INVISIBLY --
    # every key is still present, so validate_metric_keys() passes clean,
    # no NaN appears, and the output looks entirely healthy while scores are
    # silently transposed between metrics (e.g. faithfulness's numbers
    # written under answer_relevancy's key). zip also silently truncates if
    # the lists differ in length. An explicit, literal pairing has no such
    # failure mode.
    all_metric_pairs: list[tuple[str, Any]] = [
        ("faithfulness", Faithfulness()),
        ("answer_relevancy", AnswerRelevancy(strictness=ANSWER_RELEVANCY_STRICTNESS)),
        ("llm_context_precision_with_reference", LLMContextPrecisionWithReference()),
        ("context_recall", LLMContextRecall()),
    ]
    validate_pairing([(key, m.name) for key, m in all_metric_pairs], METRIC_NAMES)

    selected = set(metric_names)
    metric_pairs = [(key, m) for key, m in all_metric_pairs if key in selected]

    metrics = [m for _, m in metric_pairs]

    samples = [
        SingleTurnSample(
            user_input=row["question"],
            retrieved_contexts=row["contexts"],
            response=row["answer"],
            reference=row["ground_truth"],
        )
        for row in rows
    ]
    # Derive the result-lookup keys from the live metric instances' `.name`
    # attribute rather than hardcoding them -- this is the fix for the bug
    # where "llm_context_recall" was hardcoded but ragas actually reports
    # LLMContextRecall's score under "context_recall". METRIC_NAMES is kept
    # only as the canonical *output* key ordering (used by dry-run/summary,
    # which must stay importable without ragas); this dict is the
    # authoritative name -> canonical-key mapping for reading real results,
    # built from metric_pairs (see comment above) rather than positional zip.
    metric_name_to_canonical = {m.name: key for key, m in metric_pairs}

    dataset = EvaluationDataset(samples=samples)
    run_config = RunConfig(timeout=timeout, max_workers=max_workers)

    # Token/call capture: langchain_community.callbacks.manager's
    # get_openai_callback (the mechanism specified for this) fails to import
    # in venv-ragas -- langchain_community 0.3.31's callback manager imports
    # a CometTracer that in turn imports
    # langchain_core.tracers.langchain_v1.LangChainTracerV1, which does not
    # exist in the installed langchain_core 1.6.0. This is a real
    # incompatibility between the two pinned packages, not something fixable
    # from this module. Ragas ships its own equivalent mechanism
    # (evaluate(..., token_usage_parser=...) + result.cost_cb.usage_data)
    # that reads token usage directly off each LLM response and does not
    # depend on the broken import, so that is used instead. If it also
    # yields nothing (e.g. the provider doesn't populate token_usage on the
    # response), usage is honestly reported as NOT CAPTURED rather than as
    # zeros.
    try:
        from ragas.cost import get_token_usage_for_openai

        token_usage_parser = get_token_usage_for_openai
    except ImportError:
        token_usage_parser = None

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        token_usage_parser=token_usage_parser,
        raise_exceptions=False,
    )

    if result.scores:
        validate_metric_keys(result.scores[0], list(metric_name_to_canonical))

    prompt_tokens = completion_tokens = llm_calls = 0
    captured = False
    if result.cost_cb is not None and result.cost_cb.usage_data:
        llm_calls = len(result.cost_cb.usage_data)
        prompt_tokens = sum(u.input_tokens for u in result.cost_cb.usage_data)
        completion_tokens = sum(u.output_tokens for u in result.cost_cb.usage_data)
        captured = (prompt_tokens + completion_tokens) > 0

    metrics_out: dict[str, Any] = {}
    for metric_name, canonical in metric_name_to_canonical.items():
        values = [row_scores.get(metric_name) for row_scores in result.scores]
        metrics_out[canonical] = summarize_metric(values)

    per_ticket = []
    for row, row_scores in zip(rows, result.scores):
        entry = {"ticket_id": row["ticket_id"], "category": row["category"]}
        for metric_name, canonical in metric_name_to_canonical.items():
            value = row_scores.get(metric_name)
            entry[canonical] = None if _is_nan(value) else float(value)
        per_ticket.append(entry)

    return {
        "schema_version": 1,
        "provider": {"base_url": provider["base_url"], "model": provider["model"]},
        "config": {
            "subset": len(rows),
            "gap_set": None,  # filled in by main() with the actual flag
            "timeout": timeout,
            "max_workers": max_workers,
            "strictness": ANSWER_RELEVANCY_STRICTNESS,
            "n_tickets": len(rows),
            "metrics_selected": list(metric_names),
            "metrics_dropped": dropped_metrics(list(metric_names)),
        },
        "metrics": metrics_out,
        "usage": {
            "captured": captured,
            "prompt_tokens": prompt_tokens if captured else None,
            "completion_tokens": completion_tokens if captured else None,
            "total_tokens": (prompt_tokens + completion_tokens) if captured else None,
            "llm_calls": llm_calls if captured else None,
        },
        "per_ticket": per_ticket,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=str, default=str(DEFAULT_INPUTS_PATH))
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--gap-set", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help=(
            "Comma-separated canonical metric keys to run (default: all four). "
            f"Valid keys: {','.join(METRIC_NAMES)}."
        ),
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        metric_names = parse_metrics_arg(args.metrics)
    except ValueError as exc:
        print(f"FAILURE: {exc}")
        return 1

    rows = load_inputs(Path(args.inputs))
    only = args.only.split(",") if args.only else None
    gap_set_ids = load_gap_set_ids() if args.gap_set else None
    rows = filter_rows(rows, only=only, gap_set_ids=gap_set_ids, subset=args.subset)

    if not rows:
        print("FAILURE: no rows selected after filtering -- nothing to evaluate.")
        return 1

    if args.dry_run:
        estimate = estimate_dry_run(rows, metric_names)
        print("DRY RUN -- projections only, no network calls made.")
        print(f"tickets: {estimate['n_tickets']}")
        print("projected LLM calls per metric (estimates, to be corrected by measurement):")
        for name, n in estimate["per_metric_calls"].items():
            print(f"  {name}: {n}")
        print(f"projected total LLM calls: {estimate['total_calls']}")
        print(
            f"projected prompt tokens (chars/4 over question+answer+ground_truth+contexts): "
            f"{estimate['estimated_prompt_tokens']}"
        )
        dropped = dropped_metrics(metric_names)
        if dropped:
            print(f"metrics dropped (not selected): {dropped}")
        return 0

    try:
        output = run_evaluation(
            rows, metric_names=metric_names, timeout=args.timeout, max_workers=args.max_workers
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure unmistakably
        print(f"FAILURE: ragas evaluation raised {type(exc).__name__}: {exc}")
        return 1

    output["config"]["gap_set"] = bool(args.gap_set)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(f"Evaluated {len(rows)} tickets. Wrote {out_path}")
    print(f"provider: {output['provider']}")
    if output["config"]["metrics_dropped"]:
        print(f"metrics dropped (not selected): {output['config']['metrics_dropped']}")
    for name in metric_names:
        print("  " + metric_summary_text(name, output["metrics"][name]))
    usage = output["usage"]
    if usage["captured"]:
        print(
            f"usage: CAPTURED prompt_tokens={usage['prompt_tokens']} "
            f"completion_tokens={usage['completion_tokens']} "
            f"total_tokens={usage['total_tokens']} llm_calls={usage['llm_calls']}"
        )
    else:
        print("usage: NOT CAPTURED (token usage was unavailable from this run -- see comments in run_evaluation)")

    incomplete = [name for name in metric_names if not output["metrics"][name]["complete"]]
    if incomplete:
        print(f"WARNING: metrics with NaN cells (means computed over fewer samples than n_total): {incomplete}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
