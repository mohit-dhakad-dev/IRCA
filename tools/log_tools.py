"""Fake log and metric query tools.

These simulate the two tools an incident-diagnosis agent would call against a
real observability stack, but everything they return is synthesized from
``tools.fake_data`` — deterministically, from a hash of the call arguments —
so tests and agent runs are fully offline and reproducible.
"""

from __future__ import annotations

import hashlib
import random
import re
from datetime import datetime, timedelta

from tools.fake_data import METRIC_PROFILES, get_ticket, signature_for_ticket

SEVERITY_ORDER = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]

# Fixed base instant everything is derived from. Never datetime.now() — that
# would make tool output non-deterministic across runs.
_BASE_TIMESTAMP = datetime.fromisoformat("2025-06-01T09:00:00")

# Placeholder values used when rendering pattern_counts keys, so the keys are
# stable regardless of which real service/timestamp a call used.
_PLACEHOLDER_TIMESTAMP = "<ts>"
_PLACEHOLDER_SERVICE = "<service>"

_ACCEPTED_LEVELS = ", ".join(SEVERITY_ORDER)

# <positive int><unit>, unit in m|h|d, case-insensitive, surrounding whitespace ok.
_WINDOW_RE = re.compile(r"^(\d+)([mhd])$", re.IGNORECASE)


def _is_ambiguous(ticket: dict) -> bool:
    # Single source of truth for "this ticket has no confident root cause":
    # the same field signature_for_ticket() keys off.
    return ticket.get("gold_root_cause") is None


def _seeded_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return random.Random(int(digest[:8], 16))


def _normalize_level(level: str) -> str | None:
    normalized = level.strip().upper()
    if normalized == "WARNING":
        normalized = "WARN"
    if normalized not in SEVERITY_ORDER:
        return None
    return normalized


def _core_message(rendered_line: str) -> str:
    # Templates render as "<timestamp> <service>: <core message>"; pull just
    # the core message out for quoting in summaries.
    if ": " in rendered_line:
        return rendered_line.split(": ", 1)[1]
    return rendered_line


def _parse_window_minutes(window: str) -> int | None:
    """Parse a window like '30m', '2h', '1d' into minutes.

    Returns None for anything that doesn't match exactly: empty/whitespace,
    missing unit, unknown unit, non-positive amount, or trailing junk. This
    is total (never raises) so invalid tool args become an error observation
    instead of an unhandled exception.
    """
    match = _WINDOW_RE.match(window.strip())
    if not match:
        return None
    amount = int(match.group(1))
    if amount <= 0:
        return None
    unit = match.group(2).lower()
    if unit == "m":
        return amount
    if unit == "h":
        return amount * 60
    return amount * 60 * 24


def _split_counts(total: int, weights: list[float]) -> list[int]:
    """Split `total` across len(weights) buckets proportionally to weights,
    using largest-remainder rounding so the parts sum exactly to total."""
    weight_sum = sum(weights) or 1.0
    raw = [total * (w / weight_sum) for w in weights]
    counts = [int(r) for r in raw]
    remainder = total - sum(counts)
    remainders = sorted(range(len(weights)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[remainders[i % len(remainders)]] += 1
    return counts


def _round_value(raw: float, integer: bool) -> float:
    return int(round(raw)) if integer else round(raw, 2)


def _generate_climbing(rng: random.Random, lo: float, hi: float, n_points: int) -> list[float]:
    """Monotonically non-decreasing series from lo up to hi."""
    if n_points <= 1:
        return [hi]
    weights = [rng.uniform(0.5, 1.5) for _ in range(n_points - 1)]
    weight_sum = sum(weights)
    span = hi - lo
    values = [lo]
    for w in weights:
        values.append(values[-1] + span * (w / weight_sum))
    return values


def _generate_step(
    rng: random.Random,
    normal_lo: float,
    normal_hi: float,
    post_lo: float,
    post_hi: float,
    n_points: int,
) -> list[float]:
    """First ~1/3 of points drawn from the pre-step baseline, the rest from
    the post-step band (which for a low_bad metric is the collapsed band)."""
    split_idx = max(1, n_points // 3)
    values = []
    for i in range(n_points):
        if i < split_idx:
            values.append(rng.uniform(normal_lo, normal_hi))
        else:
            values.append(rng.uniform(post_lo, post_hi))
    return values


def query_logs(ticket_id: str, service: str, window: str, level: str) -> dict:
    """Return a summarised view of synthetic log lines for a ticket/service.

    Never returns raw dumps: at most 5 sample lines plus aggregate counts,
    matching the design constraint that tool output stays small relative to
    the context window.
    """
    ticket = get_ticket(ticket_id)
    if ticket is None:
        return {"status": "error", "data": {}, "summary": f"Unknown ticket id '{ticket_id}'."}

    if _parse_window_minutes(window) is None:
        return {
            "status": "error",
            "data": {},
            "summary": f"Invalid window '{window}'. Expected a value like '30m', '2h', or '1d'.",
        }

    normalized_level = _normalize_level(level)
    if normalized_level is None:
        return {
            "status": "error",
            "data": {},
            "summary": f"Unrecognized level '{level}'. Accepted levels: {_ACCEPTED_LEVELS}.",
        }

    sig = signature_for_ticket(ticket_id)
    ambiguous = _is_ambiguous(ticket)

    sig_level_idx = SEVERITY_ORDER.index(sig["log_level"])
    requested_level_idx = SEVERITY_ORDER.index(normalized_level)

    if sig_level_idx < requested_level_idx:
        zero_counts = {
            pattern.format(timestamp=_PLACEHOLDER_TIMESTAMP, service=_PLACEHOLDER_SERVICE): 0
            for pattern in sig["log_patterns"]
        }
        return {
            "status": "empty",
            "data": {
                "service": service,
                "window": window,
                "level": normalized_level,
                "total_matched": 0,
                "pattern_counts": zero_counts,
                "lines": [],
            },
            "summary": f"No {normalized_level} logs found for {service} in the last {window}.",
        }

    # `level` is deliberately excluded from the seed: the whole draw sequence
    # below (base total, dominant split, filler-per-level) must be identical
    # regardless of which level was requested, so that different levels are
    # just different slices of the same deterministic sequence and totals
    # nest monotonically (DEBUG >= INFO >= WARN >= ERROR).
    rng = _seeded_rng(ticket_id, service, window)
    patterns = sig["log_patterns"]

    if ambiguous:
        base_total = rng.randint(5, 20)
        weights = [1.0 + rng.uniform(0.0, 0.6) for _ in patterns]
        counts = _split_counts(base_total, weights)
    else:
        base_total = rng.randint(150, 400)
        dominant_frac = rng.uniform(0.55, 0.75)
        dominant_count = round(base_total * dominant_frac)
        remaining = base_total - dominant_count
        other_weights = [1.0 + rng.uniform(0.0, 1.0) for _ in patterns[1:]]
        other_counts = _split_counts(remaining, other_weights) if other_weights else []
        counts = [dominant_count] + other_counts

    # Benign filler (routine chatter unrelated to the incident) drawn for
    # every level below the signature's own level, in a fixed order. This
    # never touches pattern_counts, so the signature's own pattern counts
    # stay identical across levels — only total_matched grows as the query
    # widens to include more routine, lower-severity noise.
    filler_by_level = [rng.randint(15, 70) for _ in range(sig_level_idx)]
    filler_total = sum(filler_by_level[requested_level_idx:sig_level_idx])
    total_matched = base_total + filler_total

    pattern_counts = {}
    for pattern, count in zip(patterns, counts):
        key = pattern.format(timestamp=_PLACEHOLDER_TIMESTAMP, service=_PLACEHOLDER_SERVICE)
        pattern_counts[key] = count

    n_lines = min(5, total_matched)
    lines = []
    for i in range(n_lines):
        pattern = rng.choices(patterns, weights=counts, k=1)[0]
        line_ts = (_BASE_TIMESTAMP - timedelta(minutes=3 * i)).isoformat()
        rendered = pattern.format(timestamp=line_ts, service=service)
        lines.append(
            {
                "timestamp": line_ts,
                "level": sig["log_level"],
                "service": service,
                "message": rendered,
            }
        )

    data = {
        "service": service,
        "window": window,
        "level": normalized_level,
        "total_matched": total_matched,
        "pattern_counts": pattern_counts,
        "lines": lines,
    }

    if ambiguous:
        summary = (
            f"Found {total_matched} {normalized_level} lines for {service} in the last "
            f"{window} across {len(patterns)} unrelated patterns; no dominant error signature."
        )
    else:
        dominant_pattern, dominant_count = max(pattern_counts.items(), key=lambda kv: kv[1])
        pct = round(100 * dominant_count / total_matched)
        dominant_message = _core_message(dominant_pattern)
        summary = (
            f"Found {total_matched} {normalized_level} lines for {service} in the last "
            f"{window}; {pct}% match '{dominant_message}'."
        )

    return {"status": "ok", "data": data, "summary": summary}


def query_metrics(ticket_id: str, service: str, metric: str, window: str) -> dict:
    """Return a summarised synthetic time series for a metric/ticket/service.

    The anomaly only ever appears on the metric that matches the ticket's
    gold root-cause signature; every other known metric comes back normal.
    """
    ticket = get_ticket(ticket_id)
    if ticket is None:
        return {"status": "error", "data": {}, "summary": f"Unknown ticket id '{ticket_id}'."}

    minutes = _parse_window_minutes(window)
    if minutes is None:
        return {
            "status": "error",
            "data": {},
            "summary": f"Invalid window '{window}'. Expected a value like '30m', '2h', or '1d'.",
        }

    metric = metric.strip()
    if metric not in METRIC_PROFILES:
        available = sorted(METRIC_PROFILES)
        return {
            "status": "unknown_metric",
            "data": {"requested": metric, "available_metrics": available},
            "summary": f"Unknown metric '{metric}'. Available metrics: {', '.join(available)}.",
        }

    sig = signature_for_ticket(ticket_id)
    ambiguous = _is_ambiguous(ticket)
    profile = METRIC_PROFILES[metric]
    is_anomaly = metric == sig["metric_name"]

    normal_low, normal_high = profile["normal_range"]
    anomaly_low, anomaly_high = profile["anomaly_range"]

    if is_anomaly and ambiguous:
        # Weakly elevated: brushes the top of normal_range, never a clean
        # threshold breach into the full anomaly_range.
        span = normal_high - normal_low
        lo = normal_high - 0.1 * span
        hi = normal_high + 0.08 * span
        kind = "ambiguous"
    elif is_anomaly:
        lo, hi = anomaly_low, anomaly_high
        kind = "anomaly"
    else:
        lo, hi = normal_low, normal_high
        kind = "normal"

    rng = _seeded_rng(ticket_id, service, metric, window)
    n_points = 12
    step = max(minutes // n_points, 1)

    shape = profile.get("shape", "sustained")
    integer = profile.get("integer", False)

    if kind == "anomaly" and shape == "climbing":
        raw_values = _generate_climbing(rng, lo, hi, n_points)
    elif kind == "anomaly" and shape in ("step_up", "step_down"):
        raw_values = _generate_step(rng, normal_low, normal_high, lo, hi, n_points)
    else:
        # "sustained" anomaly, the ambiguous weak-elevation band, and the
        # normal (non-matching-metric) case are all a flat uniform draw
        # inside their respective [lo, hi] band.
        raw_values = [rng.uniform(lo, hi) for _ in range(n_points)]

    points = []
    for i, raw in enumerate(raw_values):
        t_offset = (n_points - 1 - i) * step
        t_label = (_BASE_TIMESTAMP - timedelta(minutes=t_offset)).isoformat()
        points.append({"t": t_label, "value": _round_value(raw, integer)})

    values = [p["value"] for p in points]
    if integer:
        min_v = int(min(values))
        max_v = int(max(values))
        avg_v = round(sum(values) / len(values), 1)
        current_v = int(values[-1])
    else:
        min_v = round(min(values), 2)
        max_v = round(max(values), 2)
        avg_v = round(sum(values) / len(values), 2)
        current_v = values[-1]

    data = {
        "service": service,
        "metric": metric,
        "window": window,
        "unit": profile["unit"],
        "capacity_max": profile["capacity_max"],
        "points": points,
        "min": min_v,
        "max": max_v,
        "avg": avg_v,
        "current": current_v,
    }

    unit = profile["unit"]
    cap = profile["capacity_max"]
    direction = profile.get("direction", "high_bad")
    # 100% is the definition of a percent unit, not a genuine configured
    # ceiling, so the "against a configured max" clause is only meaningful
    # for metrics with a real, non-percent-implied capacity (e.g. a
    # connection pool or a listen queue with its own numeric limit).
    show_cap = cap is not None and not (unit == "percent" and cap == 100.0)

    if kind == "anomaly":
        if shape == "climbing":
            start_v = points[0]["value"]
            cap_clause = f" — approaching the configured max of {cap:g} {unit}" if show_cap else ""
            summary = (
                f"{metric} for {service} is climbing — now at {current_v} {unit} "
                f"(started this window at {start_v}) and still rising over {window} "
                f"(avg {avg_v}){cap_clause}."
            )
        elif shape == "step_up":
            summary = (
                f"{metric} for {service} jumped from under {normal_high:g} {unit} to "
                f"{anomaly_low:g}-{anomaly_high:g} {unit} partway through the last {window} "
                f"(avg {avg_v}, peak {max_v})."
            )
        elif shape == "step_down" or direction == "low_bad":
            summary = (
                f"{metric} for {service} collapsed to {current_v} {unit} (min {min_v}) over "
                f"{window}, down from a healthy baseline near {normal_high:g} {unit} — well "
                f"below the expected range."
            )
        elif show_cap:
            summary = (
                f"{metric} for {service} is pegged at {min_v}-{max_v} {unit} against a "
                f"configured max of {cap:g} for the last {window} (avg {avg_v})."
            )
        elif cap is not None:
            summary = (
                f"{metric} for {service} is pegged at {min_v}-{max_v} {unit} for the "
                f"last {window} (avg {avg_v})."
            )
        else:
            summary = (
                f"{metric} for {service} is elevated at {min_v}-{max_v} {unit} for the "
                f"last {window} (avg {avg_v}) — well above the healthy range."
            )
    elif kind == "ambiguous":
        summary = (
            f"{metric} for {service} is mildly elevated ({min_v}-{max_v} {unit}, avg {avg_v}) "
            f"over {window} — brushing the top of the normal range but does not breach any threshold."
        )
    else:
        summary = (
            f"{metric} for {service} averaged {avg_v} {unit} over {window} "
            f"(peak {max_v}) — within normal range."
        )

    return {"status": "ok", "data": data, "summary": summary}
