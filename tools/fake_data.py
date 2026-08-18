"""Synthetic data for the fake tool layer.

Everything in this module is fabricated for the purposes of the IRCA project:
root-cause "signatures" (what a log/metric anomaly looks like for a given
incident type) and the ticket dataset loader. None of it talks to a real
telemetry backend, log store, or metrics system — it exists purely so the
agent's tools have deterministic, offline data to reason over.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

TICKETS_PATH = Path(__file__).resolve().parents[1] / "data" / "tickets.json"


# ---------------------------------------------------------------------------
# Root-cause signatures
# ---------------------------------------------------------------------------
# Each entry maps a gold_root_cause slug (or None, for the deliberately
# ambiguous tickets) to the shape of evidence that would corroborate it:
# the log lines an agent would see, and the metric that carries the anomaly.
ROOT_CAUSE_SIGNATURES: dict[str | None, dict] = {
    "db_connection_pool_exhaustion": {
        "log_patterns": [
            "{timestamp} {service}: ConnectionPoolTimeoutException: pool exhausted, "
            "timeout waiting for connection",
            "{timestamp} {service}: timeout waiting for connection from database pool",
            "{timestamp} {service}: too many connections open, request rejected",
        ],
        "log_level": "ERROR",
        "metric_name": "db_pool_active_connections",
        "metric_behavior": (
            "pegged at/near the configured max of 100 for the last ~40 minutes"
        ),
    },
    "memory_cache_overgrowth": {
        "log_patterns": [
            "{timestamp} {service}: OOM command not allowed when used memory > 'maxmemory'",
            "{timestamp} {service}: evicted_keys increasing rapidly under memory pressure",
            "{timestamp} {service}: used_memory_rss tracking near the node limit",
        ],
        "log_level": "ERROR",
        "metric_name": "cache_used_memory_percent",
        "metric_behavior": "above 90% for 15-30 min while hit rate falls",
    },
    "network_ingress_queue_exhaustion": {
        "log_patterns": [
            "{timestamp} {service}: upstream connect error while proxying request",
            "{timestamp} {service}: connection reset by peer during handshake",
            "{timestamp} {service}: alert fired for ingress_max_connections_exceeded",
        ],
        "log_level": "ERROR",
        "metric_name": "ingress_listen_queue_depth",
        "metric_behavior": (
            "sustained near 85-100% of the configured listen queue during peak"
        ),
    },
    "deploy_healthcheck_misconfiguration": {
        "log_patterns": [
            "{timestamp} {service}: readiness probe failed: connection refused",
            "{timestamp} {service}: probe failed: HTTP 503",
            "{timestamp} {service}: pod marked CrashLoopBackOff after repeated restarts",
        ],
        "log_level": "ERROR",
        "metric_name": "readiness_probe_success_rate",
        "metric_behavior": (
            "collapses from ~100% to near 0% right after the new revision rolls out"
        ),
    },
    "auth_signing_key_mismatch": {
        "log_patterns": [
            "{timestamp} {service}: signature verification failed for incoming request",
            "{timestamp} {service}: invalid token presented by client",
            "{timestamp} {service}: unable to verify JWT against current signing key",
        ],
        "log_level": "ERROR",
        "metric_name": "auth_token_verification_failure_rate",
        "metric_behavior": (
            "jumps from <1% to 20-40% immediately after a key rotation/deploy, "
            "only on a subset of nodes"
        ),
    },
    "disk_log_rotation_gap": {
        "log_patterns": [
            "{timestamp} {service}: No space left on device",
            "{timestamp} {service}: failed to rotate log file, disk full",
            "{timestamp} {service}: cannot create temp file, filesystem full",
        ],
        "log_level": "ERROR",
        "metric_name": "disk_usage_percent",
        "metric_behavior": "climbing monotonically past 93% and still rising, no rotation drop-offs",
    },
    None: {
        "log_patterns": [
            "{timestamp} {service}: slow upstream call exceeded latency budget",
            "{timestamp} {service}: request retried once after transient failure",
            "{timestamp} {service}: transient HTTP 499 client closed request",
            "{timestamp} {service}: cache miss burst detected, elevated lookup latency",
        ],
        "log_level": "WARN",
        "metric_name": "request_latency_p95",
        "metric_behavior": "mildly and intermittently elevated with no sustained spike and no threshold breach",
    },
}


# ---------------------------------------------------------------------------
# Metric profiles
# ---------------------------------------------------------------------------
# "direction" tells the tool layer which end of the range is bad:
#   "high_bad" (default sense for most telemetry) - high values are the problem.
#   "low_bad"   - low values are the problem (e.g. a success rate collapsing).
# "shape" describes how the anomaly plays out over the queried window, so the
# generated series actually matches the prose in ROOT_CAUSE_SIGNATURES:
#   "sustained" - flat-ish, uniformly inside anomaly_range the whole window.
#   "climbing"  - monotonically non-decreasing from the low to the high end
#                 of anomaly_range.
#   "step_up"   - starts in normal_range, then jumps to anomaly_range partway
#                 through the window.
#   "step_down" - same shape as step_up (starts normal, steps partway through);
#                 for a low_bad metric this renders as a collapse.
# "integer" marks metrics whose unit is a discrete count, so points/min/max/
# current render as ints instead of 2dp floats.
METRIC_PROFILES: dict[str, dict] = {
    "db_pool_active_connections": {
        "unit": "connections",
        "capacity_max": 100.0,
        "normal_range": (20.0, 60.0),
        "anomaly_range": (96.0, 100.0),
        "direction": "high_bad",
        "shape": "sustained",
        "integer": True,
    },
    "cache_used_memory_percent": {
        "unit": "percent",
        "capacity_max": 100.0,
        "normal_range": (40.0, 70.0),
        "anomaly_range": (90.0, 99.0),
        "direction": "high_bad",
        "shape": "sustained",
        "integer": False,
    },
    "ingress_listen_queue_depth": {
        "unit": "connections",
        "capacity_max": 512.0,
        "normal_range": (10.0, 60.0),
        "anomaly_range": (435.0, 512.0),
        "direction": "high_bad",
        "shape": "sustained",
        "integer": True,
    },
    "readiness_probe_success_rate": {
        "unit": "percent",
        "capacity_max": 100.0,
        "normal_range": (95.0, 100.0),
        "anomaly_range": (0.0, 15.0),
        "direction": "low_bad",
        "shape": "step_down",
        "integer": False,
    },
    "auth_token_verification_failure_rate": {
        "unit": "percent",
        "capacity_max": 100.0,
        "normal_range": (0.0, 1.0),
        "anomaly_range": (20.0, 40.0),
        "direction": "high_bad",
        "shape": "step_up",
        "integer": False,
    },
    "disk_usage_percent": {
        "unit": "percent",
        "capacity_max": 100.0,
        "normal_range": (40.0, 70.0),
        "anomaly_range": (93.0, 100.0),
        "direction": "high_bad",
        "shape": "climbing",
        "integer": False,
    },
    "request_latency_p95": {
        "unit": "ms",
        "capacity_max": None,
        "normal_range": (80.0, 220.0),
        "anomaly_range": (210.0, 260.0),
        "direction": "high_bad",
        "shape": "sustained",
        "integer": False,
    },
    # Decoy metrics: plausible things an agent might ask for, never carrying
    # the anomaly for any signature above.
    "cpu_percent": {
        "unit": "percent",
        "capacity_max": 100.0,
        "normal_range": (15.0, 45.0),
        "anomaly_range": (15.0, 45.0),
        "direction": "high_bad",
        "shape": "sustained",
        "integer": False,
    },
    "memory_percent": {
        "unit": "percent",
        "capacity_max": 100.0,
        "normal_range": (30.0, 65.0),
        "anomaly_range": (30.0, 65.0),
        "direction": "high_bad",
        "shape": "sustained",
        "integer": False,
    },
    "error_rate": {
        "unit": "percent",
        "capacity_max": None,
        "normal_range": (0.0, 2.0),
        "anomaly_range": (0.0, 2.0),
        "direction": "high_bad",
        "shape": "sustained",
        "integer": False,
    },
    "request_count": {
        "unit": "requests_per_min",
        "capacity_max": None,
        "normal_range": (500.0, 2000.0),
        "anomaly_range": (500.0, 2000.0),
        "direction": "high_bad",
        "shape": "sustained",
        "integer": True,
    },
    "network_throughput_mbps": {
        "unit": "mbps",
        "capacity_max": None,
        "normal_range": (50.0, 300.0),
        "anomaly_range": (50.0, 300.0),
        "direction": "high_bad",
        "shape": "sustained",
        "integer": False,
    },
}


# ---------------------------------------------------------------------------
# Ticket loading
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_tickets() -> list[dict]:
    return json.loads(TICKETS_PATH.read_text())


def get_ticket(ticket_id: str) -> dict | None:
    for ticket in load_tickets():
        if ticket["id"] == ticket_id:
            return ticket
    return None


def signature_for_ticket(ticket_id: str) -> dict | None:
    ticket = get_ticket(ticket_id)
    if ticket is None:
        return None
    return ROOT_CAUSE_SIGNATURES.get(ticket["gold_root_cause"])
