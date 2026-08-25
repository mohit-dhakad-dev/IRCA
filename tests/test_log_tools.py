import pytest

from tools.fake_data import METRIC_PROFILES, get_ticket, signature_for_ticket
from tools.log_tools import query_logs, query_metrics

REAL_CAUSE_TICKETS = ["T001", "T002", "T003", "T004", "T005", "T009"]

INVALID_WINDOWS = ["", "   ", "30", "2hh", "abc", "0h", "-5m"]

KEYWORD_CASES = [
    ("T001", "checkout", "pool"),
    ("T002", "cache", "memory"),
    ("T009", "collector", "space"),
    ("T005", "auth", "signature"),
    ("T003", "ingress", "upstream"),
]


@pytest.mark.parametrize("ticket_id,service,keyword", KEYWORD_CASES)
def test_query_logs_summary_contains_signature_keyword(ticket_id, service, keyword):
    result = query_logs(ticket_id, service, "2h", "ERROR")
    assert result["status"] == "ok"
    assert keyword in result["summary"].lower()


def test_query_logs_unknown_ticket():
    result = query_logs("T999", "checkout", "2h", "ERROR")
    assert result["status"] == "error"


def test_query_logs_unrecognized_level():
    result = query_logs("T001", "checkout", "2h", "NONSENSE")
    assert result["status"] == "error"


def test_query_metrics_unknown_ticket():
    result = query_metrics("T999", "checkout", "cpu_percent", "2h")
    assert result["status"] == "error"


def test_query_metrics_unknown_metric():
    result = query_metrics("T001", "checkout", "not_a_real_metric", "2h")
    assert result["status"] == "unknown_metric"
    assert result["data"]["available_metrics"]


def test_query_metrics_wrong_metric_is_normal():
    result = query_metrics("T001", "checkout", "cpu_percent", "2h")
    assert result["status"] == "ok"
    assert "normal" in result["summary"].lower()
    lo, hi = METRIC_PROFILES["cpu_percent"]["normal_range"]
    for point in result["data"]["points"]:
        assert lo <= point["value"] <= hi


def test_query_metrics_right_metric_is_anomalous():
    result = query_metrics("T001", "checkout", "db_pool_active_connections", "2h")
    assert result["status"] == "ok"
    lo, hi = METRIC_PROFILES["db_pool_active_connections"]["anomaly_range"]
    for point in result["data"]["points"]:
        assert lo <= point["value"] <= hi
    assert "normal" not in result["summary"].lower()


def test_ambiguous_ticket_error_level_is_empty():
    result = query_logs("T015", "storefront", "2h", "ERROR")
    assert result["status"] == "empty"


def test_ambiguous_ticket_warn_level_has_no_dominant_pattern():
    result = query_logs("T015", "storefront", "2h", "WARN")
    assert result["status"] == "ok"
    assert result["data"]["total_matched"] <= 25
    total = result["data"]["total_matched"]
    for count in result["data"]["pattern_counts"].values():
        assert count < 0.55 * total
    root_cause_slugs = [
        "db_connection_pool_exhaustion",
        "memory_cache_overgrowth",
        "network_ingress_queue_exhaustion",
        "deploy_healthcheck_misconfiguration",
        "auth_signing_key_mismatch",
        "disk_log_rotation_gap",
    ]
    for slug in root_cause_slugs:
        assert slug not in result["summary"]


def test_query_logs_determinism():
    first = query_logs("T001", "checkout", "2h", "ERROR")
    second = query_logs("T001", "checkout", "2h", "ERROR")
    assert first == second


def test_query_metrics_determinism():
    first = query_metrics("T001", "checkout", "db_pool_active_connections", "2h")
    second = query_metrics("T001", "checkout", "db_pool_active_connections", "2h")
    assert first == second


# --- FIX 1: window validation must be total, never raise ------------------


@pytest.mark.parametrize("window", INVALID_WINDOWS)
def test_query_logs_invalid_window_returns_error(window):
    result = query_logs("T001", "checkout", window, "ERROR")
    assert result["status"] == "error"
    assert result["data"] == {}


@pytest.mark.parametrize("window", INVALID_WINDOWS)
def test_query_metrics_invalid_window_returns_error(window):
    result = query_metrics("T001", "checkout", "cpu_percent", window)
    assert result["status"] == "error"
    assert result["data"] == {}


def test_query_metrics_empty_window_does_not_raise():
    # Direct reproduction of the reported crash: this must return an error
    # dict, not raise IndexError.
    result = query_metrics("T001", "checkout", "db_pool_active_connections", "")
    assert result["status"] == "error"


# --- FIX 2: level filtering must be monotonic ------------------------------


def test_log_count_monotonic_across_levels():
    levels = ["DEBUG", "INFO", "WARN", "ERROR"]
    results = {lvl: query_logs("T001", "checkout", "2h", lvl) for lvl in levels}
    for lvl in levels:
        assert results[lvl]["status"] == "ok"
    totals = [results[lvl]["data"]["total_matched"] for lvl in levels]
    assert totals == sorted(totals, reverse=True)

    reference = results["ERROR"]["data"]["pattern_counts"]
    for lvl in levels:
        assert results[lvl]["data"]["pattern_counts"] == reference


# --- FIX 3: the anomaly-isolation invariant, for every real cause ---------


@pytest.mark.parametrize("ticket_id", REAL_CAUSE_TICKETS)
def test_metric_anomaly_isolated_to_signature_metric(ticket_id):
    # This is the core invariant the whole eval rests on: for a real-cause
    # ticket, exactly one metric reads anomalous (no "normal" in summary)
    # and every other known metric reads normal. Must still fail if the
    # `==` in query_metrics ever regresses to `in`.
    sig = signature_for_ticket(ticket_id)
    anomaly_metric = sig["metric_name"]
    for metric in METRIC_PROFILES:
        result = query_metrics(ticket_id, "svc", metric, "2h")
        assert result["status"] == "ok"
        points = result["data"]["points"]
        # Some anomaly shapes (step_up/step_down) deliberately start inside
        # normal_range and only transition partway through the window, so
        # only the LAST third is guaranteed to be inside anomaly_range for
        # every shape (sustained, climbing, step_up, step_down alike).
        last_third = points[-(len(points) // 3):]
        if metric == anomaly_metric:
            lo, hi = METRIC_PROFILES[metric]["anomaly_range"]
            assert "normal" not in result["summary"].lower()
        else:
            lo, hi = METRIC_PROFILES[metric]["normal_range"]
            assert "normal" in result["summary"].lower()
        for point in last_third:
            assert lo <= point["value"] <= hi


# --- FIX 4: ambiguity is derived from gold_root_cause, not expected_behavior


def test_ambiguous_tickets_detected_via_gold_root_cause():
    for ticket_id, service in [("T014", "auth"), ("T015", "storefront")]:
        ticket = get_ticket(ticket_id)
        assert ticket["gold_root_cause"] is None
        result = query_logs(ticket_id, service, "2h", "WARN")
        assert result["status"] == "ok"
        assert "no dominant error signature" in result["summary"].lower()


# --- FIX 5: metric argument is whitespace-normalized -----------------------


def test_query_metrics_metric_whitespace_stripped():
    unstripped = query_metrics("T001", "checkout", " cpu_percent", "2h")
    stripped = query_metrics("T001", "checkout", "cpu_percent", "2h")
    assert unstripped == stripped


# --- Round 3: direction, shape, integer values, capacity clause -----------


def test_low_bad_metric_anomaly_avoids_saturation_language():
    # T004 -> deploy_healthcheck_misconfiguration -> readiness_probe_success_rate
    result = query_metrics("T004", "api", "readiness_probe_success_rate", "2h")
    assert result["status"] == "ok"
    summary_lower = result["summary"].lower()
    for forbidden in ("pegged", "saturated", "capacity", "configured max"):
        assert forbidden not in summary_lower
    assert "normal" not in summary_lower


def test_low_bad_metric_healthy_still_reads_normal():
    # readiness_probe_success_rate isn't T001's cause, so it should read healthy.
    result = query_metrics("T001", "api", "readiness_probe_success_rate", "2h")
    assert result["status"] == "ok"
    assert "normal" in result["summary"].lower()


@pytest.mark.parametrize(
    "ticket_id,service,metric",
    [
        ("T004", "api", "readiness_probe_success_rate"),
        ("T010", "auth", "auth_token_verification_failure_rate"),
    ],
)
def test_step_shape_has_detectable_transition(ticket_id, service, metric):
    result = query_metrics(ticket_id, service, metric, "2h")
    assert result["status"] == "ok"
    points = result["data"]["points"]
    normal_lo, normal_hi = METRIC_PROFILES[metric]["normal_range"]
    anomaly_lo, anomaly_hi = METRIC_PROFILES[metric]["anomaly_range"]
    assert normal_lo <= points[0]["value"] <= normal_hi
    assert anomaly_lo <= points[-1]["value"] <= anomaly_hi


def test_climbing_shape_is_non_decreasing():
    # T009 -> disk_log_rotation_gap -> disk_usage_percent
    result = query_metrics("T009", "collector", "disk_usage_percent", "2h")
    assert result["status"] == "ok"
    values = [p["value"] for p in result["data"]["points"]]
    assert values == sorted(values)


@pytest.mark.parametrize("metric", ["db_pool_active_connections", "ingress_listen_queue_depth"])
def test_integer_metrics_render_as_ints(metric):
    ticket_id = "T001" if metric == "db_pool_active_connections" else "T003"
    result = query_metrics(ticket_id, "svc", metric, "2h")
    assert result["status"] == "ok"
    for point in result["data"]["points"]:
        assert float(point["value"]).is_integer()
    assert float(result["data"]["min"]).is_integer()
    assert float(result["data"]["max"]).is_integer()
    assert float(result["data"]["current"]).is_integer()
    range_segment = f"{result['data']['min']}-{result['data']['max']}"
    assert "." not in range_segment


def test_capacity_clause_only_for_genuine_limits():
    percent_result = query_metrics("T009", "collector", "disk_usage_percent", "2h")
    assert "configured max" not in percent_result["summary"].lower()

    pool_result = query_metrics("T001", "checkout", "db_pool_active_connections", "2h")
    assert "configured max" in pool_result["summary"].lower()

    queue_result = query_metrics("T003", "ingress", "ingress_listen_queue_depth", "2h")
    assert "configured max" in queue_result["summary"].lower()


# --- Session 10 Step 1: byte-identity regression, main-63 must be untouched
# by the adversarial injection wiring. Golden values were captured from the
# pre-change code (git show HEAD:tools/log_tools.py at the commit this
# change was made on top of) -- see tests/fixtures/query_tools_golden.json.


import json as _json  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_GOLDEN_PATH = _Path(__file__).resolve().parent / "fixtures" / "query_tools_golden.json"
_GOLDEN = _json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))

# Derived from the golden file rather than hardcoded, so adding a ticket to the
# fixture extends the regression automatically instead of drifting out of sync.
_GOLDEN_TICKETS = sorted(_GOLDEN["metrics_by_ticket"])


@pytest.mark.parametrize("ticket_id", _GOLDEN_TICKETS)
def test_query_logs_byte_identical_to_pre_injection_golden(ticket_id):
    result = query_logs(ticket_id, _GOLDEN["service"], _GOLDEN["window"], _GOLDEN["level"])
    assert _json.dumps(result, sort_keys=True) == _GOLDEN["query_logs"][ticket_id]


@pytest.mark.parametrize("ticket_id", _GOLDEN_TICKETS)
def test_query_metrics_byte_identical_to_pre_injection_golden(ticket_id):
    metric = _GOLDEN["metrics_by_ticket"][ticket_id]
    result = query_metrics(ticket_id, _GOLDEN["service"], metric, _GOLDEN["window"])
    assert _json.dumps(result, sort_keys=True) == _GOLDEN["query_metrics"][ticket_id]
    assert "annotations" not in result["data"]
