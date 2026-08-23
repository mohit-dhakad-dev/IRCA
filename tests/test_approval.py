from datetime import timezone

import pytest

from agent.approval import (
    PendingAction,
    clear_store,
    create_pending_action,
    get_pending_action,
    list_pending_actions,
    verify_against_constraints,
)


@pytest.fixture(autouse=True)
def _clear():
    clear_store()
    yield
    clear_store()


SYNTHETIC_NO_NUMBERS = """# Widget Cache Staleness

## Category
cache

## Symptoms
- Users report stale widget data appearing after updates.

## Diagnosis Steps
1. Check whether the cache invalidation hook fired for the affected widgets.

## Root Cause
widget_cache_staleness

## Fix
- Invalidate the cache entries for affected widgets and force a refresh.

## Constraints
- Any cache invalidation must be coordinated with the on-call owner before rollout.
- Do not bypass the standard change process when clearing production caches.
"""


# --- verify_against_constraints -------------------------------------------


def test_fix_violates_max_bound_fails():
    result = verify_against_constraints(
        "Set log rotation to 500 MB per file", "RB-DISK-001"
    )
    assert result["passed"] is False
    assert "MB" in result["reason"]


def test_fix_within_bounds_passes():
    result = verify_against_constraints(
        "Cap each log file at 50 MB and retain 3 files", "RB-DISK-001"
    )
    assert result["passed"] is True


def test_runbook_with_no_numeric_constraints_always_passes(tmp_path):
    path = tmp_path / "RB-CACHE-001.md"
    path.write_text(SYNTHETIC_NO_NUMBERS)

    result = verify_against_constraints(
        "Increase cache TTL to 999999 seconds and bump pool to 100000 connections",
        "RB-CACHE-001",
        runbooks_dir=tmp_path,
    )
    assert result["passed"] is True


def test_unknown_citation_doc_id_does_not_raise():
    result = verify_against_constraints("Do something", "RB-NOPE-999")
    assert result["passed"] is False
    assert "RB-NOPE-999" in result["reason"]


def test_unit_mismatch_not_cross_compared():
    # "500 connections" has no MB/file bound to violate in RB-DISK-001: no
    # unit match AND no subject-token overlap with the MB/file bullet.
    result = verify_against_constraints(
        "Increase the pool to 500 connections", "RB-DISK-001"
    )
    assert result["passed"] is True

    # But a fix that DOES share both the unit and the subject wording of a
    # real bound must still be caught -- this is what would fail if matching
    # were removed entirely (e.g. replaced with a stub that always passes).
    result = verify_against_constraints(
        "Adjust the log rotation retention policy to allow 500 MB per file.",
        "RB-DISK-001",
    )
    assert result["passed"] is False
    assert "MB" in result["reason"]


def test_min_bound_violation_fails():
    result = verify_against_constraints(
        "Set an overlap window of 2 hours", "RB-AUTH-001"
    )
    assert result["passed"] is False
    assert "hour" in result["reason"]


def test_deploy_between_bound_scoped_by_subject():
    # RB-DEPLOY-001's Constraints has two "seconds" bounds in different
    # bullets/clauses (timeoutSeconds between 1-5s; initialDelaySeconds no
    # lower than 15s). Subject-token overlap must keep them from
    # cross-contaminating each other.
    passes = verify_against_constraints(
        "Set readiness timeoutSeconds to 3 seconds", "RB-DEPLOY-001"
    )
    assert passes["passed"] is True

    fails = verify_against_constraints(
        "Set readiness timeoutSeconds to 30 seconds", "RB-DEPLOY-001"
    )
    assert fails["passed"] is False

    fails_min = verify_against_constraints(
        "Set initialDelaySeconds to 5 seconds", "RB-DEPLOY-001"
    )
    assert fails_min["passed"] is False

    passes_min = verify_against_constraints(
        "Set initialDelaySeconds to 30 seconds", "RB-DEPLOY-001"
    )
    assert passes_min["passed"] is True


def test_db_margin_bound_not_applied_without_subject_overlap():
    # "keep a safety margin of at least 20%" is about the margin, not about
    # the pool/connection count -- a fix that never mentions a margin must
    # not be flagged even though it shares the "%" unit.
    result = verify_against_constraints(
        "Increase the connection pool by 30%", "RB-DB-001"
    )
    assert result["passed"] is True


def test_deploy_bare_number_bound():
    # "failureThreshold should not exceed 3" has no unit noun after the
    # number -- it must still be extracted as a bare-unit ("") bound and
    # compared against a bare-unit fix number sharing its subject wording.
    result = verify_against_constraints(
        "Set failureThreshold to 6", "RB-DEPLOY-001"
    )
    assert result["passed"] is False


# --- regression: parameter/unit-aware bound matching -----------------------


def test_initial_delay_at_mandated_minimum_is_not_a_violation():
    # RB-DEPLOY-001 REQUIRES initialDelaySeconds >= 15. Proposing exactly the
    # mandated minimum must never be rejected -- this was the headline bug:
    # the old code cross-matched it against timeoutSeconds' unrelated max of 5.
    result = verify_against_constraints(
        "Set initialDelaySeconds to 15 seconds", "RB-DEPLOY-001"
    )
    assert result["passed"] is True


def test_initial_delay_above_minimum_is_not_a_violation():
    result = verify_against_constraints(
        "Set initialDelaySeconds to 30 seconds", "RB-DEPLOY-001"
    )
    assert result["passed"] is True


def test_timeout_seconds_within_its_own_bound_passes():
    result = verify_against_constraints(
        "Set readiness timeoutSeconds to 3 seconds", "RB-DEPLOY-001"
    )
    assert result["passed"] is True


def test_timeout_seconds_exceeding_its_own_bound_fails():
    result = verify_against_constraints(
        "Set readiness timeoutSeconds to 9 seconds", "RB-DEPLOY-001"
    )
    assert result["passed"] is False
    assert "seconds" in result["reason"]


def test_failure_threshold_exceeding_its_own_bound_fails():
    result = verify_against_constraints(
        "Set failureThreshold to 5", "RB-DEPLOY-001"
    )
    assert result["passed"] is False


def test_absolute_value_not_compared_against_percentage_bound():
    # RB-NETWORK-001's "below 70-80%" bound is a percentage; an absolute
    # max_connections value must never be cross-compared against it (unit
    # mismatch means the comparison is skipped, not rejected).
    result = verify_against_constraints(
        "Raise max_connections to 500 and add a second replica", "RB-NETWORK-001"
    )
    assert result["passed"] is True


def test_used_memory_rss_over_percentage_bound_still_rejected():
    # Must-not-regress: a genuine same-parameter, same-unit percentage
    # breach is still caught.
    result = verify_against_constraints(
        "Set used_memory_rss to 95%", "RB-MEMORY-001"
    )
    assert result["passed"] is False
    assert "%" in result["reason"]


def test_used_memory_rss_within_percentage_bound_passes():
    result = verify_against_constraints(
        "Set used_memory_rss to 60%", "RB-MEMORY-001"
    )
    assert result["passed"] is True


def test_unassociated_number_is_not_rejected():
    # A number that matches no bound's parameter at all must be treated as
    # unverified (pass), never as a violation -- abstaining beats a false
    # rejection of a fix the verifier can't confidently parse.
    result = verify_against_constraints(
        "Bump the widget frobnicator to 42", "RB-DEPLOY-001"
    )
    assert result["passed"] is True


def test_parameter_matching_is_tolerant_of_spacing_and_case():
    # `initialDelaySeconds`, "initialDelaySeconds", and "initial delay
    # seconds" must all associate with the same bound.
    for phrasing in (
        "Set initialDelaySeconds to 15 seconds",
        "Set initialdelayseconds to 15 seconds",
        "Set initial delay seconds to 15 seconds",
    ):
        result = verify_against_constraints(phrasing, "RB-DEPLOY-001")
        assert result["passed"] is True, phrasing


# --- PendingAction ----------------------------------------------------------


def test_pending_action_defaults():
    a = PendingAction(
        ticket_id="T1",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix="Rotate logs",
        citation_doc_id="RB-DISK-001",
    )
    b = PendingAction(
        ticket_id="T2",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix="Rotate logs",
        citation_doc_id="RB-DISK-001",
    )

    assert a.status == "pending"
    assert a.action_id
    assert a.action_id != b.action_id
    assert a.created_at.tzinfo is not None
    assert a.created_at.utcoffset() == timezone.utc.utcoffset(a.created_at)


# --- in-memory store ---------------------------------------------------------


def test_store_round_trip():
    assert list_pending_actions() == []
    assert get_pending_action("nonexistent") is None

    action = create_pending_action(
        ticket_id="T1",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix="Rotate logs",
        citation_doc_id="RB-DISK-001",
    )

    assert get_pending_action(action.action_id) is action
    assert list_pending_actions() == [action]


def test_store_duplicate_action_id_raises(monkeypatch):
    action = create_pending_action(
        ticket_id="T1",
        proposed_root_cause="disk_log_rotation_gap",
        proposed_fix="Rotate logs",
        citation_doc_id="RB-DISK-001",
    )

    monkeypatch.setattr(
        "agent.approval.uuid.uuid4", lambda: type("U", (), {"hex": action.action_id})()
    )

    with pytest.raises(ValueError):
        create_pending_action(
            ticket_id="T2",
            proposed_root_cause="disk_log_rotation_gap",
            proposed_fix="Rotate logs",
            citation_doc_id="RB-DISK-001",
        )
