from datetime import timezone

import pytest

from agent.approval import (
    PendingAction,
    _extract_bounds_from_bullet_with_param,
    clear_store,
    create_pending_action,
    get_pending_action,
    list_pending_actions,
    verify_against_constraints,
)
from rag.ingest import RUNBOOKS_DIR, parse_runbook


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
    path.write_text(SYNTHETIC_NO_NUMBERS, encoding="utf-8")

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


# --- must-not-regress: genuine violations still reject ---------------------


def test_used_memory_rss_over_bound_rejected():
    result = verify_against_constraints("Set used_memory_rss to 95%", "RB-MEMORY-001")
    assert result["passed"] is False
    assert "95" in result["reason"] and "75" in result["reason"]


def test_timeout_seconds_over_bound_rejected():
    result = verify_against_constraints(
        "Set readiness timeoutSeconds to 9 seconds", "RB-DEPLOY-001"
    )
    assert result["passed"] is False
    assert "9" in result["reason"] and "5" in result["reason"]


def test_failure_threshold_over_bound_rejected():
    result = verify_against_constraints("Set failureThreshold to 5", "RB-DEPLOY-001")
    assert result["passed"] is False
    assert "5" in result["reason"] and "3" in result["reason"]


def test_maxmemory_over_bound_rejected():
    result = verify_against_constraints("Set maxmemory to 90%", "RB-MEMORY-001")
    assert result["passed"] is False
    assert "90" in result["reason"] and "80" in result["reason"]


# --- real compliant fixes that were previously false-rejected ---------------
#
# These five are taken verbatim (aside from whitespace normalisation) from
# sweep output that was falsely rejecting fully compliant fixes before the
# parameter-identity matching fix.


def test_real_fix_maxmemory_and_used_memory_rss_both_compliant():
    result = verify_against_constraints(
        "Reduce the Redis `maxmemory` setting to no more than 80 % of the "
        "node's total RAM and reconfigure to `allkeys-lru`. Verify that "
        "`used_memory_rss` stays below 75 % of the node's RAM",
        "RB-MEMORY-001",
    )
    assert result["passed"] is True


def test_real_fix_maxmemory_retry_with_explanatory_percentage():
    # The 80 in "(well under the 80% limit)" is explanatory text, not a
    # second proposed value -- but even if it is treated as one, its
    # nearest preceding identifier is `maxmemory`, whose own bound max is
    # 80%, so 80 <= 80 still passes.
    result = verify_against_constraints(
        "Lower the Redis `maxmemory` setting to 70 % of the node's total "
        "RAM (well under the 80 % limit)",
        "RB-MEMORY-001",
    )
    assert result["passed"] is True


def test_real_fix_deploy_readiness_probe_t004():
    result = verify_against_constraints(
        "readiness probe initialDelaySeconds of 20 seconds (no lower than "
        "15 seconds), timeoutSeconds to 5 seconds (within the 1-5 second "
        "range), failureThreshold to 3",
        "RB-DEPLOY-001",
    )
    assert result["passed"] is True


def test_real_fix_deploy_readiness_probe_t029():
    result = verify_against_constraints(
        "initialDelaySeconds of 20 seconds, a timeoutSeconds of 4 seconds, "
        "and a failureThreshold of 3",
        "RB-DEPLOY-001",
    )
    assert result["passed"] is True


# --- permanent full-corpus guard --------------------------------------------
#
# This test asserts the parsed result for EVERY Constraints bullet in ALL SIX
# runbooks in data/runbooks/ -- not a sample. It is intentionally brittle: if
# a runbook's Constraints section gains, loses, or rewords a bullet, this
# test WILL fail, and that failure requires a deliberate update to the
# EXPECTED_BOUNDS / NO_BOUND_SUBSTRINGS tables below (after re-checking the
# new/changed bullet's English meaning with
# `venv/bin/python -m eval.verify_constraint_parsing`) -- it must never be
# "fixed" by loosening the assertions. Each row encodes what the parser
# SHOULD produce given the bullet's actual meaning, not a transcription of
# whatever the parser currently happens to output.
#
# Two rows were judged rather than transcribed:
#   - RB-DISK-001's "cap ... at 100 MB and retain no more than 5 files":
#     neither bound has a real identifier in the bullet (no backtick/
#     camelCase/snake_case token), so both correctly fall back to the same
#     descriptive subject -- this is harmless because the two bounds have
#     different units (MB vs file) and can never be cross-compared.
#   - RB-NETWORK-001's "If a service has a `max_connections` limit of 500,
#     avoid sustained operation above 400": the 400 bound is about sustained
#     *operation*, not the `max_connections` config value itself, and the
#     clause containing 400 never refers back to `max_connections` with a
#     pronoun -- so `max_connections` is correctly NOT pulled in as this
#     bound's parameter, and it falls back to a descriptive subject.

EXPECTED_BOUNDS = [
    # (doc_id, bullet_substring, direction, value, unit, parameter)
    ("RB-AUTH-001", "Maintain at least a 24-hour key overlap", "min", 24.0, "hour", None),
    ("RB-DB-001", "keep a safety margin of at least 20%", "min", 20.0, "%", None),
    ("RB-DEPLOY-001", "Keep readiness `timeoutSeconds` between 1 and 5 seconds", "min", 1.0, "seconds", "timeoutseconds"),
    ("RB-DEPLOY-001", "Keep readiness `timeoutSeconds` between 1 and 5 seconds", "max", 5.0, "seconds", "timeoutseconds"),
    ("RB-DEPLOY-001", "`failureThreshold` should not exceed 3", "max", 3.0, "", "failurethreshold"),
    ("RB-DEPLOY-001", "`initialDelaySeconds` should be no lower than 15 seconds", "min", 15.0, "seconds", "initialdelayseconds"),
    ("RB-DISK-001", "Keep each production filesystem below 80% utilization", "max", 80.0, "%", None),
    ("RB-DISK-001", "cap any single log file at 100 MB", "max", 100.0, "MB", None),
    ("RB-DISK-001", "retain no more than 5 files", "max", 5.0, "file", None),
    ("RB-MEMORY-001", "Keep Redis `used_memory_rss` below 75% of the node's total RAM", "max", 75.0, "%", "usedmemoryrss"),
    ("RB-MEMORY-001", "leave at least 25% headroom", "min", 25.0, "%", None),
    ("RB-MEMORY-001", "If `maxmemory` is set, do not allow it to exceed 80%", "max", 80.0, "%", "maxmemory"),
    ("RB-NETWORK-001", "Keep backend connection count below 70-80%", "max", 80.0, "%", None),
    ("RB-NETWORK-001", "avoid sustained operation above 400", "max", 400.0, "", None),
]

# Bullets that must produce NO bounds at all (8 of the corpus's 18 bullets).
NO_BOUND_SUBSTRINGS = [
    ("RB-AUTH-001", "Keep the target keyset consistent"),
    ("RB-AUTH-001", "Any public-key distribution must remain valid"),
    ("RB-DB-001", "If the app is configured with a pool of 50"),
    ("RB-DB-001", "Any pool increase must be validated"),
    ("RB-DEPLOY-001", "Do not increase `maxUnavailable`"),
    ("RB-DISK-001", "Any cleanup or log redirection"),
    ("RB-MEMORY-001", "Any cache expansion must be validated"),
    ("RB-NETWORK-001", "Any tuning to backlog values"),
]


def _all_corpus_bullets() -> list[tuple[str, str]]:
    """(doc_id, bullet_source_text) for every '- ...' line in every
    runbook's Constraints section, read straight from the real corpus via
    rag.ingest.parse_runbook -- nothing hand-picked or filtered."""
    out: list[tuple[str, str]] = []
    for path in sorted(RUNBOOKS_DIR.glob("*.md")):
        doc_id = path.stem
        chunks = parse_runbook(path)
        constraints_chunk = next((c for c in chunks if c.section == "Constraints"), None)
        if constraints_chunk is None:
            continue
        for line in constraints_chunk.body.splitlines():
            line = line.strip()
            if line.startswith("-"):
                out.append((doc_id, line))
    return out


def test_full_corpus_constraint_parsing_matches_expected_table():
    """See module-level docstring above EXPECTED_BOUNDS: this drives its
    ground truth from the real runbook corpus (via rag.ingest.parse_runbook)
    and must fail loudly -- not be loosened -- if a runbook's Constraints
    bullets change without a matching deliberate update here."""
    all_bullets = _all_corpus_bullets()

    assert len(all_bullets) == 18, (
        f"Expected 18 Constraints bullets across the corpus, found "
        f"{len(all_bullets)}. The runbook corpus has changed -- update "
        f"EXPECTED_BOUNDS / NO_BOUND_SUBSTRINGS deliberately."
    )

    # Every expected bullet substring must uniquely identify exactly one
    # real bullet, and every real bullet must be covered by exactly one
    # expected-bounds substring or one no-bounds substring.
    matched_bullets: set[tuple[str, str]] = set()

    bounds_by_bullet: dict[tuple[str, str], list[tuple]] = {}
    for doc_id, substring, direction, value, unit, parameter in EXPECTED_BOUNDS:
        candidates = [
            (d, b) for d, b in all_bullets if d == doc_id and substring in b
        ]
        assert len(candidates) == 1, (
            f"Expected exactly one bullet in {doc_id} containing "
            f"{substring!r}, found {len(candidates)}."
        )
        key = candidates[0]
        matched_bullets.add(key)
        bounds_by_bullet.setdefault(key, []).append(
            (direction, value, unit, parameter)
        )

    for doc_id, substring in NO_BOUND_SUBSTRINGS:
        candidates = [
            (d, b) for d, b in all_bullets if d == doc_id and substring in b
        ]
        assert len(candidates) == 1, (
            f"Expected exactly one bullet in {doc_id} containing "
            f"{substring!r}, found {len(candidates)}."
        )
        key = candidates[0]
        assert key not in matched_bullets, (
            f"Bullet {key!r} appears in both EXPECTED_BOUNDS and "
            f"NO_BOUND_SUBSTRINGS."
        )
        matched_bullets.add(key)

    assert matched_bullets == set(all_bullets), (
        "Some corpus bullets are not covered by EXPECTED_BOUNDS or "
        f"NO_BOUND_SUBSTRINGS: {set(all_bullets) - matched_bullets}"
    )

    for (doc_id, bullet_text), expected_bounds in bounds_by_bullet.items():
        actual = _extract_bounds_from_bullet_with_param(bullet_text)
        actual_tuples = [
            (op, value, unit, param) for op, value, unit, _subject, _text, param in actual
        ]
        assert sorted(actual_tuples) == sorted(expected_bounds), (
            f"{doc_id} bullet {bullet_text!r}: expected {expected_bounds}, "
            f"got {actual_tuples}"
        )

    for doc_id, substring in NO_BOUND_SUBSTRINGS:
        key = next((d, b) for d, b in all_bullets if d == doc_id and substring in b)
        actual = _extract_bounds_from_bullet_with_param(key[1])
        assert actual == [], f"{doc_id} bullet {key[1]!r}: expected NO bounds, got {actual}"
