"""Write-action approval gate (Part A: in-memory PendingAction store + a
constraint verifier).

`verify_against_constraints` is deliberately a shallow regex check over a
runbook's ``## Constraints`` bullets, NOT a general reasoning system. It
extracts numeric bounds (e.g. "no more than 5 files", "at least 24 hours",
"between 1 and 5 seconds") from each constraint bullet, together with a set
of distinctive "subject" tokens drawn from that bullet's wording (e.g.
"timeoutseconds", "initialdelayseconds"). A number pulled from the proposed
fix is checked against a bound only if the units match AND at least one of
the bound's subject tokens appears in the fix text -- unit equality alone is
not enough, since two different unrelated bullets can share a unit (two
"seconds" bounds for two different fields). This is still a shallow
heuristic: it can miss a real violation (if the fix's wording shares no
subject token with the relevant bullet) and, on unusual phrasing, it can
flag a fix that is actually safe. It is a cheap deterministic tripwire, not
a substitute for the LLM verifier or human approval -- a human still
approves every action.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rag.ingest import RUNBOOKS_DIR, parse_runbook

# ---------------------------------------------------------------------------
# PendingAction model
# ---------------------------------------------------------------------------


class PendingAction(BaseModel):
    action_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ticket_id: str
    proposed_root_cause: str
    proposed_fix: str
    citation_doc_id: str
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# In-memory store
#
# Deliberately in-memory for this MVP: a module-level dict. It does not
# survive a process restart -- a real deployment would back this with
# Postgres/SQLite (see docs/design.md Memory & State table).
# ---------------------------------------------------------------------------

_PENDING: dict[str, PendingAction] = {}


def create_pending_action(
    ticket_id: str,
    proposed_root_cause: str,
    proposed_fix: str,
    citation_doc_id: str,
) -> PendingAction:
    action = PendingAction(
        ticket_id=ticket_id,
        proposed_root_cause=proposed_root_cause,
        proposed_fix=proposed_fix,
        citation_doc_id=citation_doc_id,
    )
    if action.action_id in _PENDING:
        raise ValueError(f"action_id already exists: {action.action_id}")
    _PENDING[action.action_id] = action
    return action


def get_pending_action(action_id: str) -> PendingAction | None:
    return _PENDING.get(action_id)


def list_pending_actions() -> list[PendingAction]:
    return list(_PENDING.values())


def clear_store() -> None:
    """Test hook only: wipes the in-memory store. Not part of the runtime API."""
    _PENDING.clear()


# ---------------------------------------------------------------------------
# Constraint verifier
# ---------------------------------------------------------------------------

_NUM = r"(\d+(?:\.\d+)?)"
_UNIT = r"([A-Za-z%][A-Za-z]*)"
_NUMBER_UNIT_RE = re.compile(rf"{_NUM}\s*-?\s*{_UNIT}")

_BETWEEN_RE = re.compile(rf"\bbetween\s+{_NUM}\s+and\s+{_NUM}\s*-?\s*{_UNIT}", re.IGNORECASE)
_HYPHEN_PCT_RE = re.compile(rf"\b(?:below|under)\s+{_NUM}-{_NUM}\s*%", re.IGNORECASE)


_MAX_PHRASE_PATTERNS = [
    r"\bbelow\b",
    r"\bunder\b",
    r"\bshould not exceed\b",
    r"\bnot exceed\b",
    r"\bdo not allow it to exceed\b",
    r"\bretain no more than\b",
    r"\bno more than\b",
    r"\bcap\b.{0,40}?\bat\b",
    r"\bno higher than\b",
    r"\bavoid sustained operation above\b",
    r"\bdo not increase\b.{0,40}?\babove\b",
]

_MIN_PHRASE_PATTERNS = [
    r"\bshould be no lower than\b",
    r"\bno lower than\b",
    r"\bnot below\b",
    r"\bat least\b",
]

_ALL_PHRASE_PATTERNS = _MAX_PHRASE_PATTERNS + _MIN_PHRASE_PATTERNS

# Bare words that can follow a number without being a real unit noun (e.g.
# "above 400 without scaling" -- "without" is not a unit). A number followed
# by one of these (or by nothing at all) is treated as a bare number with
# unit key "" rather than being skipped -- see Change 3 in the module
# docstring / design discussion.
_UNIT_STOPWORDS = {
    "and", "or", "of", "the", "a", "an", "without", "per", "total",
    "this", "its", "that", "in", "on", "at", "to", "from", "for",
}

# Small stopword list for subject-token extraction (per spec): comparator
# words, articles, and generic connective words that carry no distinguishing
# "which field/thing is this bound about" signal.
_SUBJECT_STOPWORDS = {
    "the", "a", "an", "of", "to", "for", "and", "or", "not", "do", "any",
    "be", "is", "are", "keep", "should", "must", "than", "without", "with",
    "under", "above", "below", "at", "least", "more", "no", "exceed",
    "per", "each", "its", "this", "that", "if", "then", "on", "in", "by",
    "before", "after", "unless",
}


def _normalize_unit(raw: str) -> str:
    raw = raw.strip().rstrip(".,;:")
    if raw.startswith("%"):
        return "%"
    low = raw.lower()
    if low == "mb":
        return "MB"
    if low == "gb":
        return "GB"
    if low in ("seconds", "second"):
        return "seconds"
    if low in ("hour", "hours", "-hour"):
        return "hour"
    if low in ("files", "file"):
        return "file"
    if low.endswith("s") and len(low) > 1:
        low = low[:-1]
    return low


def _trailing_unit(text: str, end: int) -> str:
    """Look at the text immediately following a number (which ends at index
    `end`) and return its normalized unit, or "" if there is no valid unit
    word there (Change 3: bare numbers are allowed, not skipped)."""
    m = re.match(r"\s*-?\s*([A-Za-z%][A-Za-z]*)", text[end:])
    if m:
        raw = m.group(1)
        if raw.lower() not in _UNIT_STOPWORDS:
            return _normalize_unit(raw)
    return ""


def _first_number_and_unit(text: str) -> tuple[float, str] | None:
    """Return the (value, unit) of the first number in `text`, where unit is
    "" for a bare number with no valid trailing unit word. Returns None only
    if `text` contains no number at all."""
    m = _NUM_RE.search(text)
    if not m:
        return None
    value = float(m.group(0))
    unit = _trailing_unit(text, m.end())
    return value, unit


_NUM_RE = re.compile(_NUM)


def _clauses(bullet: str) -> list[str]:
    """Split a bullet into ';'-separated clauses. This scopes both bound
    extraction and subject-token derivation to the clause a number/phrase
    actually appears in, so an unrelated clause's vocabulary in the same
    bullet (e.g. "max pool size ...; keep a safety margin of at least 20%
    ...") does not leak into another clause's subject tokens or attract a
    number that belongs to a different clause."""
    parts = [p.strip() for p in bullet.split(";")]
    return [p for p in parts if p]


def _strip_phrase_spans(clause: str) -> str:
    """Remove every comparator-phrase match (between/hyphen-range/max/min
    phrases) from the clause, so subject-token extraction doesn't pick up
    the comparator wording itself as a distinguishing "subject" word."""
    text = clause
    for pat in (_BETWEEN_RE.pattern, _HYPHEN_PCT_RE.pattern, *_ALL_PHRASE_PATTERNS):
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return text


def _subject_tokens(clause: str) -> set[str]:
    unit_words = {m.group(2).lower() for m in _NUMBER_UNIT_RE.finditer(clause)}
    text = _strip_phrase_spans(clause)
    text = text.replace("`", " ")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).lower()
    tokens: set[str] = set()
    for word in text.split():
        if word.isdigit():
            continue
        if word in unit_words:
            continue
        if word in _SUBJECT_STOPWORDS:
            continue
        tokens.add(word)
    return tokens


# A bound is (op, value, unit, subject_tokens, bullet_text).
_Bound = tuple[str, float, str, frozenset, str]


def _comma_segments(clause: str) -> list[tuple[int, int, str]]:
    """Split `clause` into comma-delimited (start, end, text) segments."""
    segments: list[tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r",", clause):
        segments.append((start, m.start(), clause[start:m.start()]))
        start = m.end()
    segments.append((start, len(clause), clause[start:]))
    return segments


def _local_subject_tokens(clause: str, pos: int) -> set[str]:
    """Subject tokens for a bound match at `pos` within `clause`, scoped to
    avoid leaking an unrelated, backtick-named parameter from an earlier
    comma-separated "if ..." preamble into a later, textually distinct bound.

    Example: "If a service has a `max_connections` limit of 500, avoid
    sustained operation above 400 without scaling or tuning." -- the 400
    bound is about "sustained operation", not about the `max_connections`
    setting merely used to set the scene; if the preamble's backtick token
    leaked in as a subject word, a fix that sets `max_connections` to some
    unrelated value would be wrongly compared against the 400 bound. A
    preceding segment is dropped only when it names a backtick-quoted
    parameter of its own -- a plain-prose preamble (e.g. "If log rotation is
    configured, cap ... at 100 MB ...") carries no such distinct parameter
    identity and is still folded in, since it is legitimate context for the
    bound that follows it."""
    segments = _comma_segments(clause)
    local_idx = len(segments) - 1
    for i, (start, end, _) in enumerate(segments):
        if start <= pos < end:
            local_idx = i
            break

    kept = [
        text
        for i, (_, _, text) in enumerate(segments)
        if i >= local_idx or "`" not in text
    ]
    return _subject_tokens(",".join(kept))


def _extract_bounds_from_clause(clause: str) -> list[tuple[str, float, str, frozenset]]:
    bounds: list[tuple[str, float, str, frozenset]] = []

    # `masked` has any span already claimed by the between/hyphen-pct
    # patterns blanked out before the generic max/min phrase patterns run.
    # Without this, a clause like "below 70-80% of ..." matches BOTH the
    # hyphen-pct pattern (correctly yielding a 80% bound) AND the generic
    # "below" pattern (which, scanning past the range's own hyphen, fails to
    # find a valid unit word and silently produces a second, bogus bare-
    # number bound of "70 (no unit)"). That bare bound then wrongly gets
    # compared against absolute (unit-less) values in the proposed fix,
    # stripping the percentage semantics off a percentage-only constraint.
    masked = clause

    m = _BETWEEN_RE.search(clause)
    if m:
        lo, hi, raw_unit = m.groups()
        unit = _normalize_unit(raw_unit)
        subject = frozenset(_local_subject_tokens(clause, m.start()))
        bounds.append(("min", float(lo), unit, subject))
        bounds.append(("max", float(hi), unit, subject))
        masked = masked[: m.start()] + " " * (m.end() - m.start()) + masked[m.end() :]

    m = _HYPHEN_PCT_RE.search(clause)
    if m:
        _, hi = m.groups()
        subject = frozenset(_local_subject_tokens(clause, m.start()))
        bounds.append(("max", float(hi), "%", subject))
        masked = masked[: m.start()] + " " * (m.end() - m.start()) + masked[m.end() :]

    for pat in _MAX_PHRASE_PATTERNS:
        pm = re.search(pat, masked, re.IGNORECASE)
        if pm:
            res = _first_number_and_unit(masked[pm.end():])
            if res:
                subject = frozenset(_local_subject_tokens(clause, pm.start()))
                bounds.append(("max", res[0], res[1], subject))

    for pat in _MIN_PHRASE_PATTERNS:
        pm = re.search(pat, masked, re.IGNORECASE)
        if pm:
            res = _first_number_and_unit(masked[pm.end():])
            if res:
                subject = frozenset(_local_subject_tokens(clause, pm.start()))
                bounds.append(("min", res[0], res[1], subject))

    # dedupe while preserving order
    seen: set[tuple[str, float, str, frozenset]] = set()
    deduped: list[tuple[str, float, str, frozenset]] = []
    for b in bounds:
        if b not in seen:
            seen.add(b)
            deduped.append(b)
    return deduped


def _extract_bounds_from_bullet(bullet: str) -> list[_Bound]:
    bounds: list[_Bound] = []
    for clause in _clauses(bullet):
        for op, value, unit, subject in _extract_bounds_from_clause(clause):
            bounds.append((op, value, unit, subject, bullet))
    return bounds


def _extract_fix_numbers(text: str) -> list[tuple[float, str]]:
    """Every number in `text`, paired with its unit ("" if bare -- Change 3)."""
    out: list[tuple[float, str]] = []
    for m in _NUM_RE.finditer(text):
        unit = _trailing_unit(text, m.end())
        out.append((float(m.group(0)), unit))
    return out


def _squash(s: str) -> str:
    """Lowercase and strip everything but letters/digits, so parameter-name
    matching tolerates backticks, underscores, camelCase, and spacing
    differences (`initialDelaySeconds`, "initialDelaySeconds", and "initial
    delay seconds" all squash to the same string)."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _fmt_num(v: float) -> str:
    return str(int(v)) if v == int(v) else str(v)


def _fmt_unit(unit: str) -> str:
    return unit if unit else "(no unit)"


def verify_against_constraints(
    proposed_fix: str,
    citation_doc_id: str,
    runbooks_dir: Path | None = None,
) -> dict:
    """Shallow regex check of `proposed_fix` against the numeric bounds in
    `citation_doc_id`'s '## Constraints' section. Never raises -- any bad
    input (unknown doc id, unparseable runbook) yields passed=False with an
    explanatory reason instead of propagating an exception.
    """
    try:
        if runbooks_dir is None:
            runbooks_dir = RUNBOOKS_DIR

        path = Path(runbooks_dir) / f"{citation_doc_id}.md"
        if not path.is_file():
            return {
                "passed": False,
                "reason": f"Unknown citation_doc_id '{citation_doc_id}': no runbook file found at {path}.",
            }

        try:
            chunks = parse_runbook(path)
        except ValueError as e:
            return {
                "passed": False,
                "reason": f"Could not parse runbook '{citation_doc_id}': {e}",
            }

        constraints_chunk = next((c for c in chunks if c.section == "Constraints"), None)
        if constraints_chunk is None:
            return {
                "passed": False,
                "reason": f"Runbook '{citation_doc_id}' has no Constraints section.",
            }

        bullets = [
            line.strip()
            for line in constraints_chunk.body.splitlines()
            if line.strip().startswith("-")
        ]

        bullet_bounds: list[list[_Bound]] = [_extract_bounds_from_bullet(b) for b in bullets]
        all_bounds = [bound for bounds in bullet_bounds for bound in bounds]

        fix_text_lower = proposed_fix.lower()
        squashed_fix_text = _squash(proposed_fix)
        fix_numbers = _extract_fix_numbers(proposed_fix)

        if not all_bounds:
            return {
                "passed": True,
                "reason": f"No numeric constraint found in {citation_doc_id}'s Constraints section.",
            }
        if not fix_numbers:
            return {
                "passed": True,
                "reason": f"Proposed fix contains no numeric values to check against {citation_doc_id}'s constraints.",
            }

        for bounds in bullet_bounds:
            for op, bound_value, unit, subject_tokens, bullet_text in bounds:
                if subject_tokens and not any(_squash(tok) in squashed_fix_text for tok in subject_tokens):
                    # No shared subject wording between this bound and the
                    # fix text -- the bound does not apply to this fix.
                    continue
                for fix_value, fix_unit in fix_numbers:
                    if fix_unit != unit:
                        continue
                    if op == "max" and fix_value > bound_value:
                        return {
                            "passed": False,
                            "reason": (
                                f"Proposed fix value {_fmt_num(fix_value)} {_fmt_unit(fix_unit)} exceeds "
                                f"the max bound {_fmt_num(bound_value)} {_fmt_unit(unit)} in "
                                f"{citation_doc_id}: \"{bullet_text}\""
                            ),
                        }
                    if op == "min" and fix_value < bound_value:
                        return {
                            "passed": False,
                            "reason": (
                                f"Proposed fix value {_fmt_num(fix_value)} {_fmt_unit(fix_unit)} is below "
                                f"the min bound {_fmt_num(bound_value)} {_fmt_unit(unit)} in "
                                f"{citation_doc_id}: \"{bullet_text}\""
                            ),
                        }

        return {
            "passed": True,
            "reason": f"No numeric constraint in {citation_doc_id} is violated by the proposed fix.",
        }
    except Exception as e:  # pragma: no cover - safety net, never raise
        return {
            "passed": False,
            "reason": f"Constraint verification failed unexpectedly for '{citation_doc_id}': {e}",
        }
