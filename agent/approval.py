"""Write-action approval gate (Part A: in-memory PendingAction store + a
constraint verifier).

`verify_against_constraints` is deliberately a shallow regex check over a
runbook's ``## Constraints`` bullets, NOT a general reasoning system. It
extracts numeric bounds (e.g. "no more than 5 files", "at least 24 hours",
"between 1 and 5 seconds") from each constraint bullet.

Each bound is associated with a proposed fix's numbers via PARAMETER-IDENTITY
matching, not free-form subject-token overlap:

- Where a bullet names a real identifier (a backtick-quoted token, or a
  camelCase/snake_case word -- e.g. `` `timeoutSeconds` ``, `max_connections`)
  anywhere in the bullet, that identifier (nearest the bound's own number) is
  the bound's parameter. A backtick identifier sitting in an earlier,
  comma-separated preamble clause is only pulled forward into a later
  clause's bound when that clause explicitly refers back to it with a
  pronoun ("it"/"this"/"that") and the preamble names exactly one such
  identifier -- this is what distinguishes "If `maxmemory` is set, do not
  allow IT to exceed 80%" (the bound genuinely is about maxmemory) from "If a
  service has a `max_connections` limit of 500, avoid sustained operation
  above 400" (the 400 bound is about sustained operation, not the
  max_connections config value -- there is no pronoun tying them together).
- Where a bullet has no identifiable parameter (e.g. "Keep each production
  filesystem below 80% utilization"), it falls back to the previous
  descriptive-subject-token-overlap heuristic.

On the fix side, each number is associated with its NEAREST PRECEDING
occurrence of a bound's parameter name (matched case/underscore/space/
backtick-insensitively), as long as no OTHER distinct identifier token
intervenes between that occurrence and the number -- this is what lets
"initial delay seconds" (spaced, lowercase) match `initialDelaySeconds`, and
what stops an unrelated, nearer identifier's value from being misattributed
to a bound several words away (e.g. two identifiers and two numbers in the
same sentence).

A bound is compared against a fix value only if units are also compatible (a
% bound never compares against a bare value and vice versa). If a value has
no identifiable parameter, or matches no bound's parameter, the verifier
ABSTAINS on that value rather than rejecting -- a false "no violation found"
is far cheaper than a false rejection of a fix that is actually compliant
with (or even mandated by) the runbook. This is still a shallow heuristic:
it can miss a real violation (if the fix never names the relevant
parameter) and, on unusual phrasing, it can still flag a fix that is
actually safe. It is a cheap deterministic tripwire, not a substitute for
the LLM verifier or human approval -- a human still approves every action.
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


# A bound is (op, value, unit, subject_tokens, bullet_text, parameter).
# `parameter` is a squashed identifier string (see `_squash`) when the bullet
# names a real parameter, or None when the bound falls back to descriptive
# subject-token overlap.
_Bound = tuple[str, float, str, frozenset, str, str | None]


# ---------------------------------------------------------------------------
# Parameter-identity matching
#
# An "identifier" is a backtick-quoted token, or a bare camelCase/snake_case
# word -- these are the tokens that unambiguously name a specific field
# (`timeoutSeconds`, `max_connections`) rather than merely describing one in
# prose.
# ---------------------------------------------------------------------------

_BACKTICK_ID_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
_CAMEL_ID_RE = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]*)+\b")
_SNAKE_ID_RE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")
_PRONOUN_RE = re.compile(r"\b(it|this|that)\b", re.IGNORECASE)


def _identifier_tokens(text: str) -> list[tuple[int, int, str]]:
    """Every identifier-looking token in `text` as (start, end, squashed),
    ordered by position. Backtick-quoted tokens take priority over a
    camelCase/snake_case match at the same span (e.g. a snake_case word
    inside backticks is only counted once)."""
    tokens: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for m in _BACKTICK_ID_RE.finditer(text):
        tokens.append((m.start(), m.end(), _squash(m.group(1))))
        occupied.append((m.start(), m.end()))

    def _overlaps(s: int, e: int) -> bool:
        return any(s < oe and e > os for os, oe in occupied)

    for regex in (_CAMEL_ID_RE, _SNAKE_ID_RE):
        for m in regex.finditer(text):
            if _overlaps(m.start(), m.end()):
                continue
            tokens.append((m.start(), m.end(), _squash(m.group(0))))
    tokens.sort(key=lambda t: t[0])
    return tokens


def _bound_parameter(clause: str, match_start: int) -> str | None:
    """Resolve the identifying parameter for a bound whose comparator/number
    match begins at `match_start` within `clause`, or None if the bullet has
    no identifiable parameter for this bound (caller should fall back to
    descriptive subject tokens). See module docstring for the pronoun rule
    governing when a preamble's identifier is pulled into a later clause."""
    segments = _comma_segments(clause)
    local_idx = len(segments) - 1
    for i, (start, end, _) in enumerate(segments):
        if start <= match_start < end:
            local_idx = i
            break
    local_start, _local_end, local_text = segments[local_idx]

    local_ids = _identifier_tokens(local_text)
    if local_ids:
        pos_in_local = match_start - local_start
        best = min(local_ids, key=lambda t: abs(t[0] - pos_in_local))
        return best[2]

    if _PRONOUN_RE.search(local_text):
        earlier_ids: list[tuple[int, int, str]] = []
        for i, (_, _, text) in enumerate(segments):
            if i >= local_idx:
                continue
            earlier_ids.extend(_identifier_tokens(text))
        if len(earlier_ids) == 1:
            return earlier_ids[0][2]

    return None


def _param_positions(text: str, param: str) -> list[tuple[int, int]]:
    """Every span in `text` whose squashed form equals `param` (letters/
    digits only, case/underscore/space-insensitive), as (start, end) in the
    ORIGINAL text's coordinates."""
    squashed_chars: list[str] = []
    mapping: list[int] = []
    for i, ch in enumerate(text):
        if ch.isalnum():
            squashed_chars.append(ch.lower())
            mapping.append(i)
    squashed_text = "".join(squashed_chars)

    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = squashed_text.find(param, start)
        if idx == -1:
            break
        s_orig = mapping[idx]
        e_orig = mapping[idx + len(param) - 1] + 1
        spans.append((s_orig, e_orig))
        start = idx + 1
    return spans


def _fix_value_matches_param(
    fix_text: str,
    fix_id_tokens: list[tuple[int, int, str]],
    number_start: int,
    param: str,
) -> bool:
    """True if `param` is the nearest preceding identifier of the number at
    `number_start` in `fix_text` -- i.e. `param` occurs before the number,
    and no OTHER distinct identifier token intervenes between that
    occurrence and the number."""
    occurrences = _param_positions(fix_text, param)
    preceding = [(s, e) for s, e in occurrences if e <= number_start]
    if not preceding:
        return False
    _s, e = max(preceding, key=lambda t: t[1])
    for ts, te, squashed in fix_id_tokens:
        if ts >= e and te <= number_start and squashed != param:
            return False
    return True


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


def _extract_bounds_from_clause(clause: str) -> list[tuple[str, float, str, frozenset, str | None]]:
    bounds: list[tuple[str, float, str, frozenset, str | None]] = []

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
        param = _bound_parameter(clause, m.start())
        bounds.append(("min", float(lo), unit, subject, param))
        bounds.append(("max", float(hi), unit, subject, param))
        masked = masked[: m.start()] + " " * (m.end() - m.start()) + masked[m.end() :]

    m = _HYPHEN_PCT_RE.search(clause)
    if m:
        _, hi = m.groups()
        subject = frozenset(_local_subject_tokens(clause, m.start()))
        param = _bound_parameter(clause, m.start())
        bounds.append(("max", float(hi), "%", subject, param))
        masked = masked[: m.start()] + " " * (m.end() - m.start()) + masked[m.end() :]

    for pat in _MAX_PHRASE_PATTERNS:
        pm = re.search(pat, masked, re.IGNORECASE)
        if pm:
            res = _first_number_and_unit(masked[pm.end():])
            if res:
                subject = frozenset(_local_subject_tokens(clause, pm.start()))
                param = _bound_parameter(clause, pm.start())
                bounds.append(("max", res[0], res[1], subject, param))

    for pat in _MIN_PHRASE_PATTERNS:
        pm = re.search(pat, masked, re.IGNORECASE)
        if pm:
            res = _first_number_and_unit(masked[pm.end():])
            if res:
                subject = frozenset(_local_subject_tokens(clause, pm.start()))
                param = _bound_parameter(clause, pm.start())
                bounds.append(("min", res[0], res[1], subject, param))

    # dedupe while preserving order
    seen: set[tuple[str, float, str, frozenset, str | None]] = set()
    deduped: list[tuple[str, float, str, frozenset, str | None]] = []
    for b in bounds:
        if b not in seen:
            seen.add(b)
            deduped.append(b)
    return deduped


def _extract_bounds_from_bullet_with_param(bullet: str) -> list[_Bound]:
    bounds: list[_Bound] = []
    for clause in _clauses(bullet):
        for op, value, unit, subject, param in _extract_bounds_from_clause(clause):
            bounds.append((op, value, unit, subject, bullet, param))
    return bounds


def _extract_fix_numbers(text: str) -> list[tuple[float, str, int]]:
    """Every number in `text`, paired with its unit ("" if bare -- Change 3)
    and its match-start position (used for nearest-preceding-identifier
    association against a bound's parameter)."""
    out: list[tuple[float, str, int]] = []
    for m in _NUM_RE.finditer(text):
        unit = _trailing_unit(text, m.end())
        out.append((float(m.group(0)), unit, m.start()))
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

        bullet_bounds: list[list[_Bound]] = [_extract_bounds_from_bullet_with_param(b) for b in bullets]
        all_bounds = [bound for bounds in bullet_bounds for bound in bounds]

        fix_text_lower = proposed_fix.lower()
        squashed_fix_text = _squash(proposed_fix)
        fix_numbers = _extract_fix_numbers(proposed_fix)
        fix_id_tokens = _identifier_tokens(proposed_fix)

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
            for op, bound_value, unit, subject_tokens, bullet_text, param in bounds:
                if param is not None:
                    candidates = [
                        (fix_value, fix_unit)
                        for fix_value, fix_unit, fix_start in fix_numbers
                        if _fix_value_matches_param(proposed_fix, fix_id_tokens, fix_start, param)
                    ]
                else:
                    if subject_tokens and not any(_squash(tok) in squashed_fix_text for tok in subject_tokens):
                        # No shared subject wording between this bound and
                        # the fix text -- the bound does not apply to this
                        # fix.
                        continue
                    candidates = [(fix_value, fix_unit) for fix_value, fix_unit, _fix_start in fix_numbers]

                for fix_value, fix_unit in candidates:
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
