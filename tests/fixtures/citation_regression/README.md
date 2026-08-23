# citation_regression fixtures

`T009.json`, `T024.json`, `T038.json` are trimmed captures of the `state`
object from the first full 63-ticket sweep (the raw per-ticket output lives
locally at `eval/results/raw/{id}.json`, which is gitignored and not checked
in). Each fixture keeps only what is needed to rebuild the critic digest for
that ticket: `ticket_id`, `description`, `hypothesis`, and `trajectory`
(with every `search_runbooks` observation's `chunks[].doc_id` intact), plus
an `expected_doc_id` field recording the runbook doc_id these tickets should
be cited against.

All three tickets falsely escalated in that sweep because a later round's
critic reply omitted `citations`, which the pre-fix code (an unconditional
`state.citations = assessment.citations` overwrite) treated as "no longer
supported" instead of "nothing new to add" -- wiping the correct citation
from an earlier round. Replaying these exact digests against the live critic
returns the correct doc_id 9/9 times, confirming the bug was in state
management, not retrieval or the critic's judgment. These fixtures back the
regression tests in `tests/test_orchestrator.py` (offline, digest-only) and
the `@pytest.mark.live` test that replays them against the real critic.
