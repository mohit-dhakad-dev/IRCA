"""Tests for rag/retrieve.py.

Builds a throwaway chroma index in a session-scoped tmp dir once (never
touching the repo's real ./chroma_db), and exercises search_runbooks against
it: known-symptom retrieval, the no_confident_match gate, missing-index
handling, and the empty-query guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag import retrieve
from rag.ingest import COLLECTION_NAME, build_index, parse_runbook

CONTRACT_KEYS = {"doc_id", "section", "text", "score"}


@pytest.fixture(scope="session")
def index_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("chroma_test_index")
    build_index(chroma_path=path, collection_name=COLLECTION_NAME)
    yield path
    retrieve._reset_cache()


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():
    retrieve._reset_cache()
    yield
    retrieve._reset_cache()


def test_search_runbooks_db_pool_query(index_path: Path):
    result = retrieve.search_runbooks(
        "connection pool timeout errors, active connections pinned at max",
        chroma_path=index_path,
        collection_name=COLLECTION_NAME,
    )
    assert result["status"] == "ok"
    chunks = result["data"]["chunks"]
    assert len(chunks) <= 3
    for chunk in chunks:
        assert CONTRACT_KEYS.issubset(chunk.keys())
    assert "RB-DB-001" in {c["doc_id"] for c in chunks}


def test_search_runbooks_cache_memory_query(index_path: Path):
    result = retrieve.search_runbooks(
        "redis cache memory usage climbing toward saturation and evictions rising",
        chroma_path=index_path,
        collection_name=COLLECTION_NAME,
    )
    assert result["status"] == "ok"
    chunks = result["data"]["chunks"]
    assert len(chunks) <= 3
    for chunk in chunks:
        assert CONTRACT_KEYS.issubset(chunk.keys())
    assert "RB-MEMORY-001" in {c["doc_id"] for c in chunks}


def test_search_runbooks_auth_jwt_query(index_path: Path):
    result = retrieve.search_runbooks(
        "401 responses after JWT key rotation, some sessions failing others still authenticate",
        chroma_path=index_path,
        collection_name=COLLECTION_NAME,
    )
    assert result["status"] == "ok"
    chunks = result["data"]["chunks"]
    assert len(chunks) <= 3
    for chunk in chunks:
        assert CONTRACT_KEYS.issubset(chunk.keys())
    assert "RB-AUTH-001" in {c["doc_id"] for c in chunks}


def test_search_runbooks_nonsense_query_no_confident_match(index_path: Path):
    result = retrieve.search_runbooks(
        "purple bicycle tuesday marmalade recipe",
        chroma_path=index_path,
        collection_name=COLLECTION_NAME,
    )
    assert result["status"] == "no_confident_match"
    assert result["data"]["top_score"] < 0.5


def test_search_runbooks_missing_index_returns_error(tmp_path: Path):
    empty_dir = tmp_path / "no_index_here"
    result = retrieve.search_runbooks(
        "connection pool exhaustion",
        chroma_path=empty_dir,
        collection_name=COLLECTION_NAME,
    )
    assert result["status"] == "error"
    assert "rag.ingest" in result["summary"]


def test_search_runbooks_empty_query_returns_error(index_path: Path):
    result = retrieve.search_runbooks(
        "   ",
        chroma_path=index_path,
        collection_name=COLLECTION_NAME,
    )
    assert result["status"] == "error"


def test_stale_cached_collection_repairs_itself_on_rebuild(tmp_path: Path):
    """FIX 1 regression test: build -> search (populates cache) -> build again
    with NO cache reset in between -> search again must not raise a raw
    chromadb.errors.NotFoundError, and must still return a sane status. The
    test resets the module cache itself before/after so it doesn't leak state
    into the other tests in this file, but deliberately does NOT reset
    between the two build_index() calls -- that's the whole point."""
    retrieve._reset_cache()
    try:
        chroma_path = tmp_path / "chroma_db"

        build_index(chroma_path=chroma_path, collection_name=COLLECTION_NAME)
        first = retrieve.search_runbooks(
            "connection pool timeout errors, active connections pinned at max",
            chroma_path=chroma_path,
            collection_name=COLLECTION_NAME,
        )
        assert first["status"] == "ok"

        # Rebuild without resetting retrieve's cache: the collection now has
        # a new internal chroma UUID, but retrieve._CACHE still holds the old
        # (now-dead) Collection handle.
        build_index(chroma_path=chroma_path, collection_name=COLLECTION_NAME)

        second = retrieve.search_runbooks(
            "connection pool timeout errors, active connections pinned at max",
            chroma_path=chroma_path,
            collection_name=COLLECTION_NAME,
        )
        assert second["status"] == "ok"
    finally:
        retrieve._reset_cache()


def test_reopen_does_not_silently_change_embedding_behavior(tmp_path: Path):
    """Regression for the get_collection embedding-function default trap.

    chromadb 1.5.9's get_collection does NOT restore the embedding function
    used at create_collection() time -- it defaults to chroma's own
    DefaultEmbeddingFunction. Measured facts (see docs/decisions.md D6):
    omitting the argument happens to score identically today, because that
    default is ALSO all-MiniLM-L6-v2 via chroma's bundled onnx runtime; but
    a genuinely different model is accepted SILENTLY, with no conflict
    guard -- reopening this collection with paraphrase-MiniLM-L3-v2 returns
    ~0.18 and the wrong document where the right model returns ~0.69.

    So the hazard this pins is model DRIFT between ingest and query, which
    produces wrong numbers rather than an exception. That is why the test
    asserts on OBSERVED SCORES across a genuine reopen rather than on the
    class of the embedding-function object, which would pass even if chroma
    ignored it entirely."""
    retrieve._reset_cache()
    try:
        chroma_path = tmp_path / "chroma_db"
        build_index(chroma_path=chroma_path, collection_name=COLLECTION_NAME)

        queries = [
            ("connection pool timeout errors, active connections pinned at max", "RB-DB-001"),
            (
                "redis cache memory usage climbing toward saturation and evictions rising",
                "RB-MEMORY-001",
            ),
            (
                "401 responses after JWT key rotation, some sessions failing others still authenticate",
                "RB-AUTH-001",
            ),
        ]

        # First open (also populates retrieve's module cache).
        first_open = {}
        for query, expected_doc_id in queries:
            result = retrieve.search_runbooks(
                query, chroma_path=chroma_path, collection_name=COLLECTION_NAME
            )
            top = result["data"]["chunks"][0]
            assert top["doc_id"] == expected_doc_id
            first_open[query] = (top["doc_id"], top["score"])

        # Force a genuine reopen: drop the cached client/collection handle
        # and re-resolve from disk via a fresh _get_collection call.
        retrieve._reset_cache()

        reopened = {}
        for query, expected_doc_id in queries:
            collection = retrieve._get_collection(chroma_path, COLLECTION_NAME)
            results = collection.query(query_texts=[query], n_results=3)
            doc_id = results["metadatas"][0][0]["doc_id"]
            score = round(1.0 - results["distances"][0][0], 4)
            reopened[query] = (doc_id, score)

        # The reopened scores must match the first-open scores to a very
        # tight tolerance -- they should be effectively identical, since the
        # same model is embedding the same query text both times.
        for query, _ in queries:
            first_doc_id, first_score = first_open[query]
            reopened_doc_id, reopened_score = reopened[query]
            assert reopened_doc_id == first_doc_id
            assert abs(reopened_score - first_score) < 1e-6

        # Baseline: hardcoded known-good scores from an actual run of this
        # index/model combination. The 0.02 tolerance (looser than the 1e-6
        # above) absorbs minor torch/sentence-transformers version drift
        # while still being far tighter than the gap a genuinely different
        # embedding model would produce (which showed up as a >0.5 shift in
        # the negative control below).
        known_good_baseline = {
            queries[0][0]: ("RB-DB-001", 0.6922),
            queries[1][0]: ("RB-MEMORY-001", 0.8324),
            queries[2][0]: ("RB-AUTH-001", 0.7859),
        }
        for query, (baseline_doc_id, baseline_score) in known_good_baseline.items():
            doc_id, score = reopened[query]
            assert doc_id == baseline_doc_id
            assert abs(score - baseline_score) < 0.02

        # Negative control: prove this test is actually sensitive to a model
        # swap. Two approaches were tried:
        #
        # 1. chroma's own DefaultEmbeddingFunction, opened via get_collection
        #    like a real "forgot to pass the right EF" bug would. This
        #    turned out to ALSO be all-MiniLM-L6-v2 under the hood (just via
        #    chroma's bundled onnx runtime instead of sentence-transformers),
        #    so it produced near-identical scores and could not serve as a
        #    negative control -- same model, different wrapper.
        # 2. A genuinely different custom embedding function, opened via
        #    get_collection with a different .name(). This chromadb version
        #    (1.5.9) actively REJECTS that at get_collection() time with an
        #    "embedding function conflict" error against the persisted
        #    config -- which is reassuring (chroma guards against exactly
        #    this class of bug for named EFs) but means it can't be used as
        #    a negative control through get_collection either.
        #
        # So the negative control below feeds materially different
        # (hash-based, non-semantic) query embeddings directly into
        # collection.query(query_embeddings=...), bypassing the
        # embedding-function-selection step entirely. This isolates and
        # tests the thing this test actually needs to be sensitive to --
        # whether different embeddings for the same query text produce
        # materially different similarity scores against the SAME index --
        # without being blocked by chroma's (separate, already-working)
        # conflict guard.
        def _hash_embedding(text: str) -> list[float]:
            import hashlib

            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = [(b / 255.0) - 0.5 for b in digest] * 12  # pad to 384 dims
            return vector[:384]

        collection = retrieve._get_collection(chroma_path, COLLECTION_NAME)

        materially_different = False
        for query, _ in queries:
            baseline_doc_id, baseline_score = known_good_baseline[query]
            results = collection.query(query_embeddings=[_hash_embedding(query)], n_results=3)
            wrong_score = round(1.0 - results["distances"][0][0], 4)
            if abs(wrong_score - baseline_score) > 0.1:
                materially_different = True

        assert materially_different, (
            "expected hash-based (non-semantic) query embeddings to produce materially "
            "different scores than the real embedding model -- if this fails, the negative "
            "control isn't sensitive and the assertions above can't be trusted to catch a "
            "real embedding-function swap"
        )
    finally:
        retrieve._reset_cache()


def test_closing_fence_with_info_string_does_not_close_fence(tmp_path: Path):
    """FIX 4: a bare '```' closes a fence, but a line with an info string
    (e.g. '```python') appearing INSIDE an already-open fence must not close
    it -- CommonMark forbids info strings on closing fences. Confirms the
    inner '```python' line is preserved verbatim in the body and section
    boundaries still come out right."""
    runbook_text = (
        "# Fence Test Runbook\n\n"
        "## Category\n"
        "test\n\n"
        "## Symptoms\n"
        "Something happens.\n\n"
        "## Diagnosis Steps\n"
        "```\n"
        "some shell command\n"
        "```python\n"
        "print('this is body content, not a real fence open')\n"
        "```\n"
        "More diagnosis text after the fence closes.\n\n"
        "## Root Cause\n"
        "Root cause text.\n\n"
        "## Fix\n"
        "Fix text.\n\n"
        "## Constraints\n"
        "Constraints text.\n"
    )
    runbook_path = tmp_path / "RB-FENCE-001.md"
    runbook_path.write_text(runbook_text, encoding="utf-8")

    chunks = parse_runbook(runbook_path)
    by_section = {c.section: c for c in chunks}

    diagnosis_body = by_section["Diagnosis Steps"].body
    assert "```python" in diagnosis_body
    assert "print('this is body content, not a real fence open')" in diagnosis_body
    assert "More diagnosis text after the fence closes." in diagnosis_body

    # The fence-close logic must not have leaked into Root Cause: that
    # section's body must be exactly what follows, not swallowed as part of
    # an unterminated fence.
    assert by_section["Root Cause"].body == "Root cause text."
