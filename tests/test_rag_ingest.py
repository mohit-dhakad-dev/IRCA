from pathlib import Path

import pytest

from rag.ingest import CHUNK_SECTIONS, build_index, load_chunks, parse_runbook

EXPECTED_DOC_IDS = {
    "RB-AUTH-001",
    "RB-DB-001",
    "RB-DEPLOY-001",
    "RB-DISK-001",
    "RB-MEMORY-001",
    "RB-NETWORK-001",
}


def test_load_chunks_pure_parse():
    chunks = load_chunks()

    expected_total = len(EXPECTED_DOC_IDS) * len(CHUNK_SECTIONS)
    assert expected_total == 30
    assert len(chunks) == expected_total
    assert len(chunks) == 30

    by_doc: dict[str, list] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    assert set(by_doc.keys()) == EXPECTED_DOC_IDS
    for doc_id, doc_chunks in by_doc.items():
        assert len(doc_chunks) == 5
        sections = {c.section for c in doc_chunks}
        assert sections == set(CHUNK_SECTIONS)
        assert "Category" not in sections

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))

    for chunk in chunks:
        assert chunk.doc_id
        assert chunk.section
        assert chunk.title
        assert chunk.category
        assert chunk.text
        assert chunk.body
        assert chunk.text.startswith(f"{chunk.title} — {chunk.section}")

    auth_chunks = {c.section: c for c in by_doc["RB-AUTH-001"]}
    assert auth_chunks["Symptoms"].category == "auth"
    assert auth_chunks["Root Cause"].body == "auth_signing_key_mismatch"


def test_build_index_and_idempotency(tmp_path):
    chroma_path = tmp_path / "chroma_db"

    collection = build_index(chroma_path=chroma_path)
    assert collection.count() == 30

    collection2 = build_index(chroma_path=chroma_path)
    assert collection2.count() == 30

    expected_ids = {c.id for c in load_chunks()}

    result = collection2.get()
    assert len(result["ids"]) == 30
    assert set(result["ids"]) == expected_ids

    by_id = {
        id_: (doc, meta)
        for id_, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    }
    for id_, (doc, meta) in by_id.items():
        for key in ("doc_id", "section", "title", "category"):
            assert meta.get(key)

    doc, meta = by_id["RB-AUTH-001::Root Cause"]
    assert meta["category"] == "auth"
    assert "auth_signing_key_mismatch" in doc


@pytest.mark.parametrize(
    "case,content",
    [
        (
            "missing H1",
            (
                "## Category\n"
                "broken\n"
                "## Symptoms\n"
                "- something\n"
                "## Diagnosis Steps\n"
                "1. look\n"
                "## Root Cause\n"
                "broken_thing\n"
                "## Fix\n"
                "- fix it\n"
                "## Constraints\n"
                "- none\n"
            ),
        ),
        (
            "missing Category",
            (
                "# Broken Runbook\n"
                "## Symptoms\n"
                "- something\n"
                "## Diagnosis Steps\n"
                "1. look\n"
                "## Root Cause\n"
                "broken_thing\n"
                "## Fix\n"
                "- fix it\n"
                "## Constraints\n"
                "- none\n"
            ),
        ),
        (
            "missing Fix",
            (
                "# Broken Runbook\n"
                "## Category\n"
                "broken\n"
                "## Symptoms\n"
                "- something\n"
                "## Diagnosis Steps\n"
                "1. look\n"
                "## Root Cause\n"
                "broken_thing\n"
                "## Constraints\n"
                "- none\n"
                # missing '## Fix'
            ),
        ),
        (
            "duplicate Fix",
            (
                "# Broken Runbook\n"
                "## Category\n"
                "broken\n"
                "## Symptoms\n"
                "- something\n"
                "## Diagnosis Steps\n"
                "1. look\n"
                "## Root Cause\n"
                "broken_thing\n"
                "## Fix\n"
                "- fix it\n"
                "## Fix\n"
                "- fix it again\n"
                "## Constraints\n"
                "- none\n"
            ),
        ),
    ],
)
def test_parse_runbook_missing_section_raises(tmp_path, case, content):
    bad_runbook = tmp_path / "RB-BAD-001.md"
    bad_runbook.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        parse_runbook(bad_runbook)


def _fenced_runbook_content(fence: str) -> str:
    return (
        "# Real Title\n"
        "## Category\n"
        "fenced\n"
        "## Symptoms\n"
        "- something\n"
        "## Diagnosis Steps\n"
        "1. look\n"
        f"{fence}\n"
        "## Fix\n"
        "# Not A Title\n"
        f"{fence}\n"
        "## Root Cause\n"
        "fenced_thing\n"
        "## Fix\n"
        "- real fix\n"
        "## Constraints\n"
        "- none\n"
    )


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_parse_runbook_fence_aware(tmp_path, fence):
    runbook = tmp_path / "RB-FENCED-001.md"
    runbook.write_text(_fenced_runbook_content(fence), encoding="utf-8")

    chunks = parse_runbook(runbook)

    assert len(chunks) == 5
    assert {c.section for c in chunks} == set(CHUNK_SECTIONS)

    by_section = {c.section: c for c in chunks}
    assert all(c.title == "Real Title" for c in chunks)

    diagnosis = by_section["Diagnosis Steps"]
    assert "## Fix" in diagnosis.body
    assert "# Not A Title" in diagnosis.body

    fix = by_section["Fix"]
    assert fix.body == "- real fix"


def test_parse_runbook_subheader_stays_in_body(tmp_path):
    content = (
        "# Real Title\n"
        "## Category\n"
        "sub\n"
        "## Symptoms\n"
        "- something\n"
        "### Extra Detail\n"
        "more info under symptoms\n"
        "## Diagnosis Steps\n"
        "1. look\n"
        "## Root Cause\n"
        "sub_thing\n"
        "## Fix\n"
        "- fix it\n"
        "## Constraints\n"
        "- none\n"
    )
    runbook = tmp_path / "RB-SUB-001.md"
    runbook.write_text(content, encoding="utf-8")

    chunks = parse_runbook(runbook)

    assert {c.section for c in chunks} == set(CHUNK_SECTIONS)
    by_section = {c.section: c for c in chunks}
    assert "### Extra Detail" in by_section["Symptoms"].body
    assert "more info under symptoms" in by_section["Symptoms"].body
