"""RAG ingestion layer: parses runbooks in data/runbooks/ into per-section
chunks and builds a Chroma collection over them.

Chunking decision: `## Category` never becomes its own chunk. It is a single
snake_case or lower-case word (e.g. "auth"), and a near-content-free vector
like that can win a top-3 similarity slot for the wrong reason (matching on
the word "auth" rather than on symptoms/fix content). Instead its value is
hoisted into `category` metadata on every other chunk of the same document,
where it is far more useful as a retrieval filter (see Part B's
search_runbooks). The remaining 5 sections (Symptoms, Diagnosis Steps,
Root Cause, Fix, Constraints) each become exactly one chunk, so 6 runbooks
produce 6 * 5 = 30 chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb

from vectorstore import CHROMA_PATH, EMBEDDING_MODEL, rebuild_collection

RUNBOOKS_DIR = Path(__file__).resolve().parent.parent / "data" / "runbooks"
COLLECTION_NAME = "runbooks"

# The 5 sections that become chunks, in the fixed order they appear in every
# runbook. "Category" is deliberately excluded -- see module docstring.
CHUNK_SECTIONS = ["Symptoms", "Diagnosis Steps", "Root Cause", "Fix", "Constraints"]

_H1_LINE_RE = re.compile(r"^#\s+(.+?)\s*$")
_SECTION_LINE_RE = re.compile(r"^##\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
# CommonMark forbids an info string on a CLOSING fence: a line only closes an
# open fence if the fence marker is the entire line (surrounding whitespace
# allowed). The opener above deliberately still matches a trailing info
# string (e.g. "```python") since that is what OPENS a fence.
_FENCE_CLOSE_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    section: str
    title: str
    category: str
    text: str
    body: str


def _chunks_to_chroma(chunks: list[Chunk]) -> tuple[list[str], list[str], list[dict]]:
    """Map a list of Chunks to the (ids, documents, metadatas) triple chroma wants."""
    ids = [c.id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "doc_id": c.doc_id,
            "section": c.section,
            "title": c.title,
            "category": c.category,
        }
        for c in chunks
    ]
    return ids, documents, metadatas


def parse_runbook(path: Path) -> list[Chunk]:
    """Parse one runbook file into its 5 section chunks.

    Raises ValueError if the H1 title, the Category section, or any of the
    5 expected chunk sections is missing -- silent partial ingestion is
    worse than a loud failure here.
    """
    raw = path.read_text(encoding="utf-8")
    doc_id = path.stem

    title: str | None = None
    sections: dict[str, list[str]] = {}
    current: str | None = None

    fence_char: str | None = None
    fence_len = 0

    for line in raw.splitlines():
        if fence_char is not None:
            # Inside a fence: no line here is treated as H1 or section header,
            # it is verbatim body content of whatever section is open. Only a
            # bare fence marker (no info string) can close it -- a line like
            # "```python" appearing inside the fence must not close it.
            fence_match = _FENCE_CLOSE_RE.match(line)
            if fence_match:
                marker = fence_match.group(1)
                if marker[0] == fence_char and len(marker) >= fence_len:
                    fence_char = None
                    fence_len = 0
            if current is not None:
                sections[current].append(line)
            continue

        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            fence_char = marker[0]
            fence_len = len(marker)
            if current is not None:
                sections[current].append(line)
            continue

        h1_match = _H1_LINE_RE.match(line)
        if h1_match:
            if title is None:
                title = h1_match.group(1).strip()
                continue
            if current is not None:
                sections[current].append(line)
            continue

        section_match = _SECTION_LINE_RE.match(line)
        if section_match:
            name = section_match.group(1).strip()
            if name in sections:
                raise ValueError(f"{path}: duplicate '## {name}' section")
            sections[name] = []
            current = name
            continue

        if current is not None:
            sections[current].append(line)

    # An unterminated fence at EOF is treated as implicitly closed: its
    # content simply stays in the last open section rather than being lost.

    if title is None:
        raise ValueError(f"{path}: missing H1 title (expected a line starting with '# ')")

    sections = {name: "\n".join(lines).strip() for name, lines in sections.items()}

    if "Category" not in sections:
        raise ValueError(f"{path}: missing '## Category' section")
    category = sections["Category"].strip()

    chunks: list[Chunk] = []
    for section in CHUNK_SECTIONS:
        if section not in sections:
            raise ValueError(f"{path}: missing '## {section}' section")
        body = sections[section]
        text = f"{title} — {section}\n{body}"
        chunks.append(
            Chunk(
                id=f"{doc_id}::{section}",
                doc_id=doc_id,
                section=section,
                title=title,
                category=category,
                text=text,
                body=body,
            )
        )
    return chunks


def load_chunks(runbooks_dir: Path = RUNBOOKS_DIR) -> list[Chunk]:
    """Load and chunk every *.md file in runbooks_dir, sorted by filename
    for determinism. Raises if the directory is empty or missing."""
    runbooks_dir = Path(runbooks_dir)
    if not runbooks_dir.is_dir():
        raise ValueError(f"Runbooks directory not found: {runbooks_dir}")

    paths = sorted(runbooks_dir.glob("*.md"))
    if not paths:
        raise ValueError(f"No .md runbooks found in {runbooks_dir}")

    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(parse_runbook(path))
    return chunks


def build_index(
    runbooks_dir: Path = RUNBOOKS_DIR,
    chroma_path: Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL,
) -> chromadb.Collection:
    """Build (or idempotently rebuild) the Chroma collection over all runbook
    chunks. Deletes any existing collection of the same name first, so a
    rerun overwrites rather than appends.

    Cosine space is required: retrieval thresholds on score < 0.5, and
    chroma's default L2 distance has no such bounded interpretation.
    """
    chunks = load_chunks(runbooks_dir)
    ids, documents, metadatas = _chunks_to_chroma(chunks)
    return rebuild_collection(
        chroma_path=chroma_path,
        collection_name=collection_name,
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        model_name=model_name,
    )


if __name__ == "__main__":
    collection = build_index()

    chunks = load_chunks()
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1

    print("Cleared and rebuilt collection from scratch.")
    print("Chunks per doc:")
    for doc_id in sorted(counts):
        print(f"  {doc_id}: {counts[doc_id]}")
    print(f"Total chunks in collection: {collection.count()}")
