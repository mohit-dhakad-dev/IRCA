"""Shared Chroma plumbing used by rag/ and memory/. Both layers persist into
the same on-disk chroma path, in different collections -- this module is
collection-agnostic and knows nothing about runbooks or incidents.

See docs/decisions.md C3-C6, D1-D8 for the reasoning behind each piece here.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Shared default chroma path -- both rag/ and memory/ persist into this same
# on-disk store, in separate collections.
CHROMA_PATH = Path(__file__).resolve().parent / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level lazy cache of (chroma client, collection), keyed by
# (chroma_path, collection_name). Constructing SentenceTransformerEmbeddingFunction
# re-loads the sentence-transformers model, which costs real wall-clock time
# (seconds) -- callers hitting this repeatedly within a single run must not
# pay that cost on every call. Shared across rag/ and memory/: one cache,
# keyed by collection name, not one cache per layer.
_CACHE: dict[tuple[str, str], tuple[chromadb.ClientAPI, chromadb.Collection]] = {}


def _reset_cache() -> None:
    """Clear the module-level client/collection cache. Test-only hook so
    fixtures can point retrieval at a fresh tmp_path index between runs."""
    _CACHE.clear()


def _cache_key(chroma_path: Path, collection_name: str) -> tuple[str, str]:
    """Normalize the cache key so a relative and an absolute path to the same
    directory don't create two separate cache entries (and two separate
    PersistentClient handles) over the same underlying SQLite store."""
    return (str(Path(chroma_path).resolve()), collection_name)


def build_embedding_function(model_name: str = EMBEDDING_MODEL) -> SentenceTransformerEmbeddingFunction:
    """Build the SentenceTransformerEmbeddingFunction used both at index-build
    time and at query time. Naming the model explicitly on both sides matters:
    see get_collection()'s docstring for why."""
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_collection(
    chroma_path: Path,
    collection_name: str,
    model_name: str = EMBEDDING_MODEL,
) -> chromadb.Collection:
    """Return the (possibly cached) collection for chroma_path/collection_name.

    Raises chromadb.errors.NotFoundError if the collection does not exist
    (e.g. the corresponding *.ingest build_index() has never been run against
    this path) -- callers must catch this narrowly and turn it into a
    status="error" result, never let it propagate as an unhandled traceback.
    """
    key = _cache_key(chroma_path, collection_name)
    if key in _CACHE:
        return _CACHE[key][1]

    client = chromadb.PersistentClient(path=str(chroma_path))
    # IMPORTANT: get_collection does NOT restore the embedding function used at
    # create_collection() time -- it defaults to chroma's own
    # DefaultEmbeddingFunction. Measured on chromadb 1.5.9:
    #   * omitting the arg entirely still scores identically today, but only by
    #     coincidence: chroma's default is ALSO all-MiniLM-L6-v2 (via its
    #     bundled onnx runtime rather than sentence-transformers). Change
    #     EMBEDDING_MODEL and that coincidence ends.
    #   * passing a genuinely different model is accepted SILENTLY -- no
    #     conflict guard, no exception. Verified: reopening a collection built
    #     with all-MiniLM-L6-v2 but queried with paraphrase-MiniLM-L3-v2
    #     returned score 0.18 and the wrong doc where the correct model
    #     returns 0.69.
    # So the failure mode is real and unguarded, and naming the model on both
    # sides is what keeps ingest and query pinned to the same one.
    embedding_function = build_embedding_function(model_name)
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)

    _CACHE[key] = (client, collection)
    return collection


def rebuild_collection(
    chroma_path: Path,
    collection_name: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    model_name: str = EMBEDDING_MODEL,
) -> chromadb.Collection:
    """Build (or idempotently rebuild) a Chroma collection over the given
    (ids, documents, metadatas) triple. Deletes any existing collection of
    the same name first, so a rerun overwrites rather than appends.

    Cosine space is required: retrieval thresholds on a bounded similarity
    score, and chroma's default L2 distance has no such bounded
    interpretation.
    """
    client = chromadb.PersistentClient(path=str(chroma_path))

    try:
        client.delete_collection(name=collection_name)
    except NotFoundError:
        # Nothing to clear on a first-ever build. Caught narrowly: any other
        # chroma failure here is a real problem and must not be swallowed.
        pass

    embedding_function = build_embedding_function(model_name)
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    # upsert, not add: on chromadb 1.5.9, collection.add() silently no-ops on
    # a duplicate id (neither raises nor updates), so it provides no real
    # backstop. upsert() overwrites on a duplicate id, which is what makes
    # deterministic ids a real second line of defense behind the
    # clear-and-rebuild (delete_collection) above.
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    return collection


def index_not_found_error(chroma_path: Path, collection_name: str, ingest_module: str) -> dict:
    """The standard status="error" result for a missing/unbuilt index, shared
    by both call sites that can hit chromadb.errors.NotFoundError.

    ingest_module names the module the caller should tell the user to run,
    e.g. "rag.ingest" or "memory.ingest".
    """
    return {
        "status": "error",
        "data": {},
        "summary": (
            f"Runbook index not found at '{chroma_path}' (collection '{collection_name}'). "
            f"Run `python -m {ingest_module}` to build it first."
        ),
    }


def query_collection(
    chroma_path: Path,
    collection_name: str,
    query_text: str,
    top_k: int,
    model_name: str = EMBEDDING_MODEL,
) -> list[dict]:
    """Query collection_name for the top_k results most relevant to query_text,
    with evict-and-retry-once-on-NotFoundError repair of a stale cached
    collection handle, and cosine distance converted to a bounded similarity
    score.

    Raises chromadb.errors.NotFoundError if the collection cannot be resolved
    even after one retry (i.e. it genuinely does not exist) -- callers must
    catch this narrowly and turn it into a status="error" result.

    Returns a list of {"id", "text", "metadata", "score"} dicts (possibly
    empty), in the order chroma returned them.
    """
    try:
        collection = get_collection(chroma_path, collection_name, model_name)
        results = collection.query(query_texts=[query_text], n_results=top_k)
    except NotFoundError:
        # A rebuild (rebuild_collection) between the time this collection
        # handle was cached and now gives the collection a new internal chroma
        # UUID, so the cached handle's query() raises NotFoundError even
        # though get_collection() succeeded a moment ago. This can also
        # happen if a *different* process rebuilt the index under a
        # long-lived server, which no in-process reset hook could catch. The
        # cache is what keeps repeated queries fast, so we repair it (evict
        # the stale entry, re-resolve once) rather than dropping it -- and
        # retry the query exactly once against the fresh handle.
        _CACHE.pop(_cache_key(chroma_path, collection_name), None)
        collection = get_collection(chroma_path, collection_name, model_name)
        results = collection.query(query_texts=[query_text], n_results=top_k)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches = []
    for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        # Chroma's `distances` under "hnsw:space": "cosine" is 1 - cosine
        # similarity, not similarity itself. Thresholds are only meaningful
        # against a bounded [0, 1] *similarity* score, so we convert here:
        # score = 1.0 - distance.
        score = round(1.0 - distance, 4)
        matches.append({"id": doc_id, "text": text, "metadata": metadata, "score": score})

    return matches
