"""
rootai/memory/store.py

ChromaDB-backed memory of past investigations. Each stored record is one
completed investigation: its question, structured KPIQuestion, final
ExecutiveBrief tl_dr, and the top ranked causes. New investigations
retrieve semantically similar past ones to give the Hypothesis Former
prior context.

Design:
- Local embeddings via ChromaDB's default embedding function (all-MiniLM-L6-v2).
  No Groq/OpenAI calls, no API cost, runs offline. First run downloads
  the model (~90MB) to a HuggingFace cache.
- Persistent path from config.chroma_path (default: data/chroma).
- Collection name: "rootai_investigations".
- Document stored: a compact string combining question + tl_dr + top cause.
  Metadata stored: kpi_name, direction, confidence of top cause, timestamp,
  investigation_id. Enough to filter and reconstruct.
- One record per investigation. If the same investigation_id is stored
  twice (shouldn't happen but possible during dev), Chroma overwrites.
"""
from __future__ import annotations

import time
from typing import Optional

import chromadb
from chromadb.config import Settings

from rootai.config import config
from rootai.state import ExecutiveBrief, InvestigationState, KPIQuestion, PriorInvestigation


COLLECTION_NAME = "rootai_investigations"


def _get_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client. Chroma handles concurrent access."""
    return chromadb.PersistentClient(
        path=str(config.chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection():
    """Return (create if missing) the investigations collection."""
    client = _get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _investigation_to_document(question: str, brief: ExecutiveBrief, sq: Optional[KPIQuestion]) -> str:
    """
    Compose the searchable text blob for one investigation.

    The blob combines the question, the tl_dr, and the top ranked cause
    statement. This is what ChromaDB embeds and searches against.
    """
    parts = [f"Question: {question}"]
    if sq:
        parts.append(f"KPI: {sq.kpi_name} ({sq.direction})")
    parts.append(f"TL;DR: {brief.tl_dr}")
    if brief.ranked_causes:
        top = brief.ranked_causes[0]
        parts.append(f"Top cause: {top.cause} (confidence {top.confidence:.2f})")
    return "\n".join(parts)


def _investigation_to_metadata(
    inv_id: str,
    question: str,
    brief: ExecutiveBrief,
    sq: Optional[KPIQuestion],
) -> dict:
    """
    Return ChromaDB metadata. Chroma metadata values must be str, int, float, or bool.
    Lists and dicts are not allowed, so we flatten.
    """
    md = {
        "investigation_id": inv_id,
        "question": question[:500],  # cap length
        "timestamp": time.time(),
        "n_causes": len(brief.ranked_causes),
    }
    if sq:
        md["kpi_name"] = sq.kpi_name
        md["direction"] = sq.direction
        if sq.magnitude_pct is not None:
            md["magnitude_pct"] = float(sq.magnitude_pct)
    if brief.ranked_causes:
        md["top_cause_confidence"] = float(brief.ranked_causes[0].confidence)
        md["top_cause"] = brief.ranked_causes[0].cause[:300]
    return md


def store_investigation(state: InvestigationState) -> bool:
    """
    Store a completed investigation in Chroma. Returns True on success.

    Idempotent by investigation_id: re-storing overwrites.
    """
    if state.final_brief is None:
        return False

    doc = _investigation_to_document(state.original_question, state.final_brief, state.structured_question)
    md = _investigation_to_metadata(state.investigation_id, state.original_question, state.final_brief, state.structured_question)

    coll = _get_collection()
    try:
        coll.upsert(
            ids=[state.investigation_id],
            documents=[doc],
            metadatas=[md],
        )
        return True
    except Exception as e:
        # Never let memory write failure break an investigation
        print(f"  memory: store failed for {state.investigation_id}: {e}")
        return False


def query_similar(
    question: str,
    n_results: int = 3,
    min_similarity: float = 0.3,
) -> list[PriorInvestigation]:
    """
    Return up to n_results PriorInvestigation objects semantically similar
    to the given question. Filters out results below min_similarity.

    ChromaDB returns distance (lower = more similar). We convert to
    similarity score in [0, 1] via 1 / (1 + distance) which is
    monotonic and interpretable.
    """
    coll = _get_collection()
    try:
        result = coll.query(
            query_texts=[question],
            n_results=n_results,
        )
    except Exception as e:
        print(f"  memory: query failed: {e}")
        return []

    if not result or not result.get("ids") or not result["ids"][0]:
        return []

    ids = result["ids"][0]
    docs = result["documents"][0] if result.get("documents") else [""] * len(ids)
    metas = result["metadatas"][0] if result.get("metadatas") else [{}] * len(ids)
    distances = result["distances"][0] if result.get("distances") else [0.0] * len(ids)

    priors: list[PriorInvestigation] = []
    for inv_id, doc, md, dist in zip(ids, docs, metas, distances):
        similarity = 1.0 / (1.0 + float(dist))
        if similarity < min_similarity:
            continue
        top_cause = md.get("top_cause", "")
        priors.append(
            PriorInvestigation(
                id=str(inv_id),
                question=str(md.get("question", "")),
                similarity_score=similarity,
                summary=doc[:1000],
                key_causes=[top_cause] if top_cause else [],
                verdict_confidence=(
                    float(md["top_cause_confidence"]) if "top_cause_confidence" in md else None
                ),
            )
        )
    return priors


def get_collection_stats() -> dict:
    """Simple health check for the memory store."""
    coll = _get_collection()
    try:
        count = coll.count()
        return {
            "collection_name": COLLECTION_NAME,
            "path": str(config.chroma_path),
            "count": count,
        }
    except Exception as e:
        return {"error": str(e)}


def clear_memory() -> None:
    """Development helper: delete the collection entirely. Use with care."""
    client = _get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass