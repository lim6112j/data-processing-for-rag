"""Retrieval-side policy: fail closed per chunk, never per document.

Plan §5a: a single bad table must not block a 200-page document. Chunks that
grounded are retrievable with ``verified=True``; everything else is stored but
excluded unless the caller explicitly opts in.

Plan §7-3: raw page text stays reachable, so content whose normalized chunks
were quarantined is never invisible to retrieval.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.docmodel import RawDocument
from rag_ingest.ontology import ChunkRecord


class VerificationFilter(StrEnum):
    #: Default. Only chunks that passed grounding.
    VERIFIED_ONLY = "verified_only"
    #: Include unverified chunks, ranked below verified ones. Useful for recall
    #: experiments; never the default.
    INCLUDE_UNVERIFIED = "include_unverified"


class RetrievalHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    record: ChunkRecord
    score: float
    #: True when the hit is unverified and therefore ranked down.
    rank_penalized: bool = False


class ChunkStore:
    """In-memory store. The policy is the substance, not the persistence."""

    def __init__(self) -> None:
        self._records: dict[str, ChunkRecord] = {}
        self._raw: dict[str, RawDocument] = {}

    def add_raw(self, raw: RawDocument) -> None:
        self._raw[raw.doc_id] = raw

    def add(self, record: ChunkRecord) -> None:
        if record.doc_id not in self._raw:
            raise ValueError(
                f"raw document {record.doc_id} must be stored before its chunks"
            )
        self._records[record.chunk_id] = record

    def record(self, chunk_id: str) -> ChunkRecord | None:
        return self._records.get(chunk_id)

    def raw_text(self, doc_id: str, page_no: int) -> str | None:
        raw = self._raw.get(doc_id)
        if raw is None:
            return None
        page = raw.page(page_no)
        return None if page is None else page.text

    def search(
        self,
        terms: str,
        *,
        mode: VerificationFilter = VerificationFilter.VERIFIED_ONLY,
        limit: int = 10,
    ) -> list[RetrievalHit]:
        """Naive lexical search. Stands in for a vector index.

        Implemented as substring scoring so the *policy* -- what is allowed to
        appear at all, and how unverified content is ranked -- can be tested
        without an embedding model.
        """
        needle = terms.casefold().strip()
        hits: list[RetrievalHit] = []
        for record in self._records.values():
            if mode is VerificationFilter.VERIFIED_ONLY and not record.verified:
                continue
            haystack = record.embed_text.casefold()
            if not needle:
                score = 0.0
            elif needle in haystack:
                score = 1.0
            else:
                tokens = [t for t in needle.split() if t]
                if not tokens:
                    continue
                matched = sum(1 for token in tokens if token in haystack)
                if matched == 0:
                    continue
                score = matched / len(tokens)
            penalized = not record.verified
            if penalized:
                score *= 0.5
            hits.append(RetrievalHit(record=record, score=score, rank_penalized=penalized))

        hits.sort(key=lambda hit: (-hit.score, hit.record.chunk_id))
        return hits[:limit]

    def all_records(self) -> tuple[ChunkRecord, ...]:
        return tuple(self._records.values())


class IngestedDocument(BaseModel):
    """Bookkeeping record proving a re-ingest replaced, not duplicated."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    content_hash: str
    pipeline_version: str
    ontology_version: str
    chunk_ids: tuple[str, ...] = Field(default_factory=tuple)


def content_hash(payload: bytes) -> str:
    """Content-addressed document identity (plan §7-4).

    Parsers change and documents must be re-ingested; without a stable identity
    the corpus accumulates duplicate and conflicting chunks.
    """
    import hashlib

    return hashlib.sha256(payload).hexdigest()
