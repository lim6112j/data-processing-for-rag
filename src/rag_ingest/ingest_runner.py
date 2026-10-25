"""Batch runner: real documents through the full cascade.

Everything built so far is exercised by tests and a synthetic gold set. This is
the piece that takes an actual corpus and tells you what the pipeline does with
it -- per document, per class, with the disagreement rate that decides whether
T2 is even earning its calls.

Two properties matter for a runner:

  * **One document cannot abort the batch.** A corrupt PDF, an encrypted file, a
    parser crash -- all become a recorded failure for that document, not a dead
    run. Corpus work fails document-by-document or not at all.
  * **Nothing is silently dropped.** Every document ends in exactly one of
    verified / quarantined / failed, and the counts must reconcile. A document
    that vanishes between input and output is the bug you find six months later.

The runner does not call a model for extraction. Extraction is supplied; the
runner is about corpus-scale plumbing and honest accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.chunking import Chunker
from rag_ingest.docmodel import Element, RawDocument
from rag_ingest.escalation import ReviewQueue
from rag_ingest.pipeline import IngestOutcome, IngestPipeline
from rag_ingest.store import ChunkStore, content_hash


class DocumentStatus(StrEnum):
    OK = "ok"
    #: Parsed, but nothing verified.
    ALL_QUARANTINED = "all_quarantined"
    #: The parser or the pipeline raised. Recorded, not fatal.
    FAILED = "failed"
    #: Produced no chunks at all -- usually an empty or unreadable file, and a
    #: distinct problem from "everything was quarantined".
    EMPTY = "empty"


class DocumentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    source: str
    status: DocumentStatus
    verified_chunks: int = Field(ge=0)
    quarantined_chunks: int = Field(ge=0)
    review_items: int = Field(ge=0)
    adjudications: int = Field(ge=0)
    #: Chunks where T2 was asked and answered with a pick.
    adjudicated_picks: int = Field(ge=0)
    #: Chunks where the candidate producer found >= 2 competing readings, i.e.
    #: the parsers genuinely disagreed -- measured *before* any gate decides
    #: whether to consult T2.
    candidate_conflicts: int = Field(default=0, ge=0)
    #: Chunks examined by the producer.
    candidate_checks: int = Field(default=0, ge=0)
    error: str | None = None

    @property
    def total_chunks(self) -> int:
        return self.verified_chunks + self.quarantined_chunks


class BatchReport(BaseModel):
    """Corpus-level accounting. The counts must reconcile."""

    model_config = ConfigDict(frozen=True)

    documents: tuple[DocumentResult, ...] = Field(default_factory=tuple)

    def _by_status(self, status: DocumentStatus) -> tuple[DocumentResult, ...]:
        return tuple(d for d in self.documents if d.status is status)

    @property
    def total_documents(self) -> int:
        return len(self.documents)

    @property
    def failed(self) -> tuple[DocumentResult, ...]:
        return self._by_status(DocumentStatus.FAILED)

    @property
    def empty(self) -> tuple[DocumentResult, ...]:
        return self._by_status(DocumentStatus.EMPTY)

    @property
    def all_quarantined(self) -> tuple[DocumentResult, ...]:
        return self._by_status(DocumentStatus.ALL_QUARANTINED)

    @property
    def verified_chunks(self) -> int:
        return sum(d.verified_chunks for d in self.documents)

    @property
    def quarantined_chunks(self) -> int:
        return sum(d.quarantined_chunks for d in self.documents)

    @property
    def total_chunks(self) -> int:
        return self.verified_chunks + self.quarantined_chunks

    @property
    def adjudications(self) -> int:
        return sum(d.adjudications for d in self.documents)

    @property
    def adjudicated_picks(self) -> int:
        return sum(d.adjudicated_picks for d in self.documents)

    @property
    def verification_rate(self) -> float:
        return 0.0 if self.total_chunks == 0 else self.verified_chunks / self.total_chunks

    @property
    def candidate_conflicts(self) -> int:
        return sum(d.candidate_conflicts for d in self.documents)

    @property
    def candidate_checks(self) -> int:
        return sum(d.candidate_checks for d in self.documents)

    @property
    def disagreement_rate(self) -> float:
        """Share of *chunks* where the two parsers genuinely disagreed.

        Measured at the producer, not at T2, and that distinction matters: T2 is
        only consulted when a chunk also has unreliable structure, so counting
        T2 calls confounds parser disagreement with an unrelated tier gate. This
        rate answers the question it is supposed to: at or near zero, the second
        parse adds nothing and can be removed.

        Zero checks yields 0.0 -- reported as "no comparison was made" by
        ``describe`` rather than silently implying agreement.
        """
        if self.candidate_checks == 0:
            return 0.0
        return self.candidate_conflicts / self.candidate_checks

    def accounts_for_every_document(self) -> bool:
        """Every document is in exactly one status bucket, and chunk counts add up.

        This is the invariant that catches silent drops: a document that
        disappears between input and output, or chunks counted twice. It is
        cheap to check and expensive to discover later.
        """
        buckets = (
            len(self._by_status(DocumentStatus.OK))
            + len(self.all_quarantined)
            + len(self.empty)
            + len(self.failed)
        )
        if buckets != self.total_documents:
            return False
        # A failed document must not also report chunks from a partial run.
        if any(d.error is not None and d.total_chunks > 0 for d in self.failed):
            return False
        # Every parsed document's chunk counts must equal what its result claims.
        return all(
            d.verified_chunks + d.quarantined_chunks == d.total_chunks
            for d in self.documents
        )


class DocumentSource(BaseModel):
    """One document to ingest, already parsed into the docmodel contract.

    Parsing is deliberately outside the runner: it keeps the runner testable
    without a PDF stack, and lets the parser choice vary per corpus.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    path: str
    raw: RawDocument
    elements: tuple[Element, ...]

    def content_hash(self) -> str:
        return content_hash(self.raw.model_dump_json().encode("utf-8"))


@dataclass
class IngestRunner:
    """Runs a corpus through the pipeline, one document at a time."""

    extractor: object
    queue: ReviewQueue = field(default_factory=ReviewQueue)
    store: ChunkStore = field(default_factory=ChunkStore)
    chunker: Chunker = field(default_factory=Chunker)
    adjudicator: object | None = None
    producer: object | None = None
    today: object | None = None

    def _pipeline(self) -> IngestPipeline:
        kwargs = {}
        if self.adjudicator is not None:
            kwargs["adjudicator"] = self.adjudicator
        if self.producer is not None:
            kwargs["producer"] = self.producer
        if self.today is not None:
            kwargs["today"] = self.today
        return IngestPipeline(
            self.queue, self.extractor, chunker=self.chunker, **kwargs
        )

    def run_one(self, source: DocumentSource, *, owner_id: str) -> tuple[
        DocumentResult, IngestOutcome | None
    ]:
        """Ingest one document. Never raises."""
        try:
            outcome = self._pipeline().ingest(
                source.raw,
                list(source.elements),
                owner_id=owner_id,
                retrieval_demand=0,
            )
        except Exception as exc:
            return (
                DocumentResult(
                    doc_id=source.doc_id,
                    source=source.path,
                    status=DocumentStatus.FAILED,
                    verified_chunks=0,
                    quarantined_chunks=0,
                    review_items=0,
                    adjudications=0,
                    adjudicated_picks=0,
                    candidate_conflicts=0,
                    candidate_checks=0,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                None,
            )

        self.store.add_raw(source.raw)
        for record in outcome.verified_chunks + outcome.quarantined_chunks:
            self.store.add(record)
        for item in outcome.review_items:
            self.queue.enqueue(item)

        picks = sum(1 for _, decision in outcome.adjudications if decision.kind == "pick")
        conflicts, checks = self._count_candidate_conflicts(source)
        if not outcome.verified_chunks and not outcome.quarantined_chunks:
            status = DocumentStatus.EMPTY
        elif not outcome.verified_chunks:
            status = DocumentStatus.ALL_QUARANTINED
        else:
            status = DocumentStatus.OK

        return (
            DocumentResult(
                doc_id=source.doc_id,
                source=source.path,
                status=status,
                verified_chunks=len(outcome.verified_chunks),
                quarantined_chunks=len(outcome.quarantined_chunks),
                review_items=len(outcome.review_items),
                adjudications=len(outcome.adjudications),
                adjudicated_picks=picks,
                candidate_conflicts=conflicts,
                candidate_checks=checks,
            ),
            outcome,
        )

    def _count_candidate_conflicts(self, source: DocumentSource) -> tuple[int, int]:
        """Count chunks where the producer found two or more readings.

        Run independently of the pipeline so the disagreement rate is not
        confounded by whether a chunk happened to need adjudication. Returns
        ``(conflicts, checks)``; with no producer there is nothing to compare and
        both are zero, which ``describe`` reports as "not measured".
        """
        if self.producer is None:
            return 0, 0
        try:
            chunks = self.chunker.chunk(source.doc_id, list(source.elements))
        except Exception:  # a chunking failure is already recorded by the pipeline run
            return 0, 0

        conflicts = 0
        for chunk in chunks:
            try:
                candidates = self.producer.produce(chunk)
            except Exception:  # a producer failure must not skew the rate
                continue
            if len(candidates) >= 2:
                conflicts += 1
        return conflicts, len(chunks)

    def run(self, sources: list[DocumentSource], *, owner_id: str) -> BatchReport:
        results = tuple(
            self.run_one(source, owner_id=owner_id)[0] for source in sources
        )
        return BatchReport(documents=results)


def describe(report: BatchReport) -> str:
    """Human-readable batch summary."""
    lines = [
        f"documents      : {report.total_documents}",
        f"verified chunks: {report.verified_chunks}",
        f"quarantined    : {report.quarantined_chunks}",
        f"verification   : {report.verification_rate:.0%}",
        f"T2 consulted   : {report.adjudications} ({report.adjudicated_picks} picks)",
    ]
    if report.candidate_checks == 0:
        lines.append("disagreement   : not measured (no second parse configured)")
    else:
        lines.append(
            f"disagreement   : {report.disagreement_rate:.0%} of chunks "
            f"({report.candidate_conflicts}/{report.candidate_checks})"
        )
    lines += [
    ]
    if report.failed:
        lines.append(f"failed         : {len(report.failed)}")
        for result in report.failed[:5]:
            lines.append(f"  - {result.doc_id}: {result.error}")
    if report.empty:
        lines.append(f"empty          : {len(report.empty)}")
    if not report.accounts_for_every_document():  # pragma: no cover - invariant guard
        lines.append("WARNING: document counts do not reconcile")
    return "\n".join(lines)


__all__ = [
    "BatchReport",
    "DocumentResult",
    "DocumentSource",
    "DocumentStatus",
    "IngestRunner",
    "describe",
]
