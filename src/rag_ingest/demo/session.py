"""Demo session: what one browser tab is looking at.

Separated from the HTTP layer so the behaviour is testable without a server,
and so "what the demo shows" is expressible as data rather than template logic.

In-memory and single-process. Appropriate for a demo, inappropriate for a
deployment -- stated once so it is not mistaken for a design decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.adjudication import AdjudicationDecision
from rag_ingest.chunking import Chunk, Chunker
from rag_ingest.circuit import BreakerStatus, T2CircuitBreaker
from rag_ingest.docmodel import BBox, Element, RawDocument
from rag_ingest.escalation import Correction, ReviewerRole, ReviewItem, ReviewQueue
from rag_ingest.extractors import TemplateExtractor
from rag_ingest.pipeline import IngestOutcome, IngestPipeline
from rag_ingest.scoring import T2ScoreCard, feed_breaker, score


class ChunkView(BaseModel):
    """A chunk as the UI shows it: text, verdict, and where it came from."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    kind: str
    heading: str
    verified: bool
    reason: str | None = None
    page: int = Field(ge=1)
    bbox: BBox | None = None
    extraction: dict | None = None


class PageView(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_no: int = Field(ge=1)
    width: float | None = None
    height: float | None = None
    has_image: bool = False


class TierView(BaseModel):
    """One element's route through the cascade, and why it went there."""

    model_config = ConfigDict(frozen=True)

    element_id: str
    tier: str
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    layout_score: float | None = None
    ocr_score: float | None = None
    parse_score: float | None = None


class ReviewItemView(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    chunk_id: str
    reason: str
    question: str
    priority: int
    resolved: bool
    candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    adjudicated_choice: str | None = None


class IngestView(BaseModel):
    """Everything one document contributed to the screen."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    filename: str
    parser: str
    pages: tuple[PageView, ...] = Field(default_factory=tuple)
    chunks: tuple[ChunkView, ...] = Field(default_factory=tuple)
    tiers: tuple[TierView, ...] = Field(default_factory=tuple)
    review_items: tuple[ReviewItemView, ...] = Field(default_factory=tuple)
    verified_count: int = 0
    quarantined_count: int = 0

    @property
    def verification_rate(self) -> float:
        total = self.verified_count + self.quarantined_count
        return 0.0 if total == 0 else self.verified_count / total


@dataclass
class DemoSession:
    """Documents, review queue, and the T2 health signal for one demo run."""

    # Deliberately not LayeredExtractor, unlike the CLI's corpus mode. The demo
    # exists to show a verdict per region -- green entered retrieval, red withheld.
    # Admitting every prose chunk as a verbatim quote would paint the whole page
    # green and demonstrate nothing. Use the corpus run to exercise the LENIENT tier.
    extractor: object = field(default_factory=TemplateExtractor)
    adjudicator: object | None = None
    producer: object | None = None
    today: date = field(default_factory=date.today)

    #: Demo-only: force every chunk down the structure-ambiguous route.
    #:
    #: Needed because the real pipeline almost never reaches T2 -- pdfplumber
    #: reports no scores, and Docling's layout scores on readable documents sit
    #: above the threshold, so ``STRUCTURE_UNKNOWN`` does not fire. That is
    #: correct behaviour, and it makes the T2 branch unreachable in a
    #: walkthrough without this switch. It is a demonstration control, not a
    #: simulation of a real defect, and the UI labels it as such.
    force_structure_ambiguous: bool = False

    queue: ReviewQueue = field(default_factory=ReviewQueue)
    documents: dict[str, RawDocument] = field(default_factory=dict)
    chunk_index: dict[str, Chunk] = field(default_factory=dict)
    views: dict[str, IngestView] = field(default_factory=dict)
    _breaker: T2CircuitBreaker | None = None

    # -- ingestion --------------------------------------------------------

    def ingest(
        self,
        *,
        doc_id: str,
        filename: str,
        parser: str,
        raw: RawDocument,
        elements: list[Element],
    ) -> IngestView:
        pipeline = IngestPipeline(
            self.queue,
            self.extractor,
            chunker=Chunker(),
            today=self.today,
            adjudicator=self._wrapped_adjudicator(),
            producer=self.producer,
        )
        if self.force_structure_ambiguous:
            outcome = self._force_ambiguous(pipeline, raw, elements)
        else:
            outcome = pipeline.ingest(
                raw, elements, owner_id="demo-owner", retrieval_demand=5
            )

        # Re-uploading the same file produces the same doc_id, hence the same
        # chunk ids, hence the same item ids. Drop the previous parse's items
        # first: they describe a document state that no longer exists. Their
        # corrections survive -- those are the labels.
        self.queue.discard(doc_id)

        # The pipeline *produces* review items; enqueueing is the caller's job
        # (IngestRunner does the same). Omitting it leaves the review queue
        # empty while the view still lists items -- the loop would look alive
        # and accept nothing.
        for item in outcome.review_items:
            self.queue.enqueue(item)

        chunks = pipeline._chunker.chunk(raw.doc_id, list(elements))
        self.documents[doc_id] = raw
        self.chunk_index.update({chunk.chunk_id: chunk for chunk in chunks})

        view = IngestView(
            doc_id=doc_id,
            filename=filename,
            parser=parser,
            pages=tuple(
                PageView(
                    page_no=page.page_no,
                    width=page.width,
                    height=page.height,
                    has_image=page.has_image,
                )
                for page in raw.pages
            ),
            chunks=self._chunk_views(outcome, chunks),
            tiers=self._tier_views(outcome),
            review_items=self._item_views(outcome),
            verified_count=len(outcome.verified_chunks),
            quarantined_count=len(outcome.quarantined_chunks),
        )
        self.views[doc_id] = view
        return view

    def _force_ambiguous(self, pipeline, raw, elements) -> IngestOutcome:
        """Run the pipeline with every element demoted to STRUCTURE_UNKNOWN.

        Done by lowering the layout score rather than by patching the branch, so
        the decision under test is still the real ``select_tier`` logic.
        """
        from rag_ingest.docmodel import ElementScores

        demoted = [
            element.model_copy(
                update={
                    "scores": ElementScores(
                        layout_score=0.2,
                        ocr_score=element.scores.ocr_score,
                        parse_score=element.scores.parse_score,
                    )
                }
            )
            for element in elements
        ]
        return pipeline.ingest(raw, demoted, owner_id="demo-owner", retrieval_demand=5)

    def _wrapped_adjudicator(self):
        if self.adjudicator is None:
            return None
        if self._breaker is None:
            self._breaker = T2CircuitBreaker(self.adjudicator)
        return self._breaker

    # -- review -----------------------------------------------------------

    def open_items(self) -> list[ReviewItem]:
        return self.queue.open_items()

    def resolve(
        self,
        item_id: str,
        *,
        after: str,
        choice: str | None = None,
        reviewer_role: ReviewerRole = ReviewerRole.OWNER,
        reviewer_id: str = "demo-owner",
    ) -> T2ScoreCard:
        """Record a correction, then refresh the T2 health signal.

        The order matters: the correction is scored first, so the breaker sees
        the new evidence, and the returned scorecard reflects it.
        """
        item = next((i for i in self.queue.open_items() if i.item_id == item_id), None)
        if item is None:
            raise KeyError(f"no open review item {item_id}")

        correction = Correction(
            doc_id=item.doc_id,
            chunk_id=item.chunk_id,
            field="structure" if item.candidate_ids else "content",
            validator_error=item.reason.value,
            before=item.adjudicated_choice,
            after=after,
            reviewer_id=reviewer_id,
            reviewer_role=reviewer_role,
            captured_on=self.today,
            adjudication_choice=choice,
        )
        self.queue.resolve(item_id, correction)

        card = self.scorecard()
        if self._breaker is not None:
            feed_breaker(card, self._breaker)
        return card

    def scorecard(self) -> T2ScoreCard:
        return score(list(self.queue.corrections()), self.queue.items_by_chunk())

    def breaker_status(self) -> BreakerStatus | None:
        return None if self._breaker is None else self._breaker.status

    # -- accessors for the HTTP layer -------------------------------------

    def page_image(self, doc_id: str, page_no: int) -> bytes | None:
        raw = self.documents.get(doc_id)
        if raw is None:
            return None
        page = raw.page(page_no)
        return None if page is None else page.image_png

    def overlay(self, doc_id: str) -> dict:
        """Boxes to draw over pages, and the element boxes behind them.

        Chunk boxes carry the verdict; element boxes show what the parser saw.
        Keeping them separate is what lets the demo distinguish "the parser
        found a table here" from "this region was verified and is retrievable".
        """
        view = self.views.get(doc_id)
        if view is None:
            return {"chunks": [], "elements": []}

        chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "bbox": list(chunk.bbox) if chunk.bbox else None,
                "verified": chunk.verified,
                "reason": chunk.reason,
                "kind": chunk.kind,
            }
            for chunk in view.chunks
            if chunk.bbox is not None
        ]

        elements = []
        raw = self.documents.get(doc_id)
        if raw is not None:
            # Element boxes are re-derived from the chunks' segments so the
            # overlay reflects what actually reached the store.
            seen = set()
            for chunk in self.chunk_index.values():
                if chunk.doc_id != doc_id:
                    continue
                for segment in chunk.segments:
                    if segment.element_id in seen or segment.bbox is None:
                        continue
                    seen.add(segment.element_id)
                    elements.append(
                        {
                            "element_id": segment.element_id,
                            "page": segment.page,
                            "bbox": list(segment.bbox),
                        }
                    )
        return {"chunks": chunks, "elements": elements}

    # -- view builders ----------------------------------------------------

    def _chunk_views(
        self, outcome: IngestOutcome, chunks: list[Chunk]
    ) -> tuple[ChunkView, ...]:
        reasons = {item.chunk_id: item.reason.value for item in outcome.review_items}
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        verified_ids = {record.chunk_id for record in outcome.verified_chunks}

        views: list[ChunkView] = []
        for record in outcome.verified_chunks + outcome.quarantined_chunks:
            chunk = by_id.get(record.chunk_id)
            extraction = record.extraction
            views.append(
                ChunkView(
                    chunk_id=record.chunk_id,
                    text=record.text,
                    kind=chunk.kind.value if chunk else "text",
                    heading=str(chunk.heading_path) if chunk else "",
                    verified=record.chunk_id in verified_ids,
                    reason=(
                        None
                        if record.chunk_id in verified_ids
                        else reasons.get(record.chunk_id)
                    ),
                    page=chunk.page_start if chunk else 1,
                    bbox=chunk.bbox if chunk else None,
                    extraction=extraction.model_dump(mode="json") if extraction else None,
                )
            )
        return tuple(views)

    def _tier_views(self, outcome: IngestOutcome) -> tuple[TierView, ...]:
        return tuple(
            TierView(
                element_id=element_id,
                tier=decision.tier.value,
                reasons=decision.reasons,
                layout_score=decision.scores.layout_score,
                ocr_score=decision.scores.ocr_score,
                parse_score=decision.scores.parse_score,
            )
            for element_id, decision in outcome.tier_decisions
        )

    def _item_views(self, outcome: IngestOutcome) -> tuple[ReviewItemView, ...]:
        return tuple(
            ReviewItemView(
                item_id=item.item_id,
                chunk_id=item.chunk_id,
                reason=item.reason.value,
                question=item.question,
                priority=item.priority,
                resolved=item.resolved,
                candidate_ids=item.candidate_ids,
                adjudicated_choice=item.adjudicated_choice,
            )
            for item in outcome.review_items
        )

    def fresh_review_items(self) -> list[ReviewItemView]:
        """Open items across all documents, highest priority first."""
        items = [
            ReviewItemView(
                item_id=item.item_id,
                chunk_id=item.chunk_id,
                reason=item.reason.value,
                question=item.question,
                priority=item.priority,
                resolved=item.resolved,
                candidate_ids=item.candidate_ids,
                adjudicated_choice=item.adjudicated_choice,
            )
            for item in self.queue.open_items()
        ]
        return sorted(items, key=lambda i: (-i.priority, i.item_id))


def decision_is_pick(decision: AdjudicationDecision) -> bool:
    """Small helper the UI uses to count picks without re-importing internals."""
    return decision.kind == "pick"


__all__ = [
    "ChunkView",
    "DemoSession",
    "IngestView",
    "PageView",
    "ReviewItemView",
    "TierView",
]
