"""Demo session: what one browser tab is looking at.

Separated from the HTTP layer so the behaviour is testable without a server,
and so "what the demo shows" is expressible as data rather than template logic.

In-memory and single-process. Appropriate for a demo, inappropriate for a
deployment -- stated once so it is not mistaken for a design decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.adjudication import AdjudicationDecision
from rag_ingest.chunking import Chunk, Chunker
from rag_ingest.circuit import BreakerStatus, T2CircuitBreaker
from rag_ingest.docmodel import BBox, Element, ElementKind, ProvenanceRef, RawDocument
from rag_ingest.escalation import Correction, ReviewerRole, ReviewItem, ReviewQueue
from rag_ingest.extractors import LayeredExtractor
from rag_ingest.pipeline import IngestOutcome, IngestPipeline
from rag_ingest.scoring import T2ScoreCard, feed_breaker, score

if TYPE_CHECKING:
    from rag_ingest.shapes import ShapeRegistry


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
    #: Rows verified against adopted row contracts at ingest time, as
    #: (status, row_text) pairs. Empty unless a contract matched.
    row_verifications: tuple[tuple[str, str], ...] = Field(default_factory=tuple)

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
    #: Layered: typed extraction first, verbatim prose fallback. Prose is
    #: quotable, not unknowable -- without the fallback every prose chunk
    #: stays unclassified and the LENIENT tier is unreachable in the demo.
    extractor: object = field(default_factory=LayeredExtractor)
    adjudicator: object | None = None
    producer: object | None = None
    today: date = field(default_factory=date.today)
    #: Optional LLM proposer client (OpenRouter-style). When set, clusters the
    #: grammar proposer refuses fall back to the model -- whose claims are
    #: labelled source="llm" and still pass through the adoption gate.
    llm_proposer: object | None = None

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
    #: Per-document outcomes, kept so discovery can cluster the quarantined
    #: chunks after the fact. The view records decisions; the outcome holds
    #: the records discovery reads.
    outcomes: dict[str, IngestOutcome] = field(default_factory=dict)
    #: Adopted contracts are persisted here (JSON of ``to_definition`` payloads)
    #: and reloaded on construction -- adoption must survive a restart.
    shapes_path: Path | None = None
    _shapes: ShapeRegistry | None = None

    def __post_init__(self) -> None:
        if self.shapes_path is not None and self.shapes_path.exists():
            try:
                payload = json.loads(self.shapes_path.read_text())
            except (json.JSONDecodeError, OSError):
                # A corrupt file must not brick the session: fail closed to
                # builtins rather than start with a partial registry.
                return
            if isinstance(payload, list):
                from pydantic import ValidationError

                from rag_ingest.shapes import ShapeContract

                for definition in payload:
                    try:
                        self.registry().register(ShapeContract.from_definition(definition))
                    except (ValueError, ValidationError):
                        continue  # skip the bad entry, keep the good ones

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
            registry=self.registry(),
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

        self.outcomes[doc_id] = outcome
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
            row_verifications=tuple(
                (v.status, v.row_text) for v in outcome.row_verifications
            ),
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

    # -- shape discovery ---------------------------------------------------

    def discovery_report(self):
        """Shape clusters across every ingested document's quarantined chunks.

        Quarantined table chunks are routed to rows first: a table is not one
        fact, it is N rows sharing a shape, and row recurrence -- not blob
        recurrence -- is what a row-shape declaration should be asked about.
        """
        from rag_ingest.discovery import discover, discover_from_rows

        quarantined: list = []
        row_observations = []
        for outcome in self.outcomes.values():
            quarantined.extend(outcome.quarantined_chunks)
            row_observations.extend(outcome.row_observations)
        if not quarantined and not row_observations:
            return None
        report = discover(tuple(quarantined)) if quarantined else None
        row_report = discover_from_rows(tuple(row_observations)) if row_observations else None
        if report is None and row_report is None:
            return None
        if report is None:
            return row_report if row_report.unclassified_total else None
        if row_report is not None and row_report.unclassified_total:
            from dataclasses import replace

            report = replace(
                report,
                clusters=report.clusters + row_report.clusters,
                unclassified_total=report.unclassified_total + row_report.unclassified_total,
            )
        return report if report.unclassified_total else None

    def propose_for(self, shape: str):
        """A schema claim for one cluster, from its sample rows.

        The grammar proposer needs rows; the quarantined chunks are prose-shaped
        text, so the samples are their raw text split on the table pipe the
        shape hint was derived from. A cluster whose rows do not parse yields
        an honest refusal -- the UI shows that as "cannot propose", never as a
        guessed schema.
        """
        from rag_ingest.propose import propose as build_proposal

        rows: list[tuple[str, ...]] = []
        if shape.startswith("row:"):
            # Row clusters propose from the routed row cells directly -- no
            # re-parsing of chunk text, which would re-derive from a
            # different grammar than the cluster was keyed on.
            from rag_ingest.discovery import row_signatures

            all_rows = tuple(
                row for outcome in self.outcomes.values() for row in outcome.row_observations
            )
            signatures = row_signatures(all_rows)
            rows = [
                row.cells for row in all_rows
                if signatures[id(row)] == shape
            ][:5]
            if not rows:
                return None
            proposal = build_proposal(shape, tuple(rows), shape_name=f"{shape}_spec")
            if proposal.parseable or self.llm_proposer is None:
                return proposal if proposal.parseable else None
            return self._llm_fallback(shape, tuple(rows))
        for outcome in self.outcomes.values():
            for record in outcome.quarantined_chunks:
                extraction = record.extraction
                if extraction is None or getattr(extraction, "kind", "") != "unclassified":
                    continue
                if extraction.observed_shape != shape:
                    continue
                for line in record.text.strip().splitlines():
                    cells = tuple(c.strip() for c in line.split("|"))
                    if len(cells) > 1:
                        rows.append(cells)
        if not rows:
            return None
        proposal = build_proposal(shape, tuple(rows[:5]), shape_name=f"{shape}_spec")
        if proposal.parseable or self.llm_proposer is None:
            return proposal if proposal.parseable else None
        return self._llm_fallback(shape, tuple(rows[:5]))

    def _llm_fallback(self, shape: str, rows: tuple[tuple[str, ...], ...]):
        """Ask the model for a claim the grammar could not make.

        The fallback fires only after the grammar refused. A refusal from the
        model is still a refusal: the caller sees None, never a guessed
        schema. This method exists so the owner's conversation can continue
        for clusters the regexes cannot classify -- with the label that says
        a model, not a grammar, proposed it.
        """
        from rag_ingest.llm_propose import LlmProposer

        proposer = LlmProposer(client=self.llm_proposer)
        proposal = proposer.propose(shape, rows)
        return proposal if proposal.parseable else None

    def registry(self):
        """The runtime shape registry: built-ins plus anything adopted."""
        from rag_ingest.shapes import ShapeRegistry

        if self._shapes is None:
            self._shapes = ShapeRegistry.with_builtins()
        return self._shapes

    def adopt(self, contract) -> None:
        """Make an owner-approved contract effective immediately -- no restart.

        Persistence is the caller's deployment concern (the contract serializes
        with ``to_definition``); adopting here is the runtime act.
        """
        self.registry().register(contract)

    def adopt_with_gate(self, contract, cases):
        """Run the adoption gate; register only on a full pass.

        The cases are about the contract's semantics (does max>=min hold? does
        the enum bite?), not about span arithmetic, so each case's context is
        its own claimed values joined into the span text: the gate re-derives
        from a buffer where every claimed value is verbatim. A failed adoption
        leaves the registry untouched and returns the falsifications.
        """
        from rag_ingest.docmodel import Element, RawDocument
        from rag_ingest.grounding import GroundingGate
        from rag_ingest.shapes import adopt_contract

        raw, _ = RawDocument.from_elements(
            f"adoption:{contract.name}",
            [
                Element(
                    element_id="e1",
                    kind=ElementKind.PARAGRAPH,
                    page=1,
                    text="\n".join(
                        " ".join(str(v) for v in case.values.values()) for case in cases
                    ),
                )
            ],
        )
        gate = GroundingGate(raw, shapes=self.registry())
        result = adopt_contract(gate, self.registry(), contract, cases)
        if result.registered:
            self._persist_shapes()
        return result

    def _persist_shapes(self) -> None:
        from rag_ingest.shapes import ShapeRegistry

        """Write every adopted contract to disk.

        Builtins are excluded: they come from code and must not be duplicated
        into a file that would then shadow future code changes. Only contracts
        adopted through the gate are recorded.
        """
        if self.shapes_path is None:
            return
        builtin_names = {c.name for c in ShapeRegistry.with_builtins()._contracts.values()}
        adopted = [
            c.to_definition()
            for name, c in self.registry()._contracts.items()
            if name not in builtin_names
        ]
        self.shapes_path.parent.mkdir(parents=True, exist_ok=True)
        self.shapes_path.write_text(json.dumps(adopted, indent=2))

    def check_against_shape(self, shape_name: str, text: str, values: dict[str, str]):
        """Verify a new chunk's values against an adopted shape.

        The same gate, the same span re-derivation: this exists so the UI (and
        tests) can demonstrate the adopted contract biting real values.
        """
        from rag_ingest.grounding import GroundingGate
        from rag_ingest.ontology import FieldValue

        raw, located = RawDocument.from_elements(
            f"shape-check:{shape_name}",
            [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1, text=text)],
        )
        located_element = located[0]
        ref = ProvenanceRef(
            page=located_element.page,
            char_start=located_element.char_start,
            char_end=located_element.char_end,
            element_id=located_element.element_id,
        )
        gate = GroundingGate(raw, shapes=self.registry())
        claimed = {
            name: FieldValue(value=value, provenance=ref) for name, value in values.items()
        }
        return gate.check_contract(shape_name, claimed)

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
