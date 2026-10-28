"""T2 contract and its two enforced invariants.

Invariant 1: T2 never receives a grounding failure.
Invariant 2: T2 can never verify anything.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from rag_ingest.adjudication import (
    AdjudicationDecision,
    Candidate,
    DisagreementKind,
    build_question,
)
from rag_ingest.docmodel import Element, ElementKind, ProvenanceRef, RawDocument
from rag_ingest.escalation import EscalationReason, ReviewQueue
from rag_ingest.ontology import AbstainReason, FieldAbstention, FieldValue, InvoiceHeader
from rag_ingest.pipeline import Adjudicator, IngestPipeline

TODAY = date(2026, 10, 25)
BBOX = (10.0, 20.0, 300.0, 60.0)


def _ref(*, bbox: tuple[float, float, float, float] | None = BBOX) -> ProvenanceRef:
    return ProvenanceRef(page=1, char_start=0, char_end=10, bbox=bbox)


def _question(*, bbox=BBOX):
    return build_question(
        chunk_id="c1",
        doc_id="d1",
        kind=DisagreementKind.COLUMN_COUNT,
        prompt="How many columns does this table have?",
        candidates=[
            Candidate(candidate_id="a", produced_by="pdfplumber", proposed_text="4 cols"),
            Candidate(candidate_id="b", produced_by="tableformer", proposed_text="5 cols"),
        ],
        provenance=_ref(bbox=bbox),
    )


class TestVisualEvidence:
    def test_question_with_bbox_has_visual_evidence(self):
        assert _question().has_visual_evidence is True

    def test_question_without_bbox_has_no_evidence(self):
        # Without a bbox T2 cannot see the region, so it must abstain rather
        # than judge from text.
        assert _question(bbox=None).has_visual_evidence is False

    def test_raw_document_returns_no_crop_without_image(self):
        raw, _ = RawDocument.from_elements(
            "d1", [Element(element_id="e", kind=ElementKind.PARAGRAPH, page=1, text="x", bbox=BBOX)]
        )

        assert raw.visual_evidence(_ref()) is None

    def test_raw_document_returns_image_when_page_has_one(self):
        raw, located = RawDocument.from_elements(
            "d1",
            [Element(element_id="e", kind=ElementKind.PARAGRAPH, page=1, text="x", bbox=BBOX)],
            page_images={1: b"PNG"},
        )

        assert raw.visual_evidence(located[0].provenance()) == b"PNG"

    def test_bbox_survives_into_provenance(self):
        _, located = RawDocument.from_elements(
            "d1", [Element(element_id="e", kind=ElementKind.PARAGRAPH, page=1, text="x", bbox=BBOX)]
        )

        assert located[0].provenance().bbox == BBOX
        assert located[0].provenance().has_visual_evidence is True


class TestDecisionCoherence:
    def test_pick_requires_in_range_index(self):
        assert AdjudicationDecision(kind="pick", choice_index=1).is_coherent(2)
        assert not AdjudicationDecision(kind="pick", choice_index=2).is_coherent(2)
        assert not AdjudicationDecision(kind="pick").is_coherent(2)

    def test_abstain_requires_reason(self):
        assert AdjudicationDecision(
            kind="abstain", reason=AbstainReason.AMBIGUOUS
        ).is_coherent(2)
        assert not AdjudicationDecision(kind="abstain").is_coherent(2)

    def test_escalate_requires_a_question(self):
        assert AdjudicationDecision(kind="escalate", owner_question="Which?").is_coherent(2)
        assert not AdjudicationDecision(kind="escalate", owner_question="  ").is_coherent(2)

    def test_decision_has_no_value_field(self):
        # A free-text value field would reopen the channel T1 closes.
        with pytest.raises(ValidationError):
            AdjudicationDecision(kind="pick", choice_index=0, value="1234.56")  # type: ignore[call-arg]


class TestQuestionConstruction:
    def test_single_candidate_is_rejected(self):
        with pytest.raises(ValueError):
            build_question(
                chunk_id="c",
                doc_id="d",
                kind=DisagreementKind.COLUMN_COUNT,
                prompt="q",
                candidates=[Candidate(candidate_id="a", produced_by="p", proposed_text="x")],
                provenance=_ref(),
            )


class _RecordingAdjudicator(Adjudicator):
    def __init__(self, decision: AdjudicationDecision) -> None:
        self.decision = decision
        self.seen: list = []

    def adjudicate(self, question):
        self.seen.append(question)
        return self.decision


class _LyingExtractor:
    """Claims a value that is not in the source -- a grounding failure."""

    def __init__(self) -> None:
        self.ref = ProvenanceRef(page=1, char_start=0, char_end=20, bbox=BBOX)

    def extract(self, chunk):
        return InvoiceHeader(
            invoice_number=FieldValue(value="INV-9", provenance=self.ref),
            issue_date=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
            total_amount=FieldValue(value="9999.99", provenance=self.ref),
            currency=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
        )


def _doc():
    return RawDocument.from_elements(
        "d1",
        [
            Element(
                element_id="e1",
                kind=ElementKind.PARAGRAPH,
                page=1,
                text="Invoice INV-9 total 1234.56 KRW",
                bbox=BBOX,
            )
        ],
        page_images={1: b"PNG"},
    )


class TestInvariantGroundingNeverReachesT2:
    def test_ungrounded_value_skips_adjudication_entirely(self):
        adjudicator = _RecordingAdjudicator(
            AdjudicationDecision(kind="pick", choice_index=0)
        )
        raw, elements = _doc()

        outcome = IngestPipeline(
            ReviewQueue(),
            _LyingExtractor(),
            today=TODAY,
            adjudicator=adjudicator,
        ).ingest(raw, elements, owner_id="o")

        # The gate caught it and T2 was never consulted.
        assert adjudicator.seen == []
        assert outcome.adjudications == ()
        assert any(item.reason is EscalationReason.UNGROUNDED for item in outcome.review_items)


class TestInvariantT2CannotVerify:
    def test_pick_does_not_make_a_chunk_retrievable(self):
        # T2 runs, picks a reading, and the chunk STILL stays quarantined.
        adjudicator = _RecordingAdjudicator(
            AdjudicationDecision(kind="pick", choice_index=0, rationale="clear image")
        )
        raw, elements = _doc()

        outcome = IngestPipeline(
            ReviewQueue(),
            _LyingExtractor(),
            today=TODAY,
            adjudicator=adjudicator,
        ).ingest(raw, elements, owner_id="o")

        assert outcome.verified_chunks == ()
        assert len(outcome.quarantined_chunks) == 1


class TestDefaultAdjudicatorIsClosed:
    def test_pipeline_without_t2_escalates_instead_of_guessing(self):
        raw, elements = _doc()
        outcome = IngestPipeline(ReviewQueue(), _LyingExtractor(), today=TODAY).ingest(
            raw, elements, owner_id="o"
        )

        assert outcome.verified_chunks == ()
        assert outcome.review_items
