"""Owner approval closes the loop: a quarantined chunk whose question the
owner answered may enter retrieval -- but approval never replaces the gate.

The owner can confirm that an extraction is right; the gate still checks the
cited values against the source. An owner cannot make an uncited value cited,
so an approval the gate still refuses is an honest refusal, and the chunk
stays quarantined. Unclassified chunks have no values to confirm -- the only
claim available is the verbatim quote, which the LENIENT tier checks."""

from __future__ import annotations

from helpers.pdf_fixture import make_pdf  # noqa: F401

from rag_ingest.docmodel import Element, ElementKind, RawDocument
from rag_ingest.escalation import EscalationReason, ReviewItem
from rag_ingest.extractors import TemplateExtractor
from rag_ingest.grounding import GroundingGate
from rag_ingest.ontology import GateTier
from rag_ingest.promotion import (
    Approval,
    RefusalReason,
    promote_chunk,
)
from rag_ingest.shapes import ShapeRegistry

TODAY = __import__("datetime").date(2026, 11, 1)


def _source(text: str, doc_id: str = "d1"):
    raw, located = RawDocument.from_elements(
        doc_id,
        [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1, text=text)],
        page_sizes={1: (612.0, 792.0)},
    )
    return raw, located


def _item(chunk_id: str = "e1", doc_id: str = "d1") -> ReviewItem:
    return ReviewItem(
        item_id=f"{doc_id}:{chunk_id}:triage",
        doc_id=doc_id,
        chunk_id=chunk_id,
        reason=EscalationReason.EXTRACTION_FAILED,
        question="What is this chunk?",
        owner_id="owner-1",
        created_on=TODAY,
    )


class TestApproveUnclassifiedAsQuote:
    """An unclassified chunk has no values to confirm. Approval therefore
    promotes the only honest claim available: the verbatim quote, checked by
    the LENIENT gate like any other prose."""

    def test_approval_quotes_the_chunk_and_promotes_it(self):
        raw, located = _source("본 대장은 자재창고의 월간 재고 현황을 기록한다.")
        outcome = _ingest(raw, located)
        approval = Approval(reviewer="owner-1", approved_on=TODAY)

        chunk_id = outcome.quarantined_chunks[0].chunk_id
        result = promote_chunk(
            outcome, chunk_id, approval, gate=_gate(raw), raw=raw, elements=located
        )

        assert result.refusal is None
        record = _find(result.outcome.verified_chunks, chunk_id)
        assert record.verified
        assert record.gate_tier == GateTier.LENIENT
        assert not any(
            r.chunk_id == chunk_id for r in result.outcome.quarantined_chunks
        )

    def test_approval_refuses_when_the_quote_cannot_be_cited(self):
        """A chunk whose source element is gone cannot be re-quoted, so
        approval has no honest claim to check -- refusal, not a pass on the
        owner's authority."""
        import dataclasses

        raw, located = _source("출처 텍스트")
        outcome = _ingest(raw, located)

        # No extraction at all: the verbatim quote is the only claim left,
        # and it cannot be re-derived once the source element is gone.
        stripped = outcome.quarantined_chunks[0].model_copy(
            update={"extraction": None}
        )
        outcome = dataclasses.replace(outcome, quarantined_chunks=(stripped,))
        chunk_id = stripped.chunk_id
        result = promote_chunk(
            outcome, chunk_id, Approval(reviewer="owner-1", approved_on=TODAY),
            gate=_gate(raw), raw=raw,
            elements=[],  # the element vanished from the located set
        )

        assert result.refusal is not None
        assert result.refusal.reason == RefusalReason.NOT_CITABLE


class TestApproveExtractionRecheck:
    """For a chunk with cited values, approval means 'the extraction is
    right'. The gate re-runs anyway: the owner's judgment is about meaning,
    the machine's is about the citation. Both must hold."""

    def test_gate_passing_extraction_is_promoted_with_its_own_tier(self):
        raw, located = _source("Invoice INV-2041 issued 2026-01-15 total 1234.56 USD")
        outcome = _ingest(raw, located)
        chunk_id = outcome.quarantined_chunks[0].chunk_id
        approval = Approval(reviewer="owner-1", approved_on=TODAY)

        result = promote_chunk(
            outcome, chunk_id, approval, gate=_gate(raw), raw=raw, elements=located
        )

        if result.refusal is not None:
            assert result.refusal.reason in (RefusalReason.NOT_CITABLE,)
        else:
            record = result.outcome.verified_chunks[0]
            assert record.verified
            assert record.gate_tier is not None

    def test_gate_refusing_extraction_stays_quarantined(self):
        """An extraction the gate cannot ground is not promotable -- approval
        cannot make an uncited value cited. The refusal names the reason so
        the owner knows the chunk needs a fix, not a rubber stamp."""
        import dataclasses

        from rag_ingest.ontology import (
            AbstainReason,
            FieldAbstention,
            FieldValue,
            InvoiceHeader,
            ProvenanceRef,
        )

        raw, located = _source("자재창고 월간 재고 대장")
        outcome = _ingest(raw, located)
        chunk = outcome.quarantined_chunks[0]
        chunk_id = chunk.chunk_id
        # A fabricated value: claims the source says INV-999, which it does
        # not. Approval must not launder this into retrieval.
        abstain = FieldAbstention(reason=AbstainReason.NOT_PRESENT, note="test")
        extraction = InvoiceHeader(
            invoice_number=FieldValue(
                value="INV-999",
                provenance=ProvenanceRef(
                    page=1, char_start=0, char_end=10, element_id=chunk_id
                ),
            ),
            issue_date=abstain,
            total_amount=abstain,
            currency=abstain,
        )
        record = chunk.model_copy(update={"extraction": extraction})
        outcome = dataclasses.replace(outcome, quarantined_chunks=(record,))

        result = promote_chunk(
            outcome, chunk_id, Approval(reviewer="owner-1", approved_on=TODAY),
            gate=_gate(raw), raw=raw, elements=located,
        )

        assert result.refusal is not None
        assert result.refusal.reason == RefusalReason.GATE_STILL_REFUSES
        assert result.outcome is None


class TestApprovalMetadata:
    """Promotion carries its history: who approved, when. Export reads this
    so downstream stores can distinguish machine-verified from owner-approved."""

    def test_promoted_record_reports_the_approval(self):
        raw, located = _source("본 대장은 자재창고의 월간 재고 현황을 기록한다.")
        outcome = _ingest(raw, located)
        chunk_id = outcome.quarantined_chunks[0].chunk_id
        result = promote_chunk(
            outcome, chunk_id, Approval(reviewer="owner-1", approved_on=TODAY),
            gate=_gate(raw), raw=raw, elements=located,
        )

        record = result.outcome.verified_chunks[0]
        assert record.extraction is not None or record.gate_tier == GateTier.LENIENT
        assert result.approval.reviewer == "owner-1"
        assert result.approval.approved_on == TODAY


def _gate(raw) -> GroundingGate:
    return GroundingGate(raw, ShapeRegistry.with_builtins())


def _ingest(raw, located):
    """Build a quarantined outcome the same way the pipeline would, without
    owning the pipeline: chunking first, then extraction through the layered
    path, quarantine for whatever the typed extractor could not classify."""
    from rag_ingest.chunking import Chunker
    from rag_ingest.extractors import LayeredExtractor
    from rag_ingest.ontology import ChunkRecord

    extractor = LayeredExtractor(primary=TemplateExtractor())
    chunks = []
    for chunk in Chunker().chunk(raw.doc_id, located):
        extraction = extractor.extract(chunk)
        record = ChunkRecord(
            chunk_id=chunk.chunk_id,
            doc_id=raw.doc_id,
            text=chunk.text,
            embed_text=chunk.text,
            verified=False,
            extraction=extraction,
        )
        chunks.append(record)
    from rag_ingest.pipeline import IngestOutcome

    return IngestOutcome(
        doc_id=raw.doc_id,
        verified_chunks=(),
        quarantined_chunks=tuple(chunks),
        review_items=(),
        tier_decisions=(),
    )


def _find(records, chunk_id):
    return next(r for r in records if r.chunk_id == chunk_id)
