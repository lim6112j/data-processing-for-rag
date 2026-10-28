"""End-to-end: one document through T0 -> T1 -> quarantine.

These tests use a stub extractor, which is the point: tiers 0-1 work with no
model in the loop (plan section 2).
"""

from __future__ import annotations

from datetime import date

from rag_ingest.docmodel import Element, ElementKind, ProvenanceRef, RawDocument
from rag_ingest.escalation import EscalationReason, ReviewQueue
from rag_ingest.ontology import (
    AbstainReason,
    FieldAbstention,
    FieldValue,
    InvoiceHeader,
    Unclassified,
)
from rag_ingest.pipeline import IngestPipeline
from tests.conftest import heading, paragraph, table

TODAY = date(2026, 10, 25)


class StubExtractor:
    """Returns a canned extraction per chunk heading, chosen by the test."""

    def __init__(self, by_heading: dict[str, object], ref_by_heading: dict[str, ProvenanceRef]):
        self._by_heading = by_heading
        self._refs = ref_by_heading

    def extract(self, chunk):
        key = str(chunk.heading_path)
        template = self._by_heading.get(key)
        if template is None:
            return None
        ref = self._refs[key]
        if template == "invoice_ok":
            return InvoiceHeader(
                invoice_number=FieldValue(value="INV-9", provenance=ref),
                issue_date=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
                total_amount=FieldValue(value="1234.56", provenance=ref),
                currency=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
            )
        if template == "invoice_impossible_date":
            return InvoiceHeader(
                invoice_number=FieldValue(value="INV-9", provenance=ref),
                issue_date=FieldValue(value="2026-13-45", provenance=ref),
                total_amount=FieldValue(value="1234.56", provenance=ref),
                currency=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
            )
        if template == "invoice_lying":
            return InvoiceHeader(
                invoice_number=FieldValue(value="INV-9", provenance=ref),
                issue_date=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
                total_amount=FieldValue(value="9999.99", provenance=ref),
                currency=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
            )
        if template == "unclassified":
            return Unclassified(observed_shape="poem", provenance=ref)
        return None


def _build_doc():
    elements = [
        heading("Invoices", level=1, element_id="h-inv"),
        paragraph("Invoice INV-9 total 1234.56 KRW", element_id="p-inv"),
        heading("Notes", level=1, element_id="h-note"),
        paragraph("Unrelated narrative about procurement policy", element_id="p-note"),
    ]
    return RawDocument.from_elements("d1", elements)


def _refs(raw: RawDocument, page_text: str) -> ProvenanceRef:
    start = raw.pages[0].text.index(page_text)
    return ProvenanceRef(page=1, char_start=start, char_end=start + len(page_text))


class TestHappyPath:
    def test_grounded_chunk_is_verified_and_quarantine_is_empty(self):
        raw, elements = _build_doc()
        ref = ProvenanceRef(
            page=1,
            char_start=raw.pages[0].text.index("Invoice INV-9"),
            char_end=(
                raw.pages[0].text.index("Invoice INV-9")
                + len("Invoice INV-9 total 1234.56 KRW")
            ),
        )
        extractor = StubExtractor({"Invoices": "invoice_ok"}, {"Invoices": ref})
        queue = ReviewQueue()

        outcome = IngestPipeline(queue, extractor, today=TODAY).ingest(
            raw, elements, owner_id="owner-1"
        )

        assert len(outcome.verified_chunks) == 1
        assert outcome.verified_chunks[0].verified is True
        assert outcome.verified_chunks[0].ontology_version
        # The unextractable second section is quarantined, not dropped.
        assert all(not item for item in [c.verified for c in outcome.quarantined_chunks])


class TestSemanticChecks:
    """survey.md conclusion 1: grounding sees "in the source", not "means a

    real date". 2026-13-45 is verbatim in the span, so the gate passes it;
    only a deterministic semantic check can refuse it. The check runs AFTER
    the gate, and its failure quarantines -- an impossible date in retrieval
    is exactly the confident wrong answer this pipeline exists to prevent.
    """

    def _ingest_impossible_date(self):
        elements = [
            heading("Invoices", level=1, element_id="h-inv"),
            # The date is verbatim in the source -- grounding passes it.
            paragraph("Invoice INV-9 dated 2026-13-45 total 1234.56", element_id="p-inv"),
        ]
        raw, doc_elements = RawDocument.from_elements("d1", elements)
        ref = ProvenanceRef(
            page=1,
            char_start=raw.pages[0].text.index("Invoice INV-9"),
            char_end=(
                raw.pages[0].text.index("Invoice INV-9")
                + len("Invoice INV-9 dated 2026-13-45 total 1234.56")
            ),
        )
        extractor = StubExtractor(
            {"Invoices": "invoice_impossible_date"}, {"Invoices": ref}
        )
        queue = ReviewQueue()
        outcome = IngestPipeline(queue, extractor, today=TODAY).ingest(
            raw, doc_elements, owner_id="owner-1"
        )
        return outcome, queue

    def test_impossible_date_is_quarantined_despite_clean_grounding(self):
        outcome, _ = self._ingest_impossible_date()

        assert outcome.verified_chunks == ()
        assert len(outcome.quarantined_chunks) == 1

    def test_check_failure_is_a_specific_review_reason(self):
        outcome, _ = self._ingest_impossible_date()

        assert outcome.review_items
        item = outcome.review_items[0]
        assert item.reason is EscalationReason.CONSISTENCY_FAILED
        # The question must say what is wrong, not just that something is.
        assert "2026-13-45" in item.question

    def test_valid_dates_still_verify(self):
        raw, elements = _build_doc()
        ref = ProvenanceRef(
            page=1,
            char_start=raw.pages[0].text.index("Invoice INV-9"),
            char_end=(
                raw.pages[0].text.index("Invoice INV-9")
                + len("Invoice INV-9 total 1234.56 KRW")
            ),
        )
        extractor = StubExtractor({"Invoices": "invoice_ok"}, {"Invoices": ref})

        outcome = IngestPipeline(ReviewQueue(), extractor, today=TODAY).ingest(
            raw, elements, owner_id="owner-1"
        )

        assert len(outcome.verified_chunks) == 1


class TestTierGating:
    def test_unreadable_element_never_reaches_semantic_extraction(self):
        # Nothing should be extracted from text the parser distrusts.
        from rag_ingest.docmodel import ElementScores

        poor = Element(
            element_id="bad",
            kind=ElementKind.PARAGRAPH,
            page=1,
            text="!!!",
            scores=ElementScores(layout_score=0.9, parse_score=0.1),
        )
        raw, elements = RawDocument.from_elements("d2", [poor])

        outcome = IngestPipeline(ReviewQueue(), StubExtractor({}, {}), today=TODAY).ingest(
            raw, elements, owner_id="o"
        )

        assert outcome.verified_chunks == ()
        assert outcome.quarantined_chunks == ()
        assert outcome.review_items[0].reason is EscalationReason.EXTRACTION_FAILED


def test_table_is_chunked_whole_even_when_document_is_split():
    elements = [
        heading("Data", level=1, element_id="h"),
        paragraph("Some prose that will be its own chunk.", element_id="p1"),
        table((("a", str(i)) for i in range(30)), element_id="t1"),
    ]
    raw, located = RawDocument.from_elements("d3", elements)
    outcome = IngestPipeline(
        ReviewQueue(), StubExtractor({}, {}), chunker=None, today=TODAY
    ).ingest(raw, located, owner_id="o")

    table_chunks = [c for c in outcome.quarantined_chunks if "t1" in c.chunk_id]
    assert len(table_chunks) == 1
    assert table_chunks[0].text.count("\n") == 29
