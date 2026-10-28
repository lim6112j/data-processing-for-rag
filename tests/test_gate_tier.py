"""Gate strictness tiers: which bar a record must clear, and that it is enforced.

The load-bearing property is not that LENIENT admits prose -- it is that LENIENT
can never be applied to a typed record. A relaxation that reaches the wrong shape
admits an unverified value into retrieval with a citation attached, silently,
which is the failure this repository exists to prevent.
"""

from __future__ import annotations

import pytest

from rag_ingest.chunking import Chunker
from rag_ingest.docmodel import Element, ElementKind, ProvenanceRef, RawDocument
from rag_ingest.escalation import ReviewQueue
from rag_ingest.extractors import LayeredExtractor, ProseExtractor, TemplateExtractor
from rag_ingest.gate_tier import MIN_FIELDS_BY_TIER, assert_tier_applies, select_gate_tier
from rag_ingest.grounding import GroundingGate, GroundingStatus
from rag_ingest.ontology import (
    AbstainReason,
    FieldAbstention,
    FieldValue,
    GateTier,
    InvoiceHeader,
    ProseKnowledge,
    ReportSection,
    Unclassified,
)
from rag_ingest.pipeline import IngestPipeline

BBOX = (10.0, 20.0, 400.0, 60.0)
SPAN = ProvenanceRef(page=1, char_start=0, char_end=8, bbox=BBOX)


def _paragraph(text: str, *, page: int = 1, element_id: str = "e1") -> Element:
    return Element(
        element_id=element_id,
        kind=ElementKind.PARAGRAPH,
        page=page,
        text=text,
        bbox=BBOX,
    )


def _raw_and_chunk(text: str):
    raw, elements = RawDocument.from_elements("d1", [_paragraph(text)])
    return raw, Chunker().chunk("d1", elements)[0]


def _prose(text: str = "text") -> ProseKnowledge:
    provenance = ProvenanceRef(page=1, char_start=0, char_end=len(text), bbox=BBOX)
    return ProseKnowledge(text=FieldValue(value=text, provenance=provenance))


def _invoice() -> InvoiceHeader:
    absent = FieldAbstention(reason=AbstainReason.NOT_PRESENT)
    return InvoiceHeader(
        invoice_number=absent, issue_date=absent, total_amount=absent, currency=absent
    )


def _report() -> ReportSection:
    absent = FieldAbstention(reason=AbstainReason.NOT_PRESENT)
    return ReportSection(section_title=absent, primary_finding=absent)


class TestTierSelection:
    def test_invoice_is_strict(self):
        assert select_gate_tier(_invoice()) is GateTier.STRICT

    def test_report_section_is_standard(self):
        assert select_gate_tier(_report()) is GateTier.STANDARD

    def test_prose_is_lenient(self):
        assert select_gate_tier(_prose()) is GateTier.LENIENT

    def test_unclassified_is_never_lenient(self):
        # It verifies nothing either way; if a future change ever let one through,
        # it should meet the hardest bar.
        unclassified = Unclassified(
            observed_shape="prose",
            provenance=ProvenanceRef(page=1, char_start=0, char_end=4),
        )

        assert select_gate_tier(unclassified) is GateTier.STRICT

    def test_lenient_bar_admits_a_single_field(self):
        assert MIN_FIELDS_BY_TIER[GateTier.LENIENT] == 1

    def test_adding_a_tier_does_not_lower_the_existing_bars(self):
        assert MIN_FIELDS_BY_TIER[GateTier.STRICT] == 2
        assert MIN_FIELDS_BY_TIER[GateTier.STANDARD] == 2


class TestLenientCannotReachATypedRecord:
    """The invariant, enforced rather than documented."""

    @pytest.mark.parametrize(
        "extraction",
        [
            _invoice(),
            _report(),
            Unclassified(observed_shape="prose", provenance=SPAN),
        ],
        ids=["invoice", "report", "unclassified"],
    )
    def test_raises_rather_than_silently_relaxing(self, extraction):
        with pytest.raises(ValueError, match="LENIENT gate cannot be applied"):
            assert_tier_applies(GateTier.LENIENT, extraction)

    def test_allows_prose(self):
        assert_tier_applies(GateTier.LENIENT, _prose())

    def test_the_gate_itself_refuses(self):
        # Not only the helper: a caller that reaches the gate directly must also be
        # unable to relax an invoice.
        raw, chunk = _raw_and_chunk("Invoice INV-1 total $10.00")
        invoice = TemplateExtractor().extract(chunk)

        with pytest.raises(ValueError, match="LENIENT gate cannot be applied"):
            GroundingGate(raw).check(invoice, tier=GateTier.LENIENT)


class TestProseGroundingIsARealCheck:
    def test_verbatim_prose_grounds(self):
        text = "The castle nut should be inspected for cracks before reuse."
        raw, chunk = _raw_and_chunk(text)

        result = GroundingGate(raw).check(ProseExtractor().extract(chunk))

        assert result.status is GroundingStatus.GROUNDED
        assert result.verified

    def test_altered_prose_does_not_ground(self):
        # The check must not be a tautology. If the quoted text was edited -- by a
        # model, a cleaner, or a bug -- the gate has to see it.
        text = "The castle nut should be inspected for cracks before reuse."
        raw, chunk = _raw_and_chunk(text)
        quoted = ProseExtractor().extract(chunk)
        tampered = quoted.model_copy(
            update={
                "text": FieldValue(
                    value="The castle nut may be reused without inspection.",
                    provenance=quoted.text.provenance,
                )
            }
        )

        result = GroundingGate(raw).check(tampered)

        assert result.status is GroundingStatus.MISMATCH
        assert not result.verified

    def test_prose_with_an_unresolvable_span_is_unverifiable(self):
        text = "Some prose that exists in the source."
        raw, _ = _raw_and_chunk(text)
        broken = ProseKnowledge(
            text=FieldValue(
                value=text,
                provenance=ProvenanceRef(page=1, char_start=0, char_end=len(text) + 500),
            )
        )

        result = GroundingGate(raw).check(broken)

        assert result.status is GroundingStatus.UNVERIFIABLE
        assert not result.verified

    def test_a_single_field_is_enough_for_lenient_but_not_for_strict(self):
        # LENIENT exists because a required-evidence rule of two rejects all prose.
        # Same gate, same one-field extraction, only the bar differs -- and the bar
        # is honoured rather than ignored, so no caller gets a relaxed verdict by
        # asking for a strict one.
        text = "A single sentence of usable knowledge."
        raw, _ = _raw_and_chunk(text)
        gate = GroundingGate(raw)

        assert gate.check(_prose(text)).status is GroundingStatus.GROUNDED
        assert (
            gate.check(_prose(text), tier=GateTier.STRICT).status
            is GroundingStatus.INCOMPLETE
        )


class TestProseExtractorScope:
    def test_single_page_chunk_is_quoted_verbatim(self):
        raw, chunk = _raw_and_chunk("Narrative content about fasteners.")

        extraction = ProseExtractor().extract(chunk)

        assert isinstance(extraction, ProseKnowledge)
        assert extraction.text.value == "Narrative content about fasteners."
        assert raw.span_text(extraction.text.provenance) is not None

    def test_multi_page_chunk_is_quoted_through_per_page_spans(self):
        # A multi-page quote cites one span per page and the gate verifies the
        # ordered join, so quoting no longer has to refuse at a page boundary
        # (previously 32/754 chunks on the FAA manual).
        _, elements = RawDocument.from_elements(
            "d1",
            [
                _paragraph("first page", page=1, element_id="a"),
                _paragraph("second page", page=2, element_id="b"),
            ],
        )
        chunk = Chunker(target_tokens=10_000).chunk("d1", elements)[0]
        assert chunk.page_start != chunk.page_end

        extraction = ProseExtractor().extract(chunk)

        assert isinstance(extraction, ProseKnowledge)
        assert extraction.text.multi_page_spans is not None
        assert len(extraction.text.multi_page_spans) == 2

    def test_empty_chunk_yields_none(self):
        _, chunk = _raw_and_chunk("something")

        assert ProseExtractor().extract(chunk.model_copy(update={"text": "   "})) is None


class TestLayering:
    def test_invoice_keeps_its_typed_extraction(self):
        _, chunk = _raw_and_chunk("Invoice INV-2041 total $1,234.56")

        assert isinstance(LayeredExtractor().extract(chunk), InvoiceHeader)

    def test_prose_falls_back_to_a_verbatim_quote(self):
        _, chunk = _raw_and_chunk(
            "Bolts should be inspected for corrosion before installation."
        )

        assert isinstance(LayeredExtractor().extract(chunk), ProseKnowledge)

    def test_unparseable_invoice_header_is_not_buried_under_prose(self):
        # The reviewer must still be asked what the document should look like; a
        # verbatim quote would hide a template that no longer matches.
        _, chunk = _raw_and_chunk("INVOICE")

        extraction = LayeredExtractor().extract(chunk)

        assert isinstance(extraction, Unclassified)
        assert extraction.observed_shape == "invoice_header_unparseable"

    def test_part_numbers_are_admitted_only_as_quoted_prose(self):
        # Regression for the corpus finding: the token is grounded, but it is not
        # an invoice. It may enter as a verbatim quote, never as a typed record.
        _, chunk = _raw_and_chunk(
            "7-66. CASTLE NUT (AN310). The castle nut is inspected for cracks."
        )

        assert isinstance(LayeredExtractor().extract(chunk), ProseKnowledge)


class TestEndToEnd:
    def _ingest(self, text: str):
        raw, elements = RawDocument.from_elements("d1", [_paragraph(text)])
        return IngestPipeline(ReviewQueue(), LayeredExtractor()).ingest(
            raw, elements, owner_id="o"
        )

    def test_prose_document_yields_a_retrievable_chunk(self):
        # Before the tier existed this was 0 admitted / 1 quarantined -- the
        # measurement that justified building it (corpus/README.md).
        outcome = self._ingest(
            "Castle nuts should be inspected for cracks. Replace any nut whose "
            "threads show damage."
        )

        assert len(outcome.verified_chunks) == 1
        assert outcome.quarantined_chunks == ()
        assert outcome.review_items == ()

    def test_the_tier_is_stamped_on_the_record(self):
        outcome = self._ingest("A sentence of quotable knowledge about fasteners.")

        record = outcome.verified_chunks[0]
        assert record.verified is True
        assert record.gate_tier is GateTier.LENIENT

    def test_an_invoice_is_stamped_strict(self):
        outcome = self._ingest("Invoice INV-2041 total 1,234.56 KRW")

        assert outcome.verified_chunks[0].gate_tier is GateTier.STRICT
