"""T0 extraction, and the coordinate translation that makes it groundable.

The translation is the load-bearing part. A regex match found in ``Chunk.text``
says nothing about the page buffer; if the offsets are wrong the gate rejects
every value and the extractor looks broken for the wrong reason.
"""

from __future__ import annotations

import pytest

from rag_ingest.chunking import Chunker
from rag_ingest.docmodel import Element, ElementKind, RawDocument
from rag_ingest.extractors import TemplateExtractor
from rag_ingest.grounding import GroundingGate, GroundingStatus
from rag_ingest.ontology import (
    AbstainReason,
    FieldAbstention,
    FieldValue,
    InvoiceHeader,
    LineItem,
    Unclassified,
)

BBOX = (10.0, 20.0, 400.0, 60.0)


def _chunk(text: str, bbox=BBOX):
    raw, elements = RawDocument.from_elements(
        "d1",
        [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1, text=text, bbox=bbox)],
    )
    return raw, Chunker().chunk("d1", elements)[0]


class TestInvoiceExtraction:
    def test_identifier_and_amount_are_found(self):
        _, chunk = _chunk("Invoice INV-2041 total $1,234.56")

        extraction = TemplateExtractor().extract(chunk)

        assert isinstance(extraction, InvoiceHeader)
        assert isinstance(extraction.invoice_number, FieldValue)
        assert extraction.invoice_number.value == "INV-2041"
        assert extraction.total_amount.value == "1,234.56"

    def test_labelled_amount_wins_over_a_stray_number(self):
        _, chunk = _chunk("Page 3 of 12. Total: 99.50 KRW")

        extraction = TemplateExtractor().extract(chunk)

        assert extraction.total_amount.value == "99.50"

    def test_date_is_extracted_when_present(self):
        _, chunk = _chunk("Invoice INV-1 dated 2026-03-14 total $10.00")

        extraction = TemplateExtractor().extract(chunk)

        assert extraction.issue_date.value == "2026-03-14"

    def test_missing_fields_abstain_rather_than_invent(self):
        _, chunk = _chunk("Invoice INV-2041")

        extraction = TemplateExtractor().extract(chunk)

        assert isinstance(extraction.issue_date, FieldAbstention)
        assert extraction.issue_date.reason is AbstainReason.NOT_PRESENT
        assert isinstance(extraction.total_amount, FieldAbstention)

    def test_bare_integer_is_not_treated_as_an_amount(self):
        # "1200" without a currency mark is ambiguous; a wrong amount is worse
        # than a missing one.
        _, chunk = _chunk("Invoice INV-2041 line item quantity 1200")

        extraction = TemplateExtractor().extract(chunk)

        assert isinstance(extraction.total_amount, FieldAbstention)


class TestCoordinateTranslation:
    def test_extracted_value_grounds_against_the_raw_page(self):
        # The whole point: a match in the chunk must resolve back to the page.
        raw, chunk = _chunk("Invoice INV-2041 total $1,234.56")

        extraction = TemplateExtractor().extract(chunk)
        result = GroundingGate(raw).check(extraction)

        assert result.status is GroundingStatus.GROUNDED

    def test_grounding_actually_rejects_a_tampered_value(self):
        # Proves the pass above is not vacuous -- same provenance, wrong value.
        raw, chunk = _chunk("Invoice INV-2041 total $1,234.56")
        extraction = TemplateExtractor().extract(chunk)
        tampered = extraction.model_copy(
            update={
                "total_amount": extraction.total_amount.model_copy(
                    update={"value": "999.99"}
                )
            }
        )

        assert GroundingGate(raw).check(tampered).status is GroundingStatus.MISMATCH

    def test_offsets_survive_a_multi_element_chunk(self):
        # Two elements joined with "\n\n" shift every offset after the first.
        raw, elements = RawDocument.from_elements(
            "d1",
            [
                Element(element_id="a", kind=ElementKind.HEADING, page=1, text="Header", level=1),
                Element(
                    element_id="b",
                    kind=ElementKind.PARAGRAPH,
                    page=1,
                    text="Invoice INV-7777 total $5.00",
                    bbox=BBOX,
                ),
            ],
        )
        chunk = Chunker(target_tokens=1000).chunk("d1", elements)[0]

        extraction = TemplateExtractor().extract(chunk)

        assert extraction.invoice_number.value == "INV-7777"
        assert GroundingGate(raw).check(extraction).status is GroundingStatus.GROUNDED

    def test_table_chunk_offsets_resolve(self):
        raw, elements = RawDocument.from_elements(
            "d1",
            [
                Element(
                    element_id="t1",
                    kind=ElementKind.TABLE,
                    page=1,
                    text="",
                    table_rows=(("Item", "Amount"), ("Widget", "$1,200.00")),
                    bbox=BBOX,
                )
            ],
        )
        chunk = Chunker().chunk("d1", elements)[0]

        extraction = TemplateExtractor().extract(chunk)

        assert extraction.total_amount.value == "1,200.00"
        assert GroundingGate(raw).check(extraction).status is GroundingStatus.GROUNDED

    def test_match_outside_every_segment_yields_no_value(self):
        _, chunk = _chunk("Invoice INV-1 total $10.00")
        # A chunk with its segments stripped cannot locate anything.
        stripped = chunk.model_copy(update={"segments": ()})

        extraction = TemplateExtractor().extract(stripped)

        # No identifier/amount resolved -> falls back to unclassified.
        assert isinstance(extraction, Unclassified)


class TestUnclassified:
    def test_prose_without_invoice_markers_is_unclassified(self):
        _, chunk = _chunk("A haiku about procurement and the long winter")

        extraction = TemplateExtractor().extract(chunk)

        assert isinstance(extraction, Unclassified)
        assert extraction.observed_shape == "prose"

    def test_a_multi_line_paragraph_is_not_reported_as_tabular(self):
        # Newline count is not evidence of a table. On the FAA manual this
        # heuristic labelled 746 of 754 chunks tabular, because almost every
        # text chunk spans more than three lines.
        _, chunk = _chunk(
            "The castle nut should be inspected for cracks.\n"
            "Replace any nut whose threads show damage.\n"
            "Torque values are given in table 7-1."
        )

        extraction = TemplateExtractor().extract(chunk)

        assert isinstance(extraction, Unclassified)
        assert extraction.observed_shape == "prose"

    def test_unclassified_carries_a_resolvable_span(self):
        raw, chunk = _chunk("A haiku about procurement")

        extraction = TemplateExtractor().extract(chunk)

        assert raw.span_text(extraction.provenance) is not None

    def test_invoice_that_parses_nothing_is_still_unclassified(self):
        _, chunk = _chunk("INVOICE")

        extraction = TemplateExtractor().extract(chunk)

        assert isinstance(extraction, Unclassified)
        assert "unparseable" in extraction.observed_shape

    def test_empty_chunk_yields_none(self):
        _, chunk = _chunk("Invoice INV-1 $10.00")
        blank = chunk.model_copy(update={"text": "   "})

        assert TemplateExtractor().extract(blank) is None


class TestAnIdentifierAloneIsNotARecord:
    """A part number is shaped exactly like an invoice number.

    Measured on FAA AC 43.13-1B (see ``corpus/README.md``): ``_IDENTIFIER``
    matched aviation hardware designations (AN310, MS20365, NAS1223) and every
    one of them became an ``InvoiceHeader``. Grounding cannot reject those,
    because the token *is* verbatim in the source -- it verifies presence, not
    semantic role. So the corroboration has to happen here, before a record is
    opened.
    """

    def test_part_number_alone_is_unclassified_not_an_invoice(self):
        _, chunk = _chunk(
            "7-66. CASTLE NUT (AN310). The castle nut should be inspected for cracks."
        )

        extraction = TemplateExtractor().extract(chunk)

        assert isinstance(extraction, Unclassified)

    def test_standard_reference_alone_is_unclassified(self):
        _, chunk = _chunk("Fasteners shall conform to NAS1223-1235 and MS20365 as applicable.")

        extraction = TemplateExtractor().extract(chunk)

        assert isinstance(extraction, Unclassified)

    def test_monetary_evidence_alone_corroborates_a_record(self):
        # No "invoice" word, but an amount is monetary evidence; a part number
        # never is. This is the discriminating signal.
        _, chunk = _chunk("Total: 99.50 KRW")

        extraction = TemplateExtractor().extract(chunk)

        assert isinstance(extraction, InvoiceHeader)

    def test_invoice_label_alone_still_opens_a_record(self):
        # Required by test_missing_fields_abstain_rather_than_invent: a labelled
        # header with an identifier but no parseable amount must still produce
        # abstentions, not vanish into unclassified. The label is the evidence.
        _, chunk = _chunk("Invoice pending, reference INV-2041")

        extraction = TemplateExtractor().extract(chunk)

        assert isinstance(extraction, InvoiceHeader)
        assert isinstance(extraction.total_amount, FieldAbstention)

    def test_part_number_beside_an_invoice_label_is_not_discarded(self):
        # Corroboration must not be over-tightened: real invoice text may cite
        # part numbers too, and the label is the decisive evidence.
        _, chunk = _chunk("Invoice INV-2041 for part AN310 total $10.00")

        extraction = TemplateExtractor().extract(chunk)

        assert isinstance(extraction, InvoiceHeader)
        assert extraction.invoice_number.value == "INV-2041"


class TestCurrencies:
    @pytest.mark.parametrize("symbol", ["₩", "$", "€", "£", "¥"])
    def test_currency_symbols_are_recognised_before_and_after(self, symbol):
        before, chunk_a = _chunk(f"Invoice INV-1 total {symbol}10.00")
        after, chunk_b = _chunk(f"Invoice INV-1 total 10.00 {symbol}")

        ext_a = TemplateExtractor().extract(chunk_a)
        ext_b = TemplateExtractor().extract(chunk_b)

        assert ext_a.total_amount.value == "10.00"
        assert ext_b.total_amount.value == "10.00"
        assert GroundingGate(before).check(ext_a).status is GroundingStatus.GROUNDED
        assert GroundingGate(after).check(ext_b).status is GroundingStatus.GROUNDED


    def test_genuinely_missing_amount_is_not_present(self):
        _, chunk = _chunk("Invoice INV-3003 for consulting services")

        extraction = TemplateExtractor().extract(chunk)

        assert extraction.total_amount.reason is AbstainReason.NOT_PRESENT

    def test_damaged_amount_prevents_verification(self):
        # The chain that matters: the field is present but unreadable, so the
        # record cannot reach the two-field threshold and is quarantined.
        raw, chunk = _chunk("Invoice INV-3003 total l,234.56 USD")

        extraction = TemplateExtractor().extract(chunk)
        result = GroundingGate(raw).check(extraction)

        assert result.status is GroundingStatus.INCOMPLETE
        assert not result.verified


class TestLineItemCollection:
    """survey.md conclusion 1: reconciliation needs rows to reconcile.

    Deliberately narrow: only an explicitly labelled line amount (a number
    beside a "line"/"item"/"subtotal"-style label, or inside a table row) is
    collected. Every currency-marked number in the chunk is NOT a line item --
    collecting those would reconcile the total against itself.
    """

    def _lines(self, text: str) -> tuple:
        _, chunk = _chunk(text)
        extraction = TemplateExtractor().extract(chunk)
        return extraction.line_items

    def test_labelled_lines_are_collected_with_spans(self):
        lines = self._lines(
            "Invoice INV-2041\n"
            "Line 1: Widget 100.00 KRW\n"
            "Line 2: Sprocket 200.00 KRW\n"
            "Total 300.00 KRW"
        )

        assert len(lines) == 2
        assert all(isinstance(line, LineItem) for line in lines)
        assert all(isinstance(line.amount, FieldValue) for line in lines)

    def test_collected_line_amounts_ground_against_the_source(self):
        # The whole point of GroundedField: a line amount without a resolvable
        # span is worthless. Verify the spans actually resolve.
        text = (
            "Invoice INV-2041\n"
            "Line 1: Widget 100.00 KRW\n"
            "Line 2: Sprocket 200.00 KRW\n"
            "Total 300.00 KRW"
        )
        raw, chunk = _chunk(text)
        extraction = TemplateExtractor().extract(chunk)

        for line in extraction.line_items:
            span_text = raw.span_text(line.amount.provenance)
            assert span_text is not None
            assert "100" in span_text or "200" in span_text

    def test_the_total_itself_is_not_collected_as_a_line(self):
        lines = self._lines(
            "Invoice INV-2041\nLine 1: Widget 100.00 KRW\nTotal 300.00 KRW"
        )

        assert len(lines) == 1
        assert lines[0].amount.value == "100.00"




class TestLayeredSkipsTables:
    """A table quoted as prose is a table whose structure was destroyed: row
    discovery and row routing both starve, and the table enters retrieval as
    if its shape were known. The prose fallback exists for prose -- a
    tabular chunk stays unclassified so it can become a question."""

    def test_unclassified_table_is_not_quoted(self):
        from rag_ingest.chunking import Chunk, ChunkKind, ChunkSegment, HeadingPath
        from rag_ingest.extractors import LayeredExtractor

        chunk = Chunk(
            chunk_id="d:t1", doc_id="d", kind=ChunkKind.TABLE,
            text="부서 | 점검항목 | 상태 | 비고",
            heading_path=HeadingPath(), page_start=1, page_end=1,
            element_ids=("t1",), estimated_tokens=10,
            segments=(ChunkSegment(
                element_id="t1", page=1, chunk_start=0, chunk_end=18,
                raw_start=0, raw_end=18,
            ),),
        )
        result = LayeredExtractor(primary=TemplateExtractor()).extract(chunk)
        assert type(result).__name__ == "Unclassified", type(result).__name__
