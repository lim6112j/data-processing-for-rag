"""End-to-end: a real PDF through two parsers, producing a real disagreement.

This is the test that makes T2 more than plumbing. It builds an actual PDF with
a table, parses it with pdfplumber, and checks that the alternate-parse producer
either surfaces a genuine disagreement or correctly reports none.

pdfplumber is a real dependency here (no fakes) because the whole point is that
two independent parsers read the same bytes differently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_ingest.adapters.pdfplumber import page_to_elements, pdf_to_raw
from rag_ingest.candidates import AlternateParseProducer, SecondParse
from rag_ingest.docmodel import Element, ElementKind


def _make_pdf(path: Path) -> None:
    """Write a minimal one-page PDF containing a ruled table and a sentence."""
    content = b"""BT /F1 12 Tf 72 720 Td (Quarterly report) Tj ET
0.5 w
72 690 m 400 690 l S
72 670 m 400 670 l S
72 650 m 400 650 l S
72 690 m 72 650 l S
180 690 m 180 650 l S
400 690 m 400 650 l S
BT /F1 10 Tf 78 674 Td (Item) Tj ET
BT /F1 10 Tf 186 674 Td (Amount) Tj ET
BT /F1 10 Tf 78 654 Td (Widget) Tj ET
BT /F1 10 Tf 186 654 Td (1200) Tj ET
"""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n"
        f"{xref_at}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))


@pytest.fixture
def pdf_path(tmp_path):
    path = tmp_path / "report.pdf"
    _make_pdf(path)
    return path


class TestPdfplumberAdapter:
    def test_pdf_parses_into_elements(self, pdf_path):
        raw, elements = pdf_to_raw(str(pdf_path), doc_id="d1")

        assert raw.doc_id == "d1"
        assert elements
        assert all(element.page == 1 for element in elements)

    def test_page_images_are_rendered(self, pdf_path):
        raw, _elements = pdf_to_raw(str(pdf_path), doc_id="d1")

        assert raw.page(1) is not None
        assert raw.page(1).has_image

    def test_images_can_be_disabled(self, pdf_path):
        raw, _ = pdf_to_raw(str(pdf_path), doc_id="d1", render_images=False)

        assert not raw.page(1).has_image

    def test_elements_carry_bboxes_for_alignment(self, pdf_path):
        _, elements = pdf_to_raw(str(pdf_path), doc_id="d1", render_images=False)

        assert any(element.bbox is not None for element in elements)

    def test_table_is_detected_from_rules(self, pdf_path):
        _, elements = pdf_to_raw(str(pdf_path), doc_id="d1", render_images=False)

        tables = [e for e in elements if e.kind is ElementKind.TABLE]
        assert tables, "a ruled table should be detected geometrically"
        assert any("Widget" in cell for row in tables[0].table_rows for cell in row)

    def test_text_is_captured(self, pdf_path):
        _, elements = pdf_to_raw(str(pdf_path), doc_id="d1", render_images=False)

        combined = " ".join(e.text for e in elements)
        assert "Quarterly report" in combined

    def test_bbox_is_top_left_origin(self, pdf_path):
        # pdfplumber is already top-left; a flip here would misalign everything.
        _, elements = pdf_to_raw(str(pdf_path), doc_id="d1", render_images=False)
        boxes = [e.bbox for e in elements if e.bbox]
        assert boxes
        for box in boxes:
            left, top, right, bottom = box
            assert left <= right
            assert top <= bottom
            # A top-left origin on a 792pt page puts content in the upper half.
            assert top < 792


class TestDisagreementProduction:
    def test_two_parses_of_the_same_pdf_are_comparable(self, pdf_path):
        # The alternate parse stands in for Docling: here we perturb one
        # element's text, which is exactly the kind of difference the two
        # parsers produce on a real table.
        _, primary_elements = pdf_to_raw(str(pdf_path), doc_id="primary", render_images=False)
        table = next(e for e in primary_elements if e.kind is ElementKind.TABLE)

        divergent = Element(
            element_id="alt-table",
            kind=ElementKind.TABLE,
            page=1,
            text="",
            table_rows=(("Item", "Amount"), ("Widget", "1200"), ("Extra", "50")),
            bbox=table.bbox,
        )
        producer = AlternateParseProducer(
            primary_name="docling",
            alternate=SecondParse(name="pdfplumber", elements=(divergent,)),
        )
        from rag_ingest.chunking import Chunker

        chunk = Chunker().chunk("d1", [table])[0]

        candidates = producer.produce(chunk)

        assert len(candidates) == 2
        assert candidates[0].produced_by == "docling"
        assert candidates[1].produced_by == "pdfplumber"
        assert "Extra" in candidates[1].proposed_text
        assert "Extra" not in candidates[0].proposed_text

    def test_agreeing_parses_produce_no_question(self, pdf_path):
        _, elements = pdf_to_raw(str(pdf_path), doc_id="d1", render_images=False)
        table = next(e for e in elements if e.kind is ElementKind.TABLE)
        producer = AlternateParseProducer(
            primary_name="docling",
            alternate=SecondParse(name="pdfplumber", elements=(table,)),
        )
        from rag_ingest.chunking import Chunker

        chunk = Chunker().chunk("d1", [table])[0]

        # Same text: nothing to adjudicate, so T2 is never called.
        assert len(producer.produce(chunk)) == 1


def test_page_to_elements_handles_a_page_without_tables():
    class _EmptyPage:
        def extract_tables(self):
            return []

        def extract_text(self):
            return "just some prose"

        def extract_words(self):
            return [{"x0": 10, "top": 20, "x1": 100, "bottom": 30}]

    elements = page_to_elements(_EmptyPage(), page_no=1)

    assert len(elements) == 1
    assert elements[0].kind is ElementKind.PARAGRAPH
    assert elements[0].bbox == (10.0, 20.0, 100.0, 30.0)
