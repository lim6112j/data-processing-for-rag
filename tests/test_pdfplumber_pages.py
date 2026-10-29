"""Page-range parsing: a 600-page manual is ingested as slices, not as one
monolithic parse. The page numbers must stay the PDF's own -- a slice of
pages 101-110 produces elements tagged page 101-110, never renumbered."""

import pytest

pdfplumber = pytest.importorskip("pdfplumber")


def _make_multipage_pdf(path, pages=5):
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    doc = canvas.Canvas(str(path), pagesize=LETTER)
    for i in range(1, pages + 1):
        doc.drawString(72, 700, f"PAGE-MARKER-{i}")
        doc.showPage()
    doc.save()


class TestPageRangeParsing:
    def test_full_parse_covers_every_page(self, tmp_path):
        from rag_ingest.adapters.pdfplumber import pdf_to_raw

        path = tmp_path / "doc.pdf"
        _make_multipage_pdf(path, pages=5)
        _raw, elements = pdf_to_raw(str(path), doc_id="d", render_images=False)

        marker_pages = sorted({e.page for e in elements if "PAGE-MARKER" in e.text})
        assert marker_pages == [1, 2, 3, 4, 5]

    def test_slice_keeps_original_page_numbers(self, tmp_path):
        from rag_ingest.adapters.pdfplumber import pdf_to_raw

        path = tmp_path / "doc.pdf"
        _make_multipage_pdf(path, pages=5)
        raw, elements = pdf_to_raw(str(path), doc_id="d", render_images=False,
                                   pages=(2, 3))

        marker_pages = sorted({e.page for e in elements if "PAGE-MARKER" in e.text})
        assert marker_pages == [2, 3]
        # A slice is a real document slice: page buffers exist only for the
        # requested pages, so provenance back-references stay coherent.
        assert {pg.page_no for pg in raw.pages} == {2, 3}

    def test_slice_beyond_document_end_is_safe(self, tmp_path):
        from rag_ingest.adapters.pdfplumber import pdf_to_raw

        path = tmp_path / "doc.pdf"
        _make_multipage_pdf(path, pages=5)
        raw, elements = pdf_to_raw(str(path), doc_id="d", render_images=False,
                                   pages=(4, 99))

        marker_pages = sorted({e.page for e in elements if "PAGE-MARKER" in e.text})
        assert marker_pages == [4, 5]
        assert {pg.page_no for pg in raw.pages} == {4, 5}
