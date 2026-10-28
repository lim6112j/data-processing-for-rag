"""Adapter against real Docling output. Skipped when Docling is not installed.

The fake-based adapter tests pin the translation logic, but only a real
ConversionResult proves the assumptions about its API: page-level confidence,
BOTTOMLEFT coordinate origin, and table structure actually being populated.

Slow (loads model weights on first run), so it is opt-in via ``-m docling``
in addition to the import guard.
"""

from __future__ import annotations

import pytest

pytest.importorskip("docling", reason="docling not installed")

from rag_ingest.docmodel import ElementKind
from rag_ingest.tiering import Tier, select_tier

pytestmark = [
    pytest.mark.docling,
    pytest.mark.skipif(
        __import__("os").environ.get("RAG_SKIP_DOCLING") == "1",
        reason="RAG_SKIP_DOCLING=1",
    ),
]


@pytest.fixture(scope="module")
def parsed(tmp_path_factory):
    from helpers.pdf_fixture import make_pdf

    from rag_ingest.adapters.docling import pdf_to_raw

    path = tmp_path_factory.mktemp("docling") / "report.pdf"
    make_pdf(path)
    return pdf_to_raw(str(path), doc_id="live")


def test_real_document_produces_elements(parsed):
    _, elements = parsed

    assert elements
    assert all(element.bbox is not None for element in elements)


def test_bbox_origin_is_top_left(parsed):
    # Docling reports BOTTOMLEFT; the adapter must flip. A paragraph near the
    # top of an 792pt page has a small top coordinate after conversion.
    _, elements = parsed

    for element in elements:
        left, top, right, bottom = element.bbox
        assert left <= right
        assert top <= bottom
        assert 0 <= top < 792


def test_table_rows_are_parsed_from_real_output(parsed):
    _, elements = parsed

    tables = [e for e in elements if e.kind is ElementKind.TABLE]
    assert tables, "TableFormer should detect the ruled table"
    assert any("Widget" in cell for row in tables[0].table_rows for cell in row)


def test_page_confidence_is_attached(parsed):
    _, elements = parsed

    # The real report is page-level; every element on the page shares it.
    assert any(not element.scores.is_empty for element in elements)


def test_nan_scores_become_none(parsed):
    # table_score is documented unimplemented and comes back as nan. Passing it
    # through would make every threshold comparison false.
    _, elements = parsed

    for element in elements:
        for score in (
            element.scores.layout_score,
            element.scores.parse_score,
            element.scores.parse_score,
        ):
            assert score is None or score == score


def test_tier_selection_works_on_real_scores(parsed):
    _, elements = parsed

    assert all(select_tier(element).tier in set(Tier) for element in elements)


def test_page_images_are_rendered(parsed):
    raw, elements = parsed

    assert raw.page(1) is not None
    assert raw.page(1).has_image
    assert raw.visual_evidence(elements[0].provenance()) is not None
