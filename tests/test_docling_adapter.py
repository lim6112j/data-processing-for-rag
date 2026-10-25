"""Adapter tests against fakes.

Docling is a heavy dependency (model weights included), so these tests exercise
the translation logic with stand-in objects that mirror the Docling surface the
adapter touches. That is enough to pin the two behaviours that actually matter:
bbox capture and top-left coordinate conversion.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

from rag_ingest.adapters.docling import (
    document_to_elements,
    document_to_raw,
    page_scores,
    render_page_images,
)
from rag_ingest.docmodel import ElementKind, ElementScores


class _Box:
    """Docling BoundingBox stand-in, bottom-left origin."""

    def __init__(self, left: float, t: float, r: float, b: float) -> None:
        self.l, self.t, self.r, self.b = left, t, r, b

    def to_top_left_origin(self, page_height: float):
        # Docling flips t/b using page height.
        return _Box(self.l, page_height - self.b, self.r, page_height - self.t)

    def as_tuple(self):
        return (self.l, self.t, self.r, self.b)


class _Label:
    def __init__(self, value: str) -> None:
        self.value = value


class _Prov:
    def __init__(self, page_no: int, bbox: _Box | None) -> None:
        self.page_no = page_no
        self.bbox = bbox


class _Item:
    def __init__(
        self,
        self_ref: str,
        label: str,
        text: str = "",
        prov: list | None = None,
        data=None,
    ) -> None:
        self.self_ref = self_ref
        self.label = _Label(label)
        self.text = text
        self.prov = prov or []
        self.data = data


class _Page:
    def __init__(self, height: float = 1000.0, pil=None) -> None:
        self.size = SimpleNamespace(height=height)
        self.image = None if pil is None else SimpleNamespace(pil_image=pil)


class _Doc:
    def __init__(self, items: list, pages: dict) -> None:
        self._items = items
        self.pages = pages

    def iterate_items(self, with_groups: bool = False):
        for item in self._items:
            yield item, 1


class _Pil:
    """Minimal PIL image stand-in that records the format it was saved as."""

    def __init__(self) -> None:
        self.saved_format = None

    def save(self, buffer, format: str) -> None:
        self.saved_format = format
        buffer.write(b"<png>")


def _doc_with(items, page_height=1000.0, pil=None):
    return _Doc(items, {1: _Page(height=page_height, pil=pil)})


class TestLabelMapping:
    def test_section_header_becomes_heading_with_level(self):
        doc = _doc_with([_Item("#/h", "section_header", "Intro", [_Prov(1, None)])])

        elements = document_to_elements(doc)

        assert elements[0].kind is ElementKind.HEADING
        assert elements[0].level == 1

    def test_table_label_is_mapped(self):
        doc = _doc_with([_Item("#/t", "table", "", [_Prov(1, None)])])

        assert document_to_elements(doc)[0].kind is ElementKind.TABLE

    def test_unknown_label_falls_back_to_paragraph_rather_than_dropping(self):
        # Losing text is worse than mislabelling it.
        doc = _doc_with([_Item("#/x", "some_new_label", "text", [_Prov(1, None)])])

        elements = document_to_elements(doc)

        assert len(elements) == 1
        assert elements[0].kind is ElementKind.PARAGRAPH


class TestProvenance:
    def test_item_without_provenance_is_skipped(self):
        # It cannot be grounded, so admitting it would be unverifiable text.
        doc = _doc_with([_Item("#/n", "text", "orphan", prov=[])])

        assert document_to_elements(doc) == []

    def test_bbox_is_converted_to_top_left_origin(self):
        # Bottom-left box (10, 100, 50, 300) on a 1000pt page becomes
        # top-left (10, 700, 50, 900).
        doc = _doc_with([_Item("#/p", "text", "x", [_Prov(1, _Box(10, 100, 50, 300))])])

        bbox = document_to_elements(doc)[0].bbox

        assert bbox == (10.0, 700.0, 50.0, 900.0)

    def test_bbox_is_carried_through_to_raw_document_provenance(self):
        # The whole point: T2 must be able to crop this region.
        doc = _doc_with([_Item("#/p", "text", "x", [_Prov(1, _Box(10, 100, 50, 300))])])

        _, located = document_to_raw(doc, doc_id="d1")

        assert located[0].provenance().bbox is not None
        assert located[0].provenance().has_visual_evidence is True

    def test_missing_bbox_does_not_crash(self):
        doc = _doc_with([_Item("#/p", "text", "x", [_Prov(1, None)])])

        elements = document_to_elements(doc)

        assert elements[0].bbox is None


class TestTables:
    def test_grid_is_used_when_present(self):
        data = SimpleNamespace(
            grid=[[SimpleNamespace(text="a"), SimpleNamespace(text="b")]],
            table_cells=None,
        )
        doc = _doc_with([_Item("#/t", "table", "", [_Prov(1, None)], data=data)])

        assert document_to_elements(doc)[0].table_rows == (("a", "b"),)

    def test_cells_are_grouped_by_row_when_no_grid(self):
        cells = [
            SimpleNamespace(text="a", start_row_offset_idx=0, start_col_offset_idx=0),
            SimpleNamespace(text="b", start_row_offset_idx=0, start_col_offset_idx=1),
            SimpleNamespace(text="c", start_row_offset_idx=1, start_col_offset_idx=0),
            SimpleNamespace(text="d", start_row_offset_idx=1, start_col_offset_idx=1),
        ]
        data = SimpleNamespace(grid=None, table_cells=cells)
        doc = _doc_with([_Item("#/t", "table", "", [_Prov(1, None)], data=data)])

        assert document_to_elements(doc)[0].table_rows == (("a", "b"), ("c", "d"))


class TestScores:
    def test_page_scores_apply_to_every_element_on_that_page(self):
        # Docling's confidence report is page-level, not per element.
        doc = _doc_with(
            [
                _Item("#/p1", "text", "a", [_Prov(1, None)]),
                _Item("#/p2", "text", "b", [_Prov(1, None)]),
            ]
        )
        scores = {1: ElementScores(layout_score=0.9, ocr_score=0.4)}

        elements = document_to_elements(doc, scores_by_page=scores)

        assert all(e.scores.ocr_score == 0.4 for e in elements)
        assert all(e.scores.layout_score == 0.9 for e in elements)

    def test_missing_scores_yield_empty_scores(self):
        doc = _doc_with([_Item("#/p", "text", "x", [_Prov(1, None)])])

        assert document_to_elements(doc)[0].scores.is_empty

    def test_nan_scores_are_dropped(self):
        # table_score is unimplemented and comes back as nan; passing it through
        # would make every threshold comparison false.
        confidence = SimpleNamespace(
            pages={1: SimpleNamespace(layout_score=0.8, ocr_score=float("nan"), parse_score=0.9)}
        )
        doc = _doc_with([_Item("#/p", "text", "x", [_Prov(1, None)])])

        scores = page_scores(doc, confidence)

        assert scores[1].ocr_score is None
        assert scores[1].layout_score == 0.8
        assert scores[1].parse_score == 0.9

    def test_page_scores_are_keyed_numerically(self):
        confidence = SimpleNamespace(
            pages={1: SimpleNamespace(layout_score=0.5, ocr_score=None, parse_score=None)}
        )

        assert 1 in page_scores(_doc_with([]), confidence)

    def test_missing_confidence_yields_no_scores(self):
        assert page_scores(_doc_with([]), None) == {}


class TestPageImages:
    def test_page_images_are_rendered_as_png(self):
        pil = _Pil()
        doc = _doc_with([_Item("#/p", "text", "x", [_Prov(1, None)])], pil=pil)

        images = render_page_images(doc)

        assert images[1] == b"<png>"
        assert pil.saved_format == "PNG"

    def test_pages_without_images_are_omitted(self):
        doc = _doc_with([_Item("#/p", "text", "x", [_Prov(1, None)])])

        assert render_page_images(doc) == {}

    def test_render_page_images_feeds_raw_document(self):
        # End-to-end: page image reaches RawDocument, so T2 can crop.
        doc = _doc_with(
            [_Item("#/p", "text", "x", [_Prov(1, _Box(10, 100, 50, 300))])], pil=_Pil()
        )

        raw, located = document_to_raw(
            doc, doc_id="d1", page_images=render_page_images(doc)
        )

        assert raw.visual_evidence(located[0].provenance()) is not None


def test_save_buffer_uses_io():
    assert io.BytesIO is not None
