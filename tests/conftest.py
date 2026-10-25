from __future__ import annotations

import pytest

from rag_ingest.docmodel import (
    Element,
    ElementKind,
    ElementScores,
    RawDocument,
)


def heading(text: str, level: int = 1, page: int = 1, element_id: str = "h") -> Element:
    return Element(
        element_id=element_id,
        kind=ElementKind.HEADING,
        page=page,
        text=text,
        level=level,
    )


def paragraph(text: str, page: int = 1, element_id: str = "p") -> Element:
    return Element(element_id=element_id, kind=ElementKind.PARAGRAPH, page=page, text=text)


def table(
    rows: tuple[tuple[str, ...], ...], page: int = 1, element_id: str = "t"
) -> Element:
    return Element(
        element_id=element_id,
        kind=ElementKind.TABLE,
        page=page,
        text="",
        table_rows=rows,
    )


def reference(text: str, page: int = 1, element_id: str = "r") -> Element:
    return Element(element_id=element_id, kind=ElementKind.REFERENCE, page=page, text=text)


def caption(text: str, page: int = 1, element_id: str = "c") -> Element:
    return Element(
        element_id=element_id, kind=ElementKind.FIGURE_CAPTION, page=page, text=text
    )


@pytest.fixture
def located():
    """Build a raw document plus elements with resolved char offsets."""

    def _build(doc_id: str, elements: list[Element]):
        return RawDocument.from_elements(doc_id, elements)

    return _build


def scores(
    layout: float | None = None, ocr: float | None = None, parse: float | None = None
) -> ElementScores:
    return ElementScores(layout_score=layout, ocr_score=ocr, parse_score=parse)
