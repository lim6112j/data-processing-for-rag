"""pdfplumber -> docmodel adapter: the second parse stream.

This is what gives T2 a genuine disagreement to settle. Docling (layout model +
TableFormer) and pdfplumber (raw text layer + geometric line detection) fail in
*different* ways, which is exactly the property that makes the pair useful:

  * Docling reconstructs structure from a layout model, so it handles scans and
    complex layouts but can invent a grid.
  * pdfplumber reads the text layer and finds table rules geometrically, so it
    is faithful to digital text but blind to anything that is not a line.

Same-origin parsers (e.g. Docling ACCURATE vs FAST) share failure modes and
agree on the same wrong answer. Cross-origin parsers disagree where the truth
is actually uncertain.

pdfplumber bboxes already use a top-left origin, so no flip is needed here --
unlike the Docling adapter. Getting this backwards would misalign every
candidate and quietly send T2 verdicts about the wrong regions.
"""

from __future__ import annotations

from typing import Any

from rag_ingest.docmodel import (
    BBox,
    Element,
    ElementKind,
    RawDocument,
)

#: pdfplumber reports coordinates in PDF points; no scaling is applied because
#: the Docling adapter also works in points. A mismatch here would silently
#: break alignment (which is bbox-based), so both must stay in the same space.
COORDINATE_SPACE = "pdf points, top-left origin"


def _bbox_from_plumber(obj: Any) -> BBox | None:
    """Read a bbox from either shape pdfplumber exposes.

    Words and characters carry ``x0``/``top``/``x1``/``bottom``; table objects
    instead carry a single ``bbox`` tuple. Handling only the first shape yields
    ``None`` for every table, which silently disables bbox alignment -- the
    failure would look like "the parsers never disagree" rather than like a bug.
    """
    box = getattr(obj, "bbox", None)
    if isinstance(box, tuple) and len(box) == 4:
        x0, top, x1, bottom = box
        return (float(x0), float(top), float(x1), float(bottom))

    x0 = getattr(obj, "x0", None)
    top = getattr(obj, "top", None)
    x1 = getattr(obj, "x1", None)
    bottom = getattr(obj, "bottom", None)
    if None in (x0, top, x1, bottom):
        return None
    return (float(x0), float(top), float(x1), float(bottom))


def _table_rows(table: Any) -> tuple[tuple[str, ...], ...]:
    """Normalize pdfplumber's extract_tables output (list[list[str|None]])."""
    rows: list[tuple[str, ...]] = []
    for row in table or []:
        cells = tuple("" if cell is None else str(cell).strip() for cell in row)
        if any(cells):
            rows.append(cells)
    return tuple(rows)


def _text_box(words: list[dict[str, Any]]) -> BBox | None:
    """Bounding box enclosing every word on the page."""
    if not words:
        return None
    return (
        float(min(word["x0"] for word in words)),
        float(min(word["top"] for word in words)),
        float(max(word["x1"] for word in words)),
        float(max(word["bottom"] for word in words)),
    )


def page_to_elements(page: Any, page_no: int) -> list[Element]:
    """Convert one pdfplumber page into elements.

    Tables come from geometric detection; remaining text becomes one paragraph
    element carrying the page's text bounding box. That coarse granularity is
    deliberate: reimplementing layout analysis here would make the second parser
    agree with the first for the wrong reason, and the point is independent
    *failure*, not equal finesse.
    """
    elements: list[Element] = []
    tables = page.extract_tables() or []
    finder = page.find_tables() if tables else None

    for index, table in enumerate(tables):
        rows = _table_rows(table)
        if not rows:
            continue
        # A one-column "table" is not a table: pdfplumber's box detector
        # fences plain text lines (FAA safety-tip lists do this), and there
        # is no column structure to preserve. Routing it as rows starves
        # discovery with junk one-cell clusters and steals the block from
        # the prose path that can quote it. Let the paragraph carry it.
        if max(len(row) for row in rows) < 2:
            continue
        box = _bbox_from_plumber(finder[index]) if finder else None
        elements.append(
            Element(
                element_id=f"p{page_no}-tbl{index}",
                kind=ElementKind.TABLE,
                page=page_no,
                text="",
                table_rows=rows,
                bbox=box,
            )
        )

    text = (page.extract_text() or "").strip()
    if text:
        elements.append(
            Element(
                element_id=f"p{page_no}-text",
                kind=ElementKind.PARAGRAPH,
                page=page_no,
                text=text,
                bbox=_text_box(page.extract_words()),
            )
        )

    return elements


def pdf_to_raw(
    path: str,
    *,
    doc_id: str,
    render_images: bool = True,
    image_resolution: int = 144,
    pages: tuple[int, int] | None = None,
) -> tuple[RawDocument, list[Element]]:
    """Parse a PDF with pdfplumber into the docmodel contract.

    ``pages`` selects an inclusive 1-based page range ``(start, end)`` so a
    600-page manual is ingested as slices. Page numbers are NOT renumbered:
    a slice of pages 101-110 produces elements tagged 101-110, so provenance
    spans still point at the physical document the reader holds. A range
    extending past the last page clamps there -- slices are composed by a
    caller that does not know the page count in advance.

    ``render_images`` produces page PNGs so T2 can crop evidence. Keep it on
    unless memory is tight: without images T2 abstains on every question.
    Imported lazily so the package works without pdfplumber installed.
    """
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("pdfplumber adapter needs 'pdfplumber': uv add pdfplumber") from exc

    elements: list[Element] = []
    page_images: dict[int, bytes] = {}
    page_sizes: dict[int, tuple[float, float]] = {}

    with pdfplumber.open(path) as pdf:
        selected = _page_slice(pdf.pages, pages)
        for page_no, page in selected:
            elements.extend(page_to_elements(page, page_no))
            page_sizes[page_no] = (float(page.width), float(page.height))
            if render_images:
                png = _render_page(page, resolution=image_resolution)
                if png is not None:
                    page_images[page_no] = png

    return RawDocument.from_elements(
        doc_id, elements, page_images=page_images, page_sizes=page_sizes
    )


def page_count(path: str) -> int:
    """The document's page count -- cheap metadata, not a parse.

    Auto-slicing needs this before choosing boundaries: counting pages opens
    the PDF's cross-reference table, not its content streams.
    """
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("pdfplumber adapter needs 'pdfplumber': uv add pdfplumber") from exc
    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)


def iter_pdf_slices(
    path: str,
    *,
    doc_id: str,
    batch: int,
    render_images: bool = True,
    image_resolution: int = 144,
):
    """Yield ``(start, end, raw, elements)`` for consecutive page batches.

    Auto-slicing used to re-open the PDF and re-pay per-call warmup for
    every slice -- 33 slices over a 646-page manual spent more time in
    overhead than in pages. One open, one walk, a flush at every batch
    boundary: total parse cost is the pages themselves, and memory holds one
    batch's elements and images, never the whole document.
    """
    if batch < 1:
        raise ValueError("batch must be >= 1")
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("pdfplumber adapter needs 'pdfplumber': uv add pdfplumber") from exc

    with pdfplumber.open(path) as pdf:
        elements: list[Element] = []
        page_images: dict[int, bytes] = {}
        page_sizes: dict[int, tuple[float, float]] = {}
        batch_start = 1

        def flush(batch_end: int):
            raw, located = RawDocument.from_elements(
                f"{doc_id}", elements, page_images=page_images, page_sizes=page_sizes
            )
            return batch_start, batch_end, raw, located

        for page_no, page in enumerate(pdf.pages, start=1):
            elements.extend(page_to_elements(page, page_no))
            page_sizes[page_no] = (float(page.width), float(page.height))
            if render_images:
                png = _render_page(page, resolution=image_resolution)
                if png is not None:
                    page_images[page_no] = png
            if page_no % batch == 0 or page_no == len(pdf.pages):
                yield flush(page_no)
                elements = []
                page_images = {}
                page_sizes = {}
                batch_start = page_no + 1


def _page_slice(
    all_pages: list[Any],
    pages: tuple[int, int] | None,
) -> list[tuple[int, Any]]:
    """Resolve a requested inclusive page range to (page_no, page) pairs.

    ``None`` means every page. The lower bound is clamped up from below 1 and
    the upper bound clamped down to the document's end: callers building
    slices with overlap cannot know the page count beforehand, and a range
    past the end is a request for "the rest", not an error.
    """
    enumerated = list(enumerate(all_pages, start=1))
    if pages is None:
        return enumerated
    start, end = pages
    start = max(start, 1)
    end = min(end, len(enumerated))
    if start > end:
        return []
    return enumerated[start - 1 : end]


def _render_page(page: Any, *, resolution: int) -> bytes | None:
    """Render a page to PNG bytes, or None when rendering is unavailable."""
    try:
        image = page.to_image(resolution=resolution).original
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:  # rendering is optional, never fatal
        return None


def second_parse_from_pdf(path: str, *, name: str = "pdfplumber") -> Any:
    """Convenience: build a ``SecondParse`` for the alternate-parse producer."""
    from rag_ingest.candidates import SecondParse

    _, elements = pdf_to_raw(path, doc_id="alternate", render_images=False)
    return SecondParse(name=name, elements=tuple(elements))


__all__ = [
    "COORDINATE_SPACE",
    "page_to_elements",
    "pdf_to_raw",
    "second_parse_from_pdf",
]
