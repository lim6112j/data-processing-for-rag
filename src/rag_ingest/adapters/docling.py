"""Docling -> docmodel adapter.

Translates a ``DoclingDocument`` into ``Element``/``RawDocument``. Imports of
``docling`` are deferred so the rest of the package works without it installed;
this module is only importable-as-a-unit when Docling is present.

Two things in here matter more than the label mapping:

**bbox is not optional.** ``item.prov[0].bbox`` is the only source of visual
evidence. Without it T2 cannot crop the region it is supposed to judge, so it
must abstain on every character-identity question -- which is precisely the
question T2 exists for. Dropping the bbox here silently disables half of T2.

**Page images drive T2, not the gate.** The grounding gate re-derives values
from the *text buffer*, so a self-consistent OCR error still passes T1 (see the
README limitation). T2 sees the page image, which is what closes that gap --
but only if ``generate_page_images`` is on. Off by default in Docling, so the
converter options below turn it on.

Coordinate-space caveat: Docling bboxes use a bottom-left origin; the
``BBox`` convention in ``docmodel`` is (left, top, right, bottom) with the
origin at the top-left. ``_to_top_left`` does that conversion, and it must
happen here -- a flipped bbox crops the wrong region and T2 answers
confidently about the wrong part of the page.
"""

from __future__ import annotations

from typing import Any

from rag_ingest.docmodel import (
    BBox,
    Element,
    ElementKind,
    ElementScores,
    RawDocument,
)

#: Docling label -> our smaller kind set. Unknown labels fall back to PARAGRAPH
#: rather than being dropped: losing text is worse than mislabelling it, and
#: the tier selector will still route it.
_LABEL_MAP: dict[str, ElementKind] = {
    "title": ElementKind.HEADING,
    "section_header": ElementKind.HEADING,
    "text": ElementKind.PARAGRAPH,
    "paragraph": ElementKind.PARAGRAPH,
    "table": ElementKind.TABLE,
    "list": ElementKind.LIST,
    "list_item": ElementKind.LIST,
    "caption": ElementKind.FIGURE_CAPTION,
    "reference": ElementKind.REFERENCE,
    "footnote": ElementKind.PARAGRAPH,
    "formula": ElementKind.PARAGRAPH,
    "code": ElementKind.PARAGRAPH,
}


def _label_name(item: Any) -> str:
    label = getattr(item, "label", None)
    return getattr(label, "value", str(label or "")).lower()


def _to_top_left(bbox: Any, page_height: float | None) -> BBox | None:
    """Convert Docling's bottom-left-origin box to top-left origin.

    Docling's ``to_top_left_origin(page_height=...)`` does the flip; calling it
    requires the page height, which the caller supplies from ``page.size``.
    """
    if bbox is None:
        return None
    if page_height is not None and hasattr(bbox, "to_top_left_origin"):
        flipped = bbox.to_top_left_origin(page_height=page_height)
        return (float(flipped.l), float(flipped.t), float(flipped.r), float(flipped.b))
    return (float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b))


def _finite(value: Any) -> float | None:
    """Coerce a Docling score, treating ``nan`` as absent.

    Docling emits ``nan`` for scores it does not compute -- ``table_score`` is
    documented as not yet implemented and ``ocr_score`` is ``nan`` on pages with
    no OCR pass. Passing ``nan`` through would make every comparison false and
    every threshold decision silently wrong.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def page_scores(document: Any, confidence: Any) -> dict[int, ElementScores]:
    """Map 1-based page number -> scores.

    Docling's confidence report is **page-level** (``confidence.pages[n]``), not
    per element. Element-level attribution does not exist in the API, so the
    adapter applies the page's scores to every element on that page. That is a
    real coarseness: a single bad table drags the whole page's OCR score down,
    and the tier selector acts on the page rather than the element.
    """
    scores: dict[int, ElementScores] = {}
    pages = getattr(confidence, "pages", None) or {}
    for page_no, entry in pages.items():
        scores[int(page_no)] = ElementScores(
            layout_score=_finite(getattr(entry, "layout_score", None)),
            ocr_score=_finite(getattr(entry, "ocr_score", None)),
            parse_score=_finite(getattr(entry, "parse_score", None)),
        )
    return scores


def _table_rows(item: Any) -> tuple[tuple[str, ...], ...] | None:
    """Flatten a Docling table into a row grid.

    Uses ``data.grid`` when present (it already accounts for spans); otherwise
    falls back to ``data.table_cells`` grouping by ``start_row_offset_idx``.
    """
    data = getattr(item, "data", None)
    if data is None:
        return None

    grid = getattr(data, "grid", None)
    if grid:
        return tuple(tuple(cell.text for cell in row) for row in grid)

    cells = getattr(data, "table_cells", None)
    if not cells:
        return None
    rows: dict[int, list[tuple[int, str]]] = {}
    for cell in cells:
        index = getattr(cell, "start_row_offset_idx", 0)
        column = getattr(cell, "start_col_offset_idx", 0)
        rows.setdefault(index, []).append((column, cell.text))
    return tuple(tuple(text for _, text in sorted(row)) for _, row in sorted(rows.items()))


def document_to_elements(
    document: Any,
    *,
    scores_by_page: dict[int, ElementScores] | None = None,
) -> list[Element]:
    """Convert a ``DoclingDocument`` into located ``Element``s.

    Every item's first provenance entry supplies ``page`` and ``bbox``. Items
    without provenance are skipped -- they cannot be grounded, and admitting
    them would put unverifiable text into the pipeline.
    """
    scores = scores_by_page or {}
    elements: list[Element] = []

    for item, level in document.iterate_items(with_groups=False):
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        first = prov[0]
        page_no = int(first.page_no)
        page = document.pages.get(page_no)
        page_height = float(page.size.height) if page is not None and page.size else None

        kind = _LABEL_MAP.get(_label_name(item), ElementKind.PARAGRAPH)
        is_heading = kind is ElementKind.HEADING
        text = getattr(item, "text", "") or ""

        elements.append(
            Element(
                element_id=getattr(item, "self_ref", f"p{page_no}-{len(elements)}"),
                kind=kind,
                page=page_no,
                text=text,
                # Docling's tree level is nesting depth, not heading rank; for a
                # heading the two usually agree, and the chunker tolerates a
                # wrong rank far better than it tolerates a missing one.
                level=max(1, min(level, 6)) if is_heading else None,
                table_rows=_table_rows(item) if kind is ElementKind.TABLE else None,
                scores=scores.get(page_no, ElementScores()),
                bbox=_to_top_left(getattr(first, "bbox", None), page_height),
            )
        )

    return elements


def document_to_raw(
    document: Any,
    *,
    doc_id: str,
    confidence: Any = None,
    page_images: dict[int, bytes] | None = None,
) -> tuple[RawDocument, list[Element]]:
    """Full conversion: elements plus the raw document T1 and T2 both need.

    ``confidence`` is the conversion result's ``confidence`` report; without it
    every element has empty scores and the tier selector cannot act, which
    means no document ever reaches T2.
    ``page_images`` should come from ``render_page_images``. Omitting it is
    legal -- T1 still works, T2 loses its evidence channel.
    """
    scores = page_scores(document, confidence) if confidence is not None else {}
    elements = document_to_elements(document, scores_by_page=scores)
    return RawDocument.from_elements(doc_id, elements, page_images=page_images)


def render_page_images(document: Any) -> dict[int, bytes]:
    """Extract rendered pages as PNG bytes, keyed by 1-based page number.

    Requires the converter to have run with ``generate_page_images=True``;
    returns an empty dict otherwise, which downgrades T2 rather than crashing.
    """
    images: dict[int, bytes] = {}
    for page_no, page in document.pages.items():
        image = getattr(page, "image", None)
        pil = getattr(image, "pil_image", None)
        if pil is None:
            continue
        buffer = __import__("io").BytesIO()
        pil.save(buffer, format="PNG")
        images[int(page_no)] = buffer.getvalue()
    return images


def pdf_to_raw(
    path: str,
    *,
    doc_id: str,
    render_images: bool = True,
    images_scale: float = 2.0,
    do_table_structure: bool = True,
    do_ocr: bool = True,
    table_mode: str = "accurate",
) -> tuple[RawDocument, list[Element]]:
    """Parse a PDF with Docling into the docmodel contract.

    This is the primary parse stream: it runs a layout model and TableFormer, so
    it reconstructs structure on scans and complex layouts that a text-layer
    parser cannot see.

    ``table_mode`` selects TableFormer's mode (``"accurate"`` or ``"fast"``).
    Running the *same* parser in two modes produces a much weaker disagreement
    signal than two different parsers, because both modes share the layout
    model's failure modes -- they tend to agree on the same wrong answer.

    ``render_images`` turns on page PNGs. Without them T2 has no evidence
    channel and must abstain on every question.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableFormerMode,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("Docling adapter needs 'docling': uv add docling") from exc

    options = PdfPipelineOptions()
    options.generate_page_images = render_images
    options.images_scale = images_scale
    options.do_table_structure = do_table_structure
    options.do_ocr = do_ocr
    if do_table_structure:
        options.table_structure_options.mode = (
            TableFormerMode.ACCURATE
            if table_mode.lower() == "accurate"
            else TableFormerMode.FAST
        )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    result = converter.convert(path)
    document = result.document

    images = render_page_images(document) if render_images else None
    sizes = {
        int(page_no): (float(page.size.width), float(page.size.height))
        for page_no, page in document.pages.items()
        if getattr(page, "size", None) is not None
    }
    elements = document_to_elements(
        document,
        scores_by_page=page_scores(document, getattr(result, "confidence", None)),
    )
    return RawDocument.from_elements(
        doc_id, elements, page_images=images, page_sizes=sizes
    )


def second_parse_from_pdf(path: str, *, name: str = "pdfplumber") -> Any:
    """Build a ``SecondParse`` from pdfplumber for the alternate-parse producer.

    Cross-origin pairing is deliberate: Docling reconstructs structure with a
    layout model, pdfplumber reads the text layer and finds tables from ruled
    lines. They fail differently, which is what makes their disagreement
    informative. Two settings of one parser would mostly reproduce the same
    mistakes.
    """
    from rag_ingest.adapters.pdfplumber import pdf_to_raw as plumber_to_raw
    from rag_ingest.candidates import SecondParse

    _, elements = plumber_to_raw(path, doc_id="alternate", render_images=False)
    _ = name
    return SecondParse(name="pdfplumber", elements=tuple(elements))


__all__ = [
    "document_to_elements",
    "document_to_raw",
    "page_scores",
    "pdf_to_raw",
    "render_page_images",
    "second_parse_from_pdf",
]
