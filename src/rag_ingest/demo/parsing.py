"""Parser selection for the demo server.

Mirrors the CLI's ``--parser auto`` so the demo and the batch runner disagree
about nothing: Docling when importable, pdfplumber otherwise.
"""

from __future__ import annotations

from pathlib import Path

from rag_ingest.docmodel import Element, RawDocument


def parse_pdf(
    path: Path, *, doc_id: str, parser: str = "auto",
    pages: tuple[int, int] | None = None,
) -> tuple[RawDocument, list[Element], str]:
    """Parse a PDF, returning the raw document, elements, and which parser ran.

    ``pages`` is an inclusive 1-based range so a large document is ingested
    as slices; ``None`` parses everything. Page numbers inside the parsed
    output are the PDF's own, never renumbered.
    """
    choice = parser.lower()

    if choice in ("auto", "docling"):
        try:
            from rag_ingest.adapters.docling import pdf_to_raw as docling_to_raw
        except ImportError:
            if choice == "docling":
                raise RuntimeError("docling is not installed; try parser=pdfplumber") from None
        else:
            raw, elements = docling_to_raw(str(path), doc_id=doc_id)
            return raw, elements, "docling"

    from rag_ingest.adapters.pdfplumber import pdf_to_raw as plumber_to_raw

    raw, elements = plumber_to_raw(str(path), doc_id=doc_id, pages=pages)
    return raw, elements, "pdfplumber"


__all__ = ["parse_pdf"]
