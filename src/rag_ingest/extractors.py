"""T0 extraction: deterministic, provenance-producing, model-free.

Plan §2 puts the model in T2, not here. These extractors are pattern matchers:
cheap, reproducible, and auditable. Their job is to find values *and* to say
exactly where each value came from, because a value without provenance cannot
be grounded and therefore cannot be verified.

The hard part is not the regex. It is the coordinate translation: a match is
found in ``Chunk.text``, but grounding re-derives from the page buffer. The chunk
re-joins elements with "\\n\\n" and a table may be rendered with pipes, so chunk
offsets and raw offsets drift apart. ``Chunk.span_for`` carries both systems, and
every extractor here goes through it rather than computing offsets itself.

Two failure modes are deliberately preferred over guessing:

  * no match -> abstention with a reason, not a blank value;
  * match without a resolvable span -> no value at all, because an ungroundable
    value is worse than an absent one.
"""

from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from rag_ingest.chunking import Chunk
from rag_ingest.ontology import (
    AbstainReason,
    Extraction,
    FieldAbstention,
    FieldValue,
    InvoiceHeader,
    Unclassified,
)

#: Amounts, tolerating thousands separators and currency symbols.
#: Deliberately not matching a bare integer: "1200" in a table cell is ambiguous
#: without a currency marker, and a wrong amount is worse than a missing one.
_AMOUNT = re.compile(
    r"(?P<currency>[₩$€£¥])\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d{2})"
    r"|(?P<num2>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d{2})\s?(?P<currency2>[₩$€£¥])"
)

#: Invoice/reference identifiers: a letter prefix, separator, digits.
_IDENTIFIER = re.compile(r"\b(?P<id>[A-Z]{2,5}[-/]?\d{3,10})\b")

#: A label followed by a value, which is the only place a bare number is trusted.
_LABELLED_AMOUNT = re.compile(
    r"(?i:total|amount|sum)\s*[:=]?\s*(?P<currency>[₩$€£¥])?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d{2})"
)

_DATE = re.compile(r"\b(?P<date>\d{4}-\d{2}-\d{2})\b")

#: Something that looks like it *wants* to be an amount but is damaged -- e.g.
#: OCR read a lowercase L for the leading 1 ("l,234.56"). Detecting this is what
#: separates "the field is absent" from "the field is there and unreadable":
#: reporting the latter as NOT_PRESENT hides a real extraction failure inside a
#: legitimate-looking abstention.
_DAMAGED_AMOUNT = re.compile(r"[lIO|][,.]\d{3}|\d[,.]\d{3}[,.][lIO]")

#: Text that looks like a document header rather than an invoice.
_INVOICE_HINT = re.compile(r"(?i)\b(invoice|tax invoice|receipt|청구서|계산서)\b")
_IDENTIFIER_HINT = re.compile(r"(?i)\binvoice\s*(?:no|number|#)\b")


class Extractor(Protocol):
    def extract(self, chunk: Chunk) -> Extraction | None: ...


class TemplateExtractor(BaseModel):
    """Pattern-based extractor for invoice-shaped chunks.

    Returns ``Unclassified`` -- not ``None`` and not a failure -- when nothing
    recognisable is present. That distinction matters: an unclassified chunk is
    a scope observation for the ontology, while a failed extraction is a parser
    problem, and conflating them corrupts the metrics that drive rework.
    """

    model_config = ConfigDict(frozen=True)

    #: Require an explicit "Invoice no." style label, not just any identifier.
    require_identifier_label: bool = False

    def extract(self, chunk: Chunk) -> Extraction | None:
        if not chunk.text.strip():
            return None

        identifier = self._identifier(chunk)
        amount = self._amount(chunk)
        issue_date = self._date(chunk)
        currency = self._currency(chunk)

        looked_like_invoice = bool(_INVOICE_HINT.search(chunk.text))

        if not any(
            isinstance(field, FieldValue)
            for field in (identifier, amount, issue_date, currency)
        ):
            if not looked_like_invoice:
                return Unclassified(
                    observed_shape=_shape_hint(chunk.text),
                    provenance=_whole_chunk_span(chunk),
                )
            # It said "invoice" but nothing was parseable: genuinely unclassified,
            # and the review question should ask what the document should look like.
            return Unclassified(
                observed_shape="invoice_header_unparseable",
                provenance=_whole_chunk_span(chunk),
            )

        return InvoiceHeader(
            invoice_number=identifier,
            issue_date=issue_date,
            total_amount=amount,
            currency=currency,
        )

    # -- field helpers ----------------------------------------------------

    def _identifier(self, chunk: Chunk):
        pattern = _IDENTIFIER
        for match in pattern.finditer(chunk.text):
            if self.require_identifier_label and not _IDENTIFIER_HINT.search(
                chunk.text[: match.start()][-40:]
            ):
                continue
            span = chunk.span_for(match.start(), match.end())
            if span is None:
                continue
            return FieldValue(value=match.group("id"), provenance=span)
        return FieldAbstention(reason=AbstainReason.NOT_PRESENT)

    def _amount(self, chunk: Chunk):
        # Prefer a labelled amount; fall back to a currency-marked one.
        labelled = _LABELLED_AMOUNT.search(chunk.text)
        if labelled is not None:
            span = chunk.span_for(labelled.start("num"), labelled.end("num"))
            if span is not None:
                return FieldValue(value=labelled.group("num"), provenance=span)

        for match in _AMOUNT.finditer(chunk.text):
            number = match.group("num") or match.group("num2")
            if not number:
                continue
            start, end = match.span()
            span = chunk.span_for(start, end)
            if span is None:
                continue
            return FieldValue(value=number, provenance=span)

        if _DAMAGED_AMOUNT.search(chunk.text):
            # An amount is present but unreadable. Saying NOT_PRESENT here would
            # be a lie that makes an OCR failure look like a clean document.
            return FieldAbstention(reason=AbstainReason.ILLEGIBLE)
        return FieldAbstention(reason=AbstainReason.NOT_PRESENT)

    def _currency(self, chunk: Chunk):
        labelled = _LABELLED_AMOUNT.search(chunk.text)
        if labelled is not None and labelled.group("currency"):
            span = chunk.span_for(labelled.start(), labelled.end())
            if span is not None:
                return FieldValue(value=labelled.group("currency"), provenance=span)

        for match in _AMOUNT.finditer(chunk.text):
            currency = match.group("currency") or match.group("currency2")
            if not currency:
                continue
            span = chunk.span_for(match.start(), match.end())
            if span is not None:
                return FieldValue(value=currency, provenance=span)

        return FieldAbstention(reason=AbstainReason.NOT_PRESENT)

    def _date(self, chunk: Chunk):
        match = _DATE.search(chunk.text)
        if match is None:
            return FieldAbstention(reason=AbstainReason.NOT_PRESENT)
        span = chunk.span_for(match.start("date"), match.end("date"))
        if span is None:
            return FieldAbstention(reason=AbstainReason.AMBIGUOUS)
        return FieldValue(value=match.group("date"), provenance=span)


def _whole_chunk_span(chunk: Chunk):
    """A provenance ref covering the chunk, for unclassified reporting."""
    from rag_ingest.docmodel import ProvenanceRef

    if chunk.segments:
        first = chunk.segments[0]
        return ProvenanceRef(
            page=first.page,
            char_start=first.raw_start,
            char_end=first.raw_end,
            element_id=first.element_id,
            bbox=first.bbox,
        )
    return ProvenanceRef(page=chunk.page_start, char_start=0, char_end=1, bbox=chunk.bbox)


def _shape_hint(text: str) -> str:
    """A crude label for what a chunk looks like, for the review queue."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    if "|" in stripped or stripped.count("\n") > 2:
        return "tabular"
    if len(stripped) < 40:
        return "heading_or_label"
    return "prose"


class NoExtractor(BaseModel):
    """Extraction deliberately disabled: every chunk becomes unclassified."""

    model_config = ConfigDict(frozen=True)

    def extract(self, chunk: Chunk) -> Extraction | None:
        return None


__all__ = [
    "Extractor",
    "NoExtractor",
    "TemplateExtractor",
]
