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

from rag_ingest.chunking import Chunk, ChunkKind
from rag_ingest.defects import DAMAGED_AMOUNT
from rag_ingest.ontology import (
    AbstainReason,
    Extraction,
    FieldAbstention,
    FieldValue,
    InvoiceHeader,
    LineItem,
    ProseKnowledge,
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

#: An explicitly labelled line row: "Line 1:", "Item 2:", "1." before a
#: currency-marked amount. Deliberately narrow -- collecting every
#: currency-marked number would reconcile the total against itself, since the
#: total itself is a currency-marked number. Only a row label makes a number
#: a line item, and the damaged-amount alternative keeps a damaged ROW visible
#: as ILLEGIBLE rather than disappearing into a clean-looking record.
_LINE_ROW = re.compile(
    r"(?im)^\s*(?:line|item|품목)\s*\d+\s*[:\-]\s*.*?"
    r"(?P<currency>[₩$€£¥])?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d{2})"
    r"\s?(?P<currency2>[₩$€£¥])?",
)

#: Something that looks like it *wants* to be an amount but is damaged -- e.g.
#: OCR read a lowercase L for the leading 1 ("l,234.56"). Detecting this is what
#: separates "the field is absent" from "the field is there and unreadable":
#: reporting the latter as NOT_PRESENT hides a real extraction failure inside a
#: legitimate-looking abstention. Owned by ``defects`` and shared, because the
#: same pattern drives defect marking on admitted prose.
_DAMAGED_AMOUNT = DAMAGED_AMOUNT

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
                    observed_shape=_shape_hint(chunk),
                    provenance=_whole_chunk_span(chunk),
                )
            # It said "invoice" but nothing was parseable: genuinely unclassified,
            # and the review question should ask what the document should look like.
            return Unclassified(
                observed_shape="invoice_header_unparseable",
                provenance=_whole_chunk_span(chunk),
            )

        # Corroboration. An identifier alone does not open a record, because a
        # part number is shaped exactly like an invoice number -- measured on FAA
        # AC 43.13-1B, where ``_IDENTIFIER`` matched AN310/MS20365/NAS1223 and
        # every one of them became an InvoiceHeader. Grounding cannot reject
        # those: the token *is* verbatim in the source, so the gate sees no
        # mismatch. It verifies presence, not semantic role, so the refusal has
        # to happen here, before the record exists.
        #
        # Either kind of evidence is enough on its own: a monetary amount is
        # evidence no hardware designation can supply, and an explicit invoice
        # label is the strongest evidence of all. Requiring both would drop real
        # invoice headers whose amount did not parse (see
        # ``test_missing_fields_abstain_rather_than_invent``).
        corroborated = isinstance(amount, FieldValue) or looked_like_invoice
        if not corroborated:
            return Unclassified(
                observed_shape=_shape_hint(chunk),
                provenance=_whole_chunk_span(chunk),
            )

        return InvoiceHeader(
            invoice_number=identifier,
            issue_date=issue_date,
            total_amount=amount,
            currency=currency,
            line_items=self._line_items(chunk),
        )

    # -- field helpers ----------------------------------------------------

    def _line_items(self, chunk: Chunk) -> tuple[LineItem, ...]:
        """Explicitly labelled line rows, spans resolvable or row dropped.

        A row whose amount cannot be anchored to the page buffer is worthless
        for reconciliation, so it is skipped entirely rather than carried with
        a fabricated span -- same rule as every other field here. Damaged but
        anchored amounts become ILLEGIBLE abstentions so the gap stays
        visible.
        """
        items: list[LineItem] = []
        for match in _LINE_ROW.finditer(chunk.text):
            amount = self._line_amount(match, chunk)
            if amount is None:
                continue
            items.append(LineItem(amount=amount))
        return tuple(items)

    def _line_amount(self, match, chunk: Chunk):
        num = match.group("num")
        # The amount portion must not begin mid-number: "l,234.56" leaves
        # "234.56" for the regex, which would read a damaged row as a clean
        # 234.56 and poison reconciliation. Two characters of look-behind is
        # the minimum the damage pattern needs ([lIO|] + separator); a
        # damaged-amount hit in that window means the row exists but its
        # amount is unreadable.
        look_behind = match.group(0)[max(0, match.start("num") - match.start(0) - 2) :]
        if num is None or _DAMAGED_AMOUNT.search(look_behind):
            return FieldAbstention(reason=AbstainReason.ILLEGIBLE)
        span = chunk.span_for(match.start("num"), match.end("num"))
        if span is None:
            return None
        return FieldValue(value=num, provenance=span)

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


def _shape_hint(chunk: Chunk) -> str:
    """A crude label for what a chunk looks like, for the review queue.

    Table-ness comes from the chunker's own ``kind`` plus the pipe rendering a
    table produces -- never from counting newlines. Counting newlines reported
    any three-line paragraph as a table, which mislabelled 746 of 754 chunks on
    the FAA manual (see ``corpus/README.md``) and corrupts the document-shape
    metric that ``survey.md`` says decides whether a lenient tier is worth
    building.
    """
    stripped = chunk.text.strip()
    if not stripped:
        return "empty"
    if chunk.kind is ChunkKind.TABLE or "|" in stripped:
        return "tabular"
    if len(stripped) < 40:
        return "heading_or_label"
    return "prose"


class NoExtractor(BaseModel):
    """Extraction deliberately disabled: every chunk becomes unclassified."""

    model_config = ConfigDict(frozen=True)

    def extract(self, chunk: Chunk) -> Extraction | None:
        return None


class ProseExtractor:
    """Quotes the chunk verbatim, with the span that contains it.

    Mechanical on purpose: no model, no regex, no interpretation. The only claim
    it makes is "this text is what the source says at this span", which is
    exactly what a LENIENT gate can check. Anything stronger would be a claim
    this extractor cannot support.

    A chunk spanning pages is declined. ``ProvenanceRef`` cites one page, so a
    multi-page chunk cannot be quoted through a single resolvable span, and
    quoting only its first page would quietly understate what was admitted. On a
    646-page manual 96% of chunks are single-page (``corpus/README.md``), so the
    honest refusal costs little; the alternative costs the meaning of "verified".
    """

    def extract(self, chunk: Chunk) -> Extraction | None:
        text = chunk.text.strip()
        if not text:
            return None
        if chunk.page_start != chunk.page_end:
            return Unclassified(
                observed_shape="multi_page_span",
                provenance=_whole_chunk_span(chunk),
            )
        span = _chunk_span(chunk)
        if span is None:
            return Unclassified(
                observed_shape=_shape_hint(chunk),
                provenance=_whole_chunk_span(chunk),
            )
        return ProseKnowledge(text=FieldValue(value=text, provenance=span))


class LayeredExtractor:
    """Tries a typed extractor, then falls back to verbatim prose.

    The fallback is what makes the LENIENT tier reachable at all: without it the
    only typed extractor in the tree knows one invoice shape, so every prose chunk
    is ``unclassified`` and the gate admits nothing from a manual.

    One deliberate exception. When the template says the text *looks like* an
    invoice header but parsed nothing (``invoice_header_unparseable``), prose does
    not take over. The reviewer should be asked what the document should look
    like -- a knowledgeable record is not an answer to that question, and
    burying it under a verbatim quote would hide a template that no longer
    matches.
    """

    def __init__(
        self, primary: Extractor | None = None, fallback: Extractor | None = None
    ) -> None:
        self._primary = primary if primary is not None else TemplateExtractor()
        self._fallback = fallback if fallback is not None else ProseExtractor()

    def extract(self, chunk: Chunk) -> Extraction | None:
        first = self._primary.extract(chunk)
        if not isinstance(first, Unclassified):
            return first
        if first.observed_shape == "invoice_header_unparseable":
            return first
        second = self._fallback.extract(chunk)
        return first if second is None else second


def _chunk_span(chunk: Chunk):
    """One span covering the whole chunk, or None when it cannot be cited.

    Requires every segment on one page. ``_whole_chunk_span`` returns the first
    segment only, which is right for reporting what a region looked like but
    wrong for quoting: it would verify the head of a chunk and silently ignore
    the rest.
    """
    from rag_ingest.docmodel import ProvenanceRef

    if not chunk.segments:
        return None
    pages = {segment.page for segment in chunk.segments}
    if len(pages) != 1:
        return None
    first, last = chunk.segments[0], chunk.segments[-1]
    if last.raw_end <= first.raw_start:
        return None
    return ProvenanceRef(
        page=first.page,
        char_start=first.raw_start,
        char_end=last.raw_end,
        element_id=first.element_id,
        bbox=chunk.bbox,
    )


__all__ = [
    "Extractor",
    "LayeredExtractor",
    "NoExtractor",
    "ProseExtractor",
    "TemplateExtractor",
]
