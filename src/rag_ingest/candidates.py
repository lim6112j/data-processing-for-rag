"""Candidate production: where T2's disagreements actually come from.

Plan §2 gives T2 one job -- settle disagreements between extractors -- but a
disagreement has to be *manufactured* by running two parses over the same
region. This module is the seam that does it.

The hard part is not calling the second parser; it is **alignment**. Two parses
of the same page produce element lists that differ in count, order and
granularity. Before T2 can be asked "which reading is right?", the two readings
must first be established to describe the *same region*. Misaligned candidates
produce a confident answer about two different things, which is worse than no
adjudication at all.

Alignment is done on bbox overlap, not on text similarity: text is exactly what
the two parses disagree about, so using it to align them would beg the question.

A producer that cannot find a genuine counterpart returns fewer than two
candidates, and T2 is skipped. That is the correct behaviour -- inventing a
second reading to have something to adjudicate is fiction.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from rag_ingest.adjudication import Candidate
from rag_ingest.chunking import Chunk
from rag_ingest.docmodel import BBox, Element

#: Minimum intersection-over-union for two boxes to be considered the same
#: region. Deliberately loose: parsers disagree about padding, and a near-miss
#: is still a genuine disagreement worth surfacing. Below this the boxes are
#: treated as different regions entirely.
MIN_IOU = 0.4

#: A second parse's element may be nested inside the chunk's box rather than
#: covering it (e.g. one table row vs the whole table). Containment this high
#: is accepted as the same region.
MIN_CONTAINMENT = 0.8

#: Cap on how much larger the containing box may be. A page-scale element
#: covers every chunk on the page, so unbounded nesting would make it "agree"
#: with all of them and manufacture disagreements everywhere.
#:
#: 12 admits a table up to ~11 rows against one of its rows (the realistic
#: granularity difference) while rejecting a page-sized element against a
#: normal chunk (~24x on US Letter). Beyond the cap the element is not a
#: specific counterpart, and skipping is the safe direction: no adjudication
#: rather than a verdict about two different regions.
MAX_NESTING_RATIO = 12.0


def _area(box: BBox) -> float:
    left, top, right, bottom = box
    return max(0.0, right - left) * max(0.0, bottom - top)


def _intersection(a: BBox, b: BBox) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def iou(a: BBox, b: BBox) -> float:
    """Intersection over union. 0.0 when either box is degenerate."""
    inter = _intersection(a, b)
    if inter == 0.0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return 0.0 if union <= 0 else inter / union


def containment(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner`` covered by ``outer``."""
    inner_area = _area(inner)
    if inner_area <= 0:
        return 0.0
    return _intersection(inner, outer) / inner_area


def _nesting_is_local(inner: BBox, outer: BBox) -> bool:
    """Whether ``inner`` sits inside ``outer`` without outer being page-scale.

    Without this cap, a parser that returns one element covering the whole page
    would "agree" with every chunk on that page, manufacturing a disagreement
    against all of them.
    """
    inner_area = _area(inner)
    outer_area = _area(outer)
    if inner_area <= 0 or outer_area <= 0:
        return False
    return outer_area / inner_area <= MAX_NESTING_RATIO


def boxes_agree(a: BBox, b: BBox) -> bool:
    """Whether two boxes address the same region closely enough to compare.

    Containment is checked **both ways**, because the two parsers may differ in
    granularity in either direction: one may return a whole table where the
    other returned a single row, or vice versa. Testing one direction only
    would silently drop half of those genuine disagreements. ``_nesting_is_local``
    keeps a page-scale element from matching everything.
    """
    if iou(a, b) >= MIN_IOU:
        return True
    if containment(a, b) >= MIN_CONTAINMENT and _nesting_is_local(a, b):
        return True
    return containment(b, a) >= MIN_CONTAINMENT and _nesting_is_local(b, a)


class CandidateProducer(Protocol):
    """Supplies competing readings for one chunk.

    Returning fewer than two candidates means "nothing to adjudicate", and T2
    is skipped. Implementations must not fabricate a second reading.
    """

    def produce(self, chunk: Chunk) -> list[Candidate]: ...


class NullProducer:
    """Default producer: one reading, therefore no disagreement.

    This is what a single-parser deployment gets, and it is honest -- there is
    genuinely nothing for T2 to decide.
    """

    def produce(self, chunk: Chunk) -> list[Candidate]:
        if not chunk.text.strip():
            return []
        return [
            Candidate(
                candidate_id=f"{chunk.chunk_id}:single",
                produced_by="parser_only",
                proposed_text=chunk.text,
                bbox=chunk.bbox,
            )
        ]


class SecondParse(BaseModel):
    """A parse of the same document by a different parser (or settings)."""

    model_config = ConfigDict(frozen=True)

    name: str
    elements: tuple[Element, ...]

    def elements_for(self, page: int) -> tuple[Element, ...]:
        return tuple(element for element in self.elements if element.page == page)


class AlternateParseProducer(BaseModel):
    """Builds candidates from a second parse, aligned by page and bbox.

    ``model_config`` is frozen so a producer cannot mutate between documents;
    the second parse is fixed at construction.
    """

    model_config = ConfigDict(frozen=True)

    primary_name: str = "parser_primary"
    alternate: SecondParse
    #: Emit a candidate when the alternate parse found nothing at that region.
    #:
    #: Defaults to **False**, and that default is deliberate. Cross-origin
    #: parsers routinely segment differently -- pdfplumber emits one text block
    #: per page where Docling emits many paragraphs, so a paragraph chunk often
    #: has no counterpart at all. Treating that as "the other parser read
    #: nothing here" manufactured a disagreement on essentially every paragraph
    #: and would have sent constant noise to T2.
    #:
    #: A true absence (one parser found a table, the other found no content
    #: whatsoever at that box) is a real signal, so this is available -- but it
    #: should be enabled per document type, after the disagreement rate has been
    #: observed, not globally by default.
    allow_missing: bool = False

    def produce(self, chunk: Chunk) -> list[Candidate]:
        if not chunk.bbox:
            # Without a box there is nothing to align on, so a second reading
            # cannot be shown to describe the same region. Skip rather than
            # align on text, which is the thing in dispute.
            return []

        matches = [
            element
            for element in self.alternate.elements_for(chunk.page_start)
            if element.bbox is not None and boxes_agree(element.bbox, chunk.bbox)
        ]
        alternate_text = "\n\n".join(
            rendered for rendered in (element.rendered() for element in matches) if rendered
        )

        candidates: list[Candidate] = [
            Candidate(
                candidate_id=f"{chunk.chunk_id}:primary",
                produced_by=self.primary_name,
                proposed_text=chunk.text,
                bbox=chunk.bbox,
            )
        ]
        if not alternate_text.strip():
            if not self.allow_missing:
                return candidates
            candidates.append(
                Candidate(
                    candidate_id=f"{chunk.chunk_id}:missing",
                    produced_by=self.alternate.name,
                    proposed_text="",
                    bbox=chunk.bbox,
                    note="this parser found no content at this region",
                )
            )
            return candidates

        if _same_text(chunk.text, alternate_text):
            # Agreement is not a disagreement. Emitting identical readings would
            # spend a model call to be told they match.
            return candidates

        candidates.append(
            Candidate(
                candidate_id=f"{chunk.chunk_id}:alternate",
                produced_by=self.alternate.name,
                proposed_text=alternate_text,
                bbox=chunk.bbox,
                note="aligned by page and bounding-box overlap",
            )
        )
        return candidates


def _same_text(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def describe_disagreement(candidates: list[Candidate]) -> str:
    """Human-readable summary for the review item, or '' if no disagreement."""
    if len(candidates) < 2:
        return ""
    return " / ".join(f"{c.produced_by}: {c.proposed_text[:60]!r}" for c in candidates)


__all__ = [
    "MAX_NESTING_RATIO",
    "MIN_CONTAINMENT",
    "MIN_IOU",
    "AlternateParseProducer",
    "CandidateProducer",
    "NullProducer",
    "SecondParse",
    "boxes_agree",
    "containment",
    "describe_disagreement",
    "iou",
]
