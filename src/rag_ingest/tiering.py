"""Tier selection: decide what to attempt *before* any semantic extraction.

Plan §3, failure class A: do not attempt semantic extraction on unreliable
text. The parser's per-element scores are used as a tier selector here, not as
a correctness signal.

The corpus is born-digital documents only, so there is no OCR stage: a text
layer either parses (parse_score) or the document escalates. Layout quality
still matters for a different reason -- chunk boundaries -- not for reading
the glyphs.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.docmodel import Element, ElementScores

#: Below this layout confidence the structure (hence chunk boundaries) is suspect.
LAYOUT_FLOOR = 0.6
#: Below this parse score the digital text layer is likely damaged.
PARSE_FLOOR = 0.5


class Tier(StrEnum):
    """What the pipeline should do with one element."""

    #: Scores are acceptable; extract and chunk normally.
    EXTRACT = "extract"
    #: Text is fine but structure is unreliable; extract as flat text only.
    STRUCTURE_UNKNOWN = "structure_unknown"
    #: Nothing usable; escalate to a human without a semantic attempt.
    ESCALATE = "escalate"


class TierDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    tier: Tier
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    scores: ElementScores

    @property
    def is_extractable(self) -> bool:
        return self.tier in (Tier.EXTRACT, Tier.STRUCTURE_UNKNOWN)


def _below(value: float | None, floor: float) -> bool:
    return value is not None and value < floor


def select_tier(element: Element) -> TierDecision:
    """Route one element to a processing tier.

    An element with no scores at all is treated as extractable -- a parser
    that reports nothing is not evidence of failure, and refusing to process
    it would silently drop every document from a parser with no confidence
    support.
    """
    scores = element.scores
    if scores.is_empty:
        return TierDecision(tier=Tier.EXTRACT, reasons=("no_scores_reported",), scores=scores)

    reasons: list[str] = []

    if _below(scores.parse_score, PARSE_FLOOR):
        reasons.append(f"parse_score={scores.parse_score:.2f}<{PARSE_FLOOR}")
        return TierDecision(tier=Tier.ESCALATE, reasons=tuple(reasons), scores=scores)

    if _below(scores.layout_score, LAYOUT_FLOOR):
        reasons.append(f"layout_score={scores.layout_score:.2f}<{LAYOUT_FLOOR}")
        return TierDecision(
            tier=Tier.STRUCTURE_UNKNOWN, reasons=tuple(reasons), scores=scores
        )

    if not element.rendered().strip():
        reasons.append("element_rendered_empty")
        return TierDecision(tier=Tier.ESCALATE, reasons=tuple(reasons), scores=scores)

    return TierDecision(tier=Tier.EXTRACT, scores=scores)
