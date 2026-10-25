"""Plan section 3, failure class A: gate on OCR/layout scores before semantics."""

from __future__ import annotations

from rag_ingest.docmodel import Element, ElementKind
from rag_ingest.tiering import Tier, select_tier
from tests.conftest import scores


def _element(text: str = "readable text", **score_kwargs) -> Element:
    return Element(
        element_id="e1",
        kind=ElementKind.PARAGRAPH,
        page=1,
        text=text,
        scores=scores(**score_kwargs),
    )


def test_good_scores_extract():
    assert select_tier(_element(ocr=0.95, layout=0.9, parse=0.9)).tier is Tier.EXTRACT


def test_poor_ocr_triggers_retry_rather_than_semantic_extraction():
    decision = select_tier(_element(ocr=0.3, layout=0.95, parse=0.95))

    assert decision.tier is Tier.RETRY_OCR
    assert not decision.is_extractable


def test_poor_parse_score_triggers_retry():
    assert select_tier(_element(parse=0.2, ocr=0.95)).tier is Tier.RETRY_OCR


def test_ocr_is_adjudicated_before_layout():
    # Both bad: an unreadable glyph cannot be rescued by good structure, but a
    # structurally broken page can be re-read. OCR quality wins the decision.
    decision = select_tier(_element(ocr=0.1, layout=0.1))

    assert decision.tier is Tier.RETRY_OCR


def test_poor_layout_marks_structure_unknown_but_still_extractable():
    decision = select_tier(_element(ocr=0.9, layout=0.3))

    assert decision.tier is Tier.STRUCTURE_UNKNOWN
    assert decision.is_extractable


def test_empty_rendered_element_escalates():
    assert select_tier(_element(text="   ", ocr=0.9, layout=0.9)).tier is Tier.ESCALATE


def test_element_with_no_scores_is_treated_as_extractable():
    # A parser that reports nothing is not evidence of failure.
    decision = select_tier(_element())

    assert decision.tier is Tier.EXTRACT
    assert "no_scores_reported" in decision.reasons


def test_thresholds_boundary_is_inclusive():
    assert select_tier(_element(ocr=0.5)).tier is Tier.EXTRACT
    assert select_tier(_element(layout=0.6)).tier is Tier.EXTRACT
