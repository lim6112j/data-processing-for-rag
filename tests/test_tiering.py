"""Plan section 3, failure class A: gate on parser scores before semantics.

Born-digital corpus: no OCR tier exists. A bad parse score escalates
directly -- there is no re-scan path to offer, only a human question.
"""

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
    decision = select_tier(_element(layout=0.9, parse=0.9))
    assert decision.tier is Tier.EXTRACT


def test_bad_parse_score_escalates():
    # No re-scan path in a born-digital corpus: the text layer is either
    # there or it is not. Escalate rather than extract on damaged text.
    decision = select_tier(_element(layout=0.95, parse=0.2))
    assert decision.tier is Tier.ESCALATE


def test_bad_layout_gives_structure_unknown_not_escalation():
    # Structure suspicion is a chunking concern, not a reading failure.
    decision = select_tier(_element(layout=0.3, parse=0.9))
    assert decision.tier is Tier.STRUCTURE_UNKNOWN


def test_parse_score_is_adjudicated_before_layout():
    # Order matters: broken text makes structure questions meaningless.
    decision = select_tier(_element(layout=0.1, parse=0.1))
    assert decision.tier is Tier.ESCALATE
    assert any("parse_score" in r for r in decision.reasons)


def test_empty_text_escalates_even_with_good_scores():
    decision = select_tier(_element(text="   ", layout=0.9, parse=0.9))
    assert decision.tier is Tier.ESCALATE


def test_no_scores_at_all_is_treated_as_extractable():
    assert select_tier(_element()).tier is Tier.EXTRACT


def test_partial_scores_do_not_block():
    # A parser that reports only layout is not evidence of failure.
    assert select_tier(_element(layout=0.9)).tier is Tier.EXTRACT
