"""Live OpenRouter check. Skipped unless OPENROUTER_API_KEY is set.

This is the only test that spends money, so it is opt-in and marked. It exists
to catch wire-format drift (schema rejection, image encoding, model name) that
a fake client cannot.

Run explicitly:  uv run pytest -m live

The key is never read into a log, a fixture, or a failure message -- the SDK
resolves it from the environment.
"""

from __future__ import annotations

import os

import pytest

from rag_ingest.adapters.openrouter import OpenRouterAdjudicator
from rag_ingest.adjudication import Candidate, DisagreementKind, build_question
from rag_ingest.docmodel import ProvenanceRef

pytestmark = pytest.mark.live

_HAS_KEY = bool(os.environ.get("OPENROUTER_API_KEY"))

#: A 1x1 PNG: enough to prove the image channel is accepted. Not enough to
#: adjudicate anything, so a *correct* model answer here is an abstention.
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000002000100ffff03000006000557bfabd40000"
    "000049454e44ae426082"
)


@pytest.mark.skipif(not _HAS_KEY, reason="OPENROUTER_API_KEY not set")
def test_live_adjudication_round_trip():
    adjudicator = OpenRouterAdjudicator(timeout=60.0)
    question = build_question(
        chunk_id="live-1",
        doc_id="live-doc",
        kind=DisagreementKind.COLUMN_COUNT,
        prompt="How many columns does this table have? Answer from the image.",
        candidates=[
            Candidate(candidate_id="a", produced_by="pdfplumber", proposed_text="4 columns"),
            Candidate(candidate_id="b", produced_by="tableformer", proposed_text="5 columns"),
        ],
        provenance=ProvenanceRef(page=1, char_start=0, char_end=1, bbox=(0.0, 0.0, 1.0, 1.0)),
        context_text="a | b | c",
    )

    decision = adjudicator.adjudicate_with_evidence(question, image_png=_TINY_PNG)

    # A blank 1x1 image cannot support either reading, so a well-behaved model
    # abstains. Any coherent decision proves the wire format works; the point of
    # this test is the transport, not the judgement.
    assert decision.kind in {"pick", "abstain", "escalate"}
    assert decision.is_coherent(len(question.candidates))
    assert "transport failure" not in (decision.rationale or "")
    assert "unreadable output" not in (decision.rationale or "")
