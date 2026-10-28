# ruff: noqa: RUF001  -- Korean spec notation uses the real multiplication sign
"""Live LLM proposer check: real model, real Korean inventory rows.

Marked ``live``: needs OPENROUTER_API_KEY and network. Runs rarely; its job
is to catch prompt/schema drift against the real model, not to gate CI.
"""
import os

import pytest

from rag_ingest.llm_propose import LlmProposer

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set",
    ),
]

KOREAN_INVENTORY_ROWS = (
    ("품목명", "규격", "단위", "재고", "안전재고"),
    ("육안검사기", "KS-A 12mm", "개", "150", "30"),
    ("베어링6204", "20×47×14", "개", "2,400", "500"),
    ("윤활유ISO32", "드럼", "통", "12", "3"),
)


def test_real_model_proposes_the_korean_inventory_schema():
    proposer = LlmProposer()  # resolves OPENROUTER_API_KEY from the env

    proposal = proposer.propose(
        "row:text,text,text,number,number", KOREAN_INVENTORY_ROWS
    )

    if not proposal.parseable:
        pytest.fail(f"real model refused: {proposal.note}")

    assert proposal.source == "llm"
    assert proposal.fields
    # Whatever the model proposes must cite evidence that exists: this is the
    # invariant the layer enforces, checked here against a real response.
    for evidence in proposal.evidence:
        row = KOREAN_INVENTORY_ROWS[evidence.row_index]
        assert evidence.sample_value in row
    names = [f.name for f in proposal.fields]
    assert len(names) == len(set(names)), f"duplicate field names: {names}"
