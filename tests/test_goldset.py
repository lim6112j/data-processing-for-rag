"""The gold set is the contract. If these fail, the pipeline's decisions are wrong."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from rag_ingest.escalation import ReviewQueue
from rag_ingest.goldset import (
    GoldExtractor,
    build_document,
    check_case,
    load_goldset,
)
from rag_ingest.pipeline import IngestPipeline

GOLDSET = Path(__file__).parent.parent / "goldset" / "cases.yaml"
TODAY = date(2026, 10, 25)


@pytest.fixture(scope="module")
def goldset():
    return load_goldset(GOLDSET)


@pytest.fixture(scope="module")
def case_ids(goldset):
    return [case.id for case in goldset.cases]


def _run(case):
    raw, elements = build_document(case)
    outcome = IngestPipeline(
        ReviewQueue(), GoldExtractor(case, raw), today=TODAY
    ).ingest(raw, elements, owner_id="owner-1", retrieval_demand=5)
    hits = [record.text for record in outcome.verified_chunks]
    return check_case(case, outcome, hits), outcome


def test_goldset_loads_with_cases(goldset):
    assert len(goldset.cases) >= 5


def test_every_case_passes(goldset):
    failures = []
    for case in goldset.cases:
        result, _ = _run(case)
        if not result.passed:
            failures.append(f"{case.id}: {'; '.join(result.failures)}")

    assert not failures, "gold-set failures:\n" + "\n".join(failures)


def test_clean_invoice_verifies(goldset):
    case = next(c for c in goldset.cases if c.id == "clean-invoice")
    result, outcome = _run(case)

    assert result.passed
    assert len(outcome.verified_chunks) == 1


def test_ocr_damage_is_never_retrieved(goldset):
    # The headline case: the invented number must not reach retrieval.
    case = next(c for c in goldset.cases if c.id == "ocr-lowercase-l-total")
    result, outcome = _run(case)

    assert result.passed
    assert outcome.verified_chunks == ()
    assert all("1334.56" not in chunk.text for chunk in outcome.quarantined_chunks)


def test_unreadable_page_is_never_extracted(goldset):
    case = next(c for c in goldset.cases if c.id == "scanned-no-text-layer")
    result, outcome = _run(case)

    assert result.passed
    assert outcome.verified_chunks == ()
    assert outcome.quarantined_chunks == ()
    assert outcome.review_items


def test_unclassified_is_queued_not_dropped(goldset):
    case = next(c for c in goldset.cases if c.id == "unclassified-poem")
    result, outcome = _run(case)

    assert result.passed
    assert outcome.review_items[0].reason.value == "unclassified"


def test_long_table_stays_one_chunk(goldset):
    case = next(c for c in goldset.cases if c.id == "table-must-not-split")
    result, outcome = _run(case)

    assert result.passed
    table_chunks = [c for c in outcome.quarantined_chunks if ":table:" in c.chunk_id]
    assert len(table_chunks) == 1
    assert table_chunks[0].text.count("\n") == case.table_rows


def test_mixed_document_quarantines_only_the_bad_chunk(goldset):
    # The 200-page document case: a good section stays retrievable while the
    # illegible section is withheld. One bad chunk never blocks the document.
    case = next(c for c in goldset.cases if c.id == "mixed-document-partial-quarantine")
    result, outcome = _run(case)

    assert result.passed
    assert len(outcome.verified_chunks) == 1
    verified_text = outcome.verified_chunks[0].text
    assert "INV-2043" in verified_text
    assert "scribble" not in verified_text


def test_case_result_records_reasons(goldset):
    case = next(c for c in goldset.cases if c.id == "ocr-lowercase-l-total")
    result, _ = _run(case)

    assert result.passed
    assert result.failures == ()
