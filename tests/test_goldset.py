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


class TestReviewQuestionWording:
    """The question must say what to do, not just what is wrong.

    A reviewer reading "missing: issue_date, total_amount, currency" cannot tell
    a damaged scan from absent information -- so they cannot know whether to
    rescan or to supply a value.
    """

    def _item(self, text: str):
        """Run the real extractor over a document and return its review item."""
        from rag_ingest.docmodel import Element, ElementKind, RawDocument
        from rag_ingest.escalation import ReviewQueue
        from rag_ingest.extractors import TemplateExtractor
        from rag_ingest.pipeline import IngestPipeline

        raw, elements = RawDocument.from_elements(
            "d1",
            [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1, text=text)],
        )
        queue = ReviewQueue()
        outcome = IngestPipeline(queue, TemplateExtractor(), today=TODAY).ingest(
            raw, elements, owner_id="o"
        )
        # The pipeline produces items; enqueueing is the caller's job.
        for review_item in outcome.review_items:
            queue.enqueue(review_item)
        return queue.open_items()[0]

    def test_damaged_value_is_described_as_unreadable_not_absent(self):
        item = self._item("Invoice INV-3003 total l,234.56 USD")

        assert "could not read" in item.question
        assert "total amount" in item.question
        # The reviewer must not be told to supply a value that is right there.
        assert "did not find total amount" not in item.question

    def test_absent_fields_are_separated_from_damaged_ones(self):
        item = self._item("Invoice INV-3003 total l,234.56 USD")

        # currency is absent; total_amount is damaged. Different actions.
        assert "did not find" in item.question
        assert "currency" in item.question

    def test_empty_heading_does_not_produce_an_empty_quote(self):
        # "under ''" was the original bug: a chunk with no heading produced a
        # sentence quoting nothing.
        item = self._item("Invoice INV-3003 total l,234.56 USD")

        assert "''" not in item.question

    def test_question_names_what_was_read(self):
        item = self._item("Invoice INV-3003 total l,234.56 USD")

        assert "read invoice number" in item.question

    def test_heading_is_quoted_when_one_exists(self):
        from rag_ingest.docmodel import Element, ElementKind, RawDocument
        from rag_ingest.escalation import ReviewQueue
        from rag_ingest.extractors import TemplateExtractor
        from rag_ingest.pipeline import IngestPipeline

        raw, elements = RawDocument.from_elements(
            "d1",
            [
                Element(element_id="h", kind=ElementKind.HEADING, page=1, text="Invoices", level=1),
                Element(
                    element_id="e1",
                    kind=ElementKind.PARAGRAPH,
                    page=1,
                    text="Invoice INV-3003 total l,234.56 USD",
                ),
            ],
        )
        queue = ReviewQueue()
        outcome = IngestPipeline(queue, TemplateExtractor(), today=TODAY).ingest(
            raw, elements, owner_id="o"
        )
        for review_item in outcome.review_items:
            queue.enqueue(review_item)

        assert "under 'Invoices'" in queue.open_items()[0].question


class TestUnclassifiedQuestion:
    """An unclassified question must be answerable.

    It used to read "What is the content? It matched no known document type." --
    which does not say what kind of answer is wanted, and every unclassified
    chunk in a document produced the *same* words, so the reviewer could not tell
    which region they were being asked about.
    """

    def _questions(self, *texts):
        from rag_ingest.docmodel import Element, ElementKind, RawDocument
        from rag_ingest.escalation import ReviewQueue
        from rag_ingest.extractors import TemplateExtractor
        from rag_ingest.pipeline import IngestPipeline

        raw, elements = RawDocument.from_elements(
            "d1",
            [
                Element(
                    element_id=f"e{i}",
                    kind=ElementKind.TABLE if "|" in text else ElementKind.PARAGRAPH,
                    page=1,
                    text=text,
                    table_rows=(("Item", "Amount"), ("Widget", "1200")) if "|" in text else None,
                )
                for i, text in enumerate(texts)
            ],
        )
        queue = ReviewQueue()
        outcome = IngestPipeline(queue, TemplateExtractor(), today=TODAY).ingest(
            raw, elements, owner_id="o"
        )
        for item in outcome.review_items:
            queue.enqueue(item)
        return [item.question for item in queue.open_items()]

    def test_question_states_what_was_observed(self):
        questions = self._questions("Item | Amount\nWidget | 1200")

        assert "looks tabular" in questions[0]

    def test_question_includes_a_snippet_of_the_region(self):
        questions = self._questions("Item | Amount\nWidget | 1200")

        assert "Widget" in questions[0]

    def test_question_asks_for_a_decision_not_a_description(self):
        questions = self._questions("Quarterly report")

        assert "worth keeping" in questions[0]

    def test_two_unclassified_chunks_get_distinguishable_questions(self):
        questions = self._questions(
            "Item | Amount\nWidget | 1200",
            "Quarterly report\nItem Amount Widget 1200",
        )

        assert len(questions) == 2
        assert questions[0] != questions[1]

    def test_question_does_not_say_only_that_it_matched_nothing(self):
        questions = self._questions("Quarterly report")

        assert "What is the content?" not in questions[0]
