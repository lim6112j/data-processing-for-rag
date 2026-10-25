"""Can we say whether T2 helped? These tests pin the answer's shape.

The distinction that matters: abstaining is not the same as being wrong, and a
single accuracy number hides that.
"""

from __future__ import annotations

from datetime import date

from rag_ingest.escalation import (
    Correction,
    EscalationReason,
    ReviewerRole,
    ReviewItem,
)
from rag_ingest.scoring import Outcome, score

TODAY = date(2026, 10, 25)


def _item(
    *,
    chunk_id: str = "c1",
    candidates: tuple[str, ...] = ("c1:docling", "c1:pdfplumber"),
    choice: str | None = None,
) -> ReviewItem:
    return ReviewItem(
        item_id=f"i-{chunk_id}",
        doc_id="d1",
        chunk_id=chunk_id,
        reason=EscalationReason.STRUCTURE_AMBIGUOUS,
        question="q",
        owner_id="o",
        created_on=TODAY,
        candidate_ids=candidates,
        adjudicated_choice=choice,
    )


def _correction(
    *,
    chunk_id: str = "c1",
    after: str,
    role: ReviewerRole = ReviewerRole.OWNER,
    choice: str | None = None,
) -> Correction:
    return Correction(
        doc_id="d1",
        chunk_id=chunk_id,
        field="table_structure",
        validator_error="structure",
        before=None,
        after=after,
        reviewer_id="r",
        reviewer_role=role,
        captured_on=TODAY,
        adjudication_choice=choice,
    )


def test_t2_pick_matching_owner_answer_is_correct():
    card = score(
        [_correction(after="c1:pdfplumber")],
        {"c1": _item(choice="c1:pdfplumber")},
    )

    assert card.scored[0].outcome is Outcome.CORRECT
    assert card.precision == 1.0


def test_t2_pick_contradicting_owner_answer_is_incorrect():
    card = score(
        [_correction(after="c1:docling")],
        {"c1": _item(choice="c1:pdfplumber")},
    )

    assert card.scored[0].outcome is Outcome.INCORRECT
    assert card.precision == 0.0
    assert card.harm_rate == 1.0


def test_abstention_is_not_counted_as_an_error():
    # T2 said nothing. That costs a call but never misleads.
    card = score([_correction(after="c1:docling")], {"c1": _item(choice=None)})

    assert card.scored[0].outcome is Outcome.NO_PICK
    assert card.precision is None  # undefined, not 0%
    assert card.harm_rate == 0.0
    assert card.abstentions == 1


def test_verifier_corrections_are_excluded():
    card = score(
        [_correction(after="c1:docling", role=ReviewerRole.VERIFIER)],
        {"c1": _item(choice="c1:pdfplumber")},
    )

    assert card.scored == ()
    assert card.excluded == 1


def test_unmatched_chunk_is_excluded():
    card = score([_correction(after="c1:docling")], {})

    assert card.scored == ()
    assert card.excluded == 1


def test_owner_answer_that_names_no_candidate_is_unscorable():
    # Guessing "it sounds like X" would make the metric flatter itself.
    card = score(
        [_correction(after="the table has 4 columns")],
        {"c1": _item(choice="c1:pdfplumber")},
    )

    assert card.scored[0].outcome is Outcome.UNSCORABLE


def test_item_without_candidate_set_is_excluded():
    card = score(
        [_correction(after="c1:docling")],
        {"c1": _item(candidates=(), choice="c1:docling")},
    )

    assert card.scored == ()
    assert card.excluded == 1


class TestAggregates:
    def test_precision_over_multiple_cases(self):
        items = {
            "a": _item(chunk_id="a", candidates=("a:x", "a:y"), choice="a:x"),
            "b": _item(chunk_id="b", candidates=("b:x", "b:y"), choice="b:x"),
            "c": _item(chunk_id="c", candidates=("c:x", "c:y"), choice="c:y"),
        }
        corrections = [
            _correction(chunk_id="a", after="a:x"),
            _correction(chunk_id="b", after="b:x"),
            _correction(chunk_id="c", after="c:x"),  # T2 chose c:y -> wrong
        ]

        card = score(corrections, items)

        assert card.precision == 2 / 3
        assert card.coverage == 1.0
        assert card.incorrect == 1

    def test_coverage_distinguishes_silent_from_wrong(self):
        items = {
            "a": _item(chunk_id="a", candidates=("a:x", "a:y"), choice=None),
            "b": _item(chunk_id="b", candidates=("b:x", "b:y"), choice="b:x"),
        }
        card = score(
            [_correction(chunk_id="a", after="a:x"), _correction(chunk_id="b", after="b:x")],
            items,
        )

        assert card.coverage == 0.5
        assert card.precision == 1.0

    def test_empty_set_does_not_divide_by_zero(self):
        card = score([], {})

        assert card.total_scored == 0
        assert card.coverage == 0.0
        assert card.harm_rate == 0.0
        assert card.precision is None


class TestVerdict:
    def test_no_data_is_stated_plainly(self):
        assert score([], {}).verdict() == "no labelled cases yet"

    def test_never_picking_is_described_as_no_risk(self):
        card = score([_correction(after="c1:docling")], {"c1": _item(choice=None)})

        assert "never picks" in card.verdict()

    def test_reliable_precision_is_acknowledged(self):
        card = score([_correction(after="c1:pdfplumber")], {"c1": _item(choice="c1:pdfplumber")})

        assert "reliable" in card.verdict()

    def test_unreliable_precision_recommends_abstention(self):
        card = score([_correction(after="c1:docling")], {"c1": _item(choice="c1:pdfplumber")})

        verdict = card.verdict()
        assert "below" in verdict
        assert "abstention" in verdict


class TestQueueAnnotation:
    def test_resolve_attaches_the_t2_choice_to_the_label(self):
        # The reviewer should never have to restate what T2 chose.
        from rag_ingest.escalation import ReviewQueue

        queue = ReviewQueue()
        item = _item(choice="c1:pdfplumber")
        queue.enqueue(item)
        queue.resolve(item.item_id, _correction(after="c1:pdfplumber"))

        assert queue.corrections()[0].adjudication_choice == "c1:pdfplumber"

    def test_explicit_choice_is_not_overwritten(self):
        from rag_ingest.escalation import ReviewQueue

        queue = ReviewQueue()
        item = _item(choice="c1:pdfplumber")
        queue.enqueue(item)
        queue.resolve(
            item.item_id,
            _correction(after="c1:docling", choice="c1:docling"),
        )

        assert queue.corrections()[0].adjudication_choice == "c1:docling"


class TestExportRoundTrip:
    def test_queue_export_feeds_the_scorer(self):
        # The chain that makes the loop real: resolve -> export -> score.
        from rag_ingest.escalation import ReviewQueue

        queue = ReviewQueue()
        item = _item(choice="c1:pdfplumber")
        queue.enqueue(item)
        queue.resolve(item.item_id, _correction(after="c1:pdfplumber"))

        payload = queue.export()

        assert payload["corrections"][0]["adjudication_choice"] == "c1:pdfplumber"
        assert payload["review_items"][0]["candidate_ids"]

    def test_example_label_file_scores(self, tmp_path, capsys):
        from pathlib import Path

        from rag_ingest.cli import main

        path = Path(__file__).parent.parent / "goldset" / "t2_labels.example.json"
        assert path.exists()

        exit_code = main(["--score", str(path)])

        out = capsys.readouterr().out
        assert exit_code == 0
        assert "precision" in out
        # The example deliberately contains one wrong pick.
        assert "below" in out
