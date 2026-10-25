"""Plan section 5: chunk-scoped queue, demand x authority priority, labels."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from rag_ingest.escalation import (
    Correction,
    EscalationReason,
    ReviewerRole,
    ReviewItem,
    ReviewQueue,
)

TODAY = date(2026, 10, 25)


def _item(
    item_id: str,
    *,
    demand: int = 0,
    role: ReviewerRole = ReviewerRole.OWNER,
    created: date = TODAY,
    doc_id: str = "d1",
    reason: EscalationReason = EscalationReason.UNGROUNDED,
) -> ReviewItem:
    return ReviewItem(
        item_id=item_id,
        doc_id=doc_id,
        chunk_id=f"{item_id}-chunk",
        reason=reason,
        question="What should this value be?",
        owner_id="owner-1",
        required_role=role,
        created_on=created,
        retrieval_demand=demand,
    )


def _correction(item: ReviewItem, role: ReviewerRole = ReviewerRole.OWNER) -> Correction:
    return Correction(
        doc_id=item.doc_id,
        chunk_id=item.chunk_id,
        field="total_amount",
        validator_error="mismatch",
        before="1334.56",
        after="1234.56",
        reviewer_id="owner-1",
        reviewer_role=role,
        captured_on=TODAY,
    )


class TestPriority:
    def test_retrieval_demand_outranks_authority(self):
        busy_verifier = _item("a", demand=50, role=ReviewerRole.VERIFIER)
        idle_owner = _item("b", demand=0, role=ReviewerRole.OWNER)

        assert busy_verifier.priority > idle_owner.priority

    def test_next_item_takes_highest_priority(self):
        queue = ReviewQueue()
        queue.enqueue(_item("low", demand=1))
        queue.enqueue(_item("high", demand=99))

        item = queue.next_item()

        assert item is not None
        assert item.item_id == "high"

    def test_empty_queue_returns_none(self):
        assert ReviewQueue().next_item() is None


class TestWipLimit:
    def test_enqueue_refuses_past_the_limit(self):
        queue = ReviewQueue(wip_limit=2)

        assert queue.enqueue(_item("a")) is True
        assert queue.enqueue(_item("b")) is True
        assert queue.enqueue(_item("c")) is False
        assert len(queue.open_items()) == 2

    def test_resolving_frees_capacity(self):
        queue = ReviewQueue(wip_limit=1)
        queue.enqueue(_item("a"))
        queue.resolve("a", _correction(_item("a")))

        assert queue.enqueue(_item("b")) is True

    def test_duplicate_item_is_rejected_loudly(self):
        queue = ReviewQueue()
        queue.enqueue(_item("a"))

        with pytest.raises(ValueError):
            queue.enqueue(_item("a"))


class TestStaleness:
    def test_items_older_than_policy_are_stale(self):
        queue = ReviewQueue(stale_after_days=14)
        queue.enqueue(_item("fresh", created=TODAY))
        queue.enqueue(_item("old", created=TODAY - timedelta(days=20)))

        stale = queue.stale_items(TODAY)

        assert [item.item_id for item in stale] == ["old"]


class TestCorrectionsAsLabels:
    def test_owner_correction_is_a_label(self):
        queue = ReviewQueue()
        item = _item("a")
        queue.enqueue(item)
        queue.resolve("a", _correction(item))

        assert len(queue.labels()) == 1
        assert queue.labels()[0].is_authoritative

    def test_verifier_correction_is_recorded_but_not_a_label(self):
        queue = ReviewQueue()
        item = _item("a", role=ReviewerRole.VERIFIER)
        queue.enqueue(item)
        queue.resolve("a", _correction(item, role=ReviewerRole.VERIFIER))

        assert len(queue.corrections()) == 1
        assert len(queue.labels()) == 0

    def test_correction_for_wrong_document_is_rejected(self):
        queue = ReviewQueue()
        item = _item("a", doc_id="d1")
        queue.enqueue(item)
        foreign = Correction(
            doc_id="d2",
            chunk_id=item.chunk_id,
            field="f",
            validator_error="e",
            before=None,
            after="x",
            reviewer_id="o",
            reviewer_role=ReviewerRole.OWNER,
            captured_on=TODAY,
        )

        with pytest.raises(ValueError):
            queue.resolve("a", foreign)

    def test_unknown_item_raises(self):
        with pytest.raises(KeyError):
            ReviewQueue().resolve("nope", _correction(_item("nope")))


def test_invalid_policy_values_are_rejected():
    with pytest.raises(ValueError):
        ReviewQueue(wip_limit=0)
    with pytest.raises(ValueError):
        ReviewQueue(stale_after_days=0)
