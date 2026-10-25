"""Quarantine and the review queue.

Plan §5. Escalation is a first-class system state, not an error log. Three
things the naive "fire a correction request" framing misses and this module
implements:

  * granularity is the *chunk*, not the document;
  * priority is retrieval demand x authority, not FIFO;
  * a correction is captured as a machine-readable label, or the human tier
    stays a permanent cost centre.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.grounding import Issue


class EscalationReason(StrEnum):
    """Closed enum. Without it the queue is not triageable."""

    EXTRACTION_FAILED = "extraction_failed"
    STRUCTURE_AMBIGUOUS = "structure_ambiguous"
    SCHEMA_MISMATCH = "schema_mismatch"
    LOW_CONFIDENCE = "low_confidence"
    #: Values were grounded but too few were found to describe the chunk -- most
    #: often OCR damage hiding a field, or a template that no longer matches.
    INCOMPLETE_EXTRACTION = "incomplete_extraction"
    CONFLICT_WITH_EXISTING = "conflict_with_existing"
    UNCLASSIFIED = "unclassified"
    #: The value was claimed but is not present in the source it cites.
    UNGROUNDED = "ungrounded"


class ReviewerRole(StrEnum):
    #: The document's owner. Only their answer is authoritative.
    OWNER = "owner"
    #: Anyone else. A verifier, not a source of truth.
    VERIFIER = "verifier"


class ReviewItem(BaseModel):
    """One unit of human work. Scoped to a chunk."""

    model_config = ConfigDict(frozen=True)

    item_id: str
    doc_id: str
    chunk_id: str
    reason: EscalationReason
    issues: tuple[Issue, ...] = Field(default_factory=tuple)
    #: A specific, closed question -- "column 3 is unlabelled, what does it
    #: mean?" gets answered; "please re-upload correctly" does not.
    question: str
    owner_id: str
    required_role: ReviewerRole = ReviewerRole.OWNER
    created_on: date
    #: Expected query traffic for this document; 0 means it is not worth
    #: reviewing at all.
    retrieval_demand: int = Field(default=0, ge=0)
    #: Candidate ids, in the order presented to T2. Preserved so a later
    #: correction can be scored against what T2 actually chose.
    candidate_ids: tuple[str, ...] = Field(default_factory=tuple)
    #: The candidate id T2 selected, when it picked one. Never a verification --
    #: only a narrowing of the question.
    adjudicated_choice: str | None = None
    resolved: bool = False

    @property
    def priority(self) -> int:
        """Higher is more urgent.

        Demand dominates: reviewing a document nobody queries is wasted
        effort, and documents with zero demand should not occupy the queue.
        """
        return self.retrieval_demand * 10 + (5 if self.required_role is ReviewerRole.OWNER else 0)


class Correction(BaseModel):
    """A reviewer's answer, captured as a label.

    Plan §5c: this is the training/eval set, and the only mechanism by which
    tiers 0-2 ever improve.

    ``adjudication_choice`` records which candidate T2 picked, when T2 had
    picked one. That single field is what makes T2 measurable: comparing it
    against the owner's answer turns "we added a model" into "the model was
    right 82% of the time", which is the only form of that claim that can be
    acted on.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    chunk_id: str
    field: str
    validator_error: str
    before: str | None
    after: str
    reviewer_id: str
    reviewer_role: ReviewerRole
    captured_on: date
    #: The candidate id T2 selected, if it selected one for this chunk.
    adjudication_choice: str | None = None
    #: The candidates that were on offer. Stored on the label rather than looked
    #: up from the review item: items are discarded when a document is
    #: re-ingested, and a label that cannot be scored without them is not a
    #: label. This is what makes the eval set durable.
    candidate_ids: tuple[str, ...] = ()

    @property
    def is_authoritative(self) -> bool:
        return self.reviewer_role is ReviewerRole.OWNER


class ReviewQueue:
    """In-memory queue with a WIP limit and a staleness policy.

    Deliberately not a database. The policies are the substance; persisting
    them is a deployment detail.
    """

    def __init__(self, wip_limit: int = 25, stale_after_days: int = 14) -> None:
        if wip_limit <= 0:
            raise ValueError("wip_limit must be positive")
        if stale_after_days <= 0:
            raise ValueError("stale_after_days must be positive")
        self._items: dict[str, ReviewItem] = {}
        self._corrections: list[Correction] = []
        self._wip_limit = wip_limit
        self._stale_after_days = stale_after_days

    @property
    def wip_limit(self) -> int:
        return self._wip_limit

    def enqueue(self, item: ReviewItem) -> bool:
        """Add an item. Returns False when the WIP limit is reached.

        A hard cap is the point: an unbounded queue silently becomes a
        graveyard and the non-retrievable corpus grows without bound.
        """
        if item.item_id in self._items:
            raise ValueError(f"duplicate review item {item.item_id}")
        if len(self.open_items()) >= self._wip_limit:
            return False
        self._items[item.item_id] = item
        return True

    def discard(self, doc_id: str) -> int:
        """Drop every item for a document. Returns how many.

        Used when a document is re-ingested. Item ids are derived from
        doc_id + chunk id, so a second parse produces the *same* ids and would
        otherwise trip the duplicate guard -- that was a real 500 on re-upload.

        Resolved items go too, because the item objects describe a parse that no
        longer exists. Their **corrections are kept**: those are the labels, and
        `Correction` records both T2's pick and the owner's answer, so the label
        stays scorable without the item.
        """
        stale = [
            item_id for item_id, item in self._items.items() if item.doc_id == doc_id
        ]
        for item_id in stale:
            del self._items[item_id]
        return len(stale)

    def open_items(self) -> list[ReviewItem]:
        return [item for item in self._items.values() if not item.resolved]

    def next_item(self) -> ReviewItem | None:
        open_items = self.open_items()
        if not open_items:
            return None
        # Tie-break on item_id so ordering is deterministic in tests.
        return max(open_items, key=lambda item: (item.priority, item.item_id))

    def resolve(self, item_id: str, correction: Correction) -> None:
        """Record a reviewer's answer.

        When the item had a T2 pick, the correction is annotated with it so the
        label set can later answer "was T2 right?". Doing this here means the
        reviewer never has to restate what the model chose -- and cannot
        accidentally mis-transcribe it.
        """
        item = self._items.get(item_id)
        if item is None:
            raise KeyError(f"unknown review item {item_id}")
        if correction.doc_id != item.doc_id:
            raise ValueError("correction does not belong to this review item's document")
        if correction.chunk_id != item.chunk_id:
            raise ValueError("correction does not belong to this review item's chunk")

        annotate: dict[str, object] = {}
        if correction.adjudication_choice is None and item.adjudicated_choice is not None:
            annotate["adjudication_choice"] = item.adjudicated_choice
        if not correction.candidate_ids and item.candidate_ids:
            annotate["candidate_ids"] = item.candidate_ids
        if annotate:
            correction = correction.model_copy(update=annotate)

        self._items[item_id] = item.model_copy(update={"resolved": True})
        self._corrections.append(correction)

    def items_by_chunk(self) -> dict[str, ReviewItem]:
        """Chunk id -> item, for joining labels to adjudications when scoring."""
        return {item.chunk_id: item for item in self._items.values()}

    def export(self) -> dict[str, list[dict[str, object]]]:
        """Serializable snapshot: corrections joined with their review items.

        This is the artifact ``rag-goldset --score`` consumes. Emitting it as
        plain JSON keeps the label set portable -- it can be version-controlled,
        diffed, and scored without the queue.
        """
        return {
            "corrections": [
                correction.model_dump(mode="json") for correction in self._corrections
            ],
            "review_items": [
                item.model_dump(mode="json") for item in self._items.values()
            ],
        }

    def stale_items(self, today: date) -> list[ReviewItem]:
        return [
            item
            for item in self.open_items()
            if (today - item.created_on).days >= self._stale_after_days
        ]

    def corrections(self) -> tuple[Correction, ...]:
        return tuple(self._corrections)

    def labels(self) -> tuple[Correction, ...]:
        """Only owner corrections are labels; verifier guesses are not truth."""
        return tuple(c for c in self._corrections if c.is_authoritative)
