"""Does T2 actually help? Answer it with numbers, not with intuition.

Plan §5c says corrections must become machine-readable labels. This module
closes that loop: it scores T2's picks against authoritative (owner) corrections
and reports the three rates that decide whether T2 earns its cost.

The three outcomes are not equally bad, which is why they are counted
separately:

  * **correct** -- T2 picked the reading the owner confirmed. Saves work.
  * **incorrect** -- T2 picked a reading the owner rejected. The dangerous one:
    a confident wrong narrowing is worse than no narrowing, because it points
    the reviewer at the wrong answer.
  * **no pick** -- T2 abstained or escalated. Costs a model call but never
    misleads. This is the desired behaviour when T2 cannot see the evidence.

A single "accuracy" number hides that distinction. A system that abstains
often and is never wrong is healthier than one that always picks and is right
80% of the time, so both are reported.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.escalation import Correction


class Outcome(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    NO_PICK = "no_pick"
    #: The correction cannot be scored: no candidate set was recorded, or the
    #: choice T2 made is not among the candidates.
    UNSCORABLE = "unscorable"


class ScoredCorrection(BaseModel):
    """One labelled case, joined with what T2 did on it."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    field: str
    outcome: Outcome
    adjudicated_choice: str | None = None
    owner_choice: str | None = None
    #: True when the owner's answer matched the reading T2 picked.
    detail: str | None = None


class T2ScoreCard(BaseModel):
    """Aggregate health of T2 over a labelled set."""

    model_config = ConfigDict(frozen=True)

    scored: tuple[ScoredCorrection, ...] = Field(default_factory=tuple)
    #: Corrections skipped because they were not authoritative (verifier guesses
    #: are not ground truth) or could not be joined to an adjudication.
    excluded: int = Field(default=0, ge=0)

    def _count(self, outcome: Outcome) -> int:
        return sum(1 for item in self.scored if item.outcome is outcome)

    @property
    def picks(self) -> int:
        return self._count(Outcome.CORRECT) + self._count(Outcome.INCORRECT)

    @property
    def abstentions(self) -> int:
        return self._count(Outcome.NO_PICK)

    @property
    def total_scored(self) -> int:
        return len(self.scored)

    @property
    def precision(self) -> float | None:
        """Of the times T2 picked, how often was it right.

        None when T2 never picked -- which is not 0% and must not be reported as
        such; it means the metric is undefined for this set.
        """
        if self.picks == 0:
            return None
        return self._count(Outcome.CORRECT) / self.picks

    @property
    def coverage(self) -> float:
        """Fraction of cases where T2 committed to an answer at all."""
        if self.total_scored == 0:
            return 0.0
        return self.picks / self.total_scored

    @property
    def incorrect(self) -> int:
        return self._count(Outcome.INCORRECT)

    @property
    def harm_rate(self) -> float:
        """Fraction of all cases where T2 actively pointed the wrong way.

        The number that matters most: a wrong pick is a misleading instruction
        to a human, not merely a wasted call.
        """
        if self.total_scored == 0:
            return 0.0
        return self.incorrect / self.total_scored

    def verdict(self, *, min_correct_precision: float = 0.85) -> str:
        """A one-line read on whether T2 is earning its keep.

        Deliberately conservative: T2 should either be reliable when it commits
        or quiet. The failure mode to reject is "confident and sometimes wrong",
        because the human queue cannot distinguish those cases from correct ones.
        """
        if self.total_scored == 0:
            return "no labelled cases yet"
        precision = self.precision
        if precision is None:
            return "T2 never picks: no risk, no savings"
        if precision >= min_correct_precision:
            return "T2 is reliable on this set"
        return (
            f"T2 precision {precision:.0%} is below {min_correct_precision:.0%}: "
            "prefer abstention over picking until it improves"
        )


def _owner_choice(correction: Correction, candidate_ids: tuple[str, ...]) -> str | None:
    """Map an owner's written answer onto a candidate id.

    The reviewer states the correct *reading*, so it is matched against the
    candidate ids in order. A leading candidate id (``"d1:table:e1:alternate"``)
    is honoured directly, which lets a review UI submit the id it displayed.
    """
    text = correction.after.strip()
    if text in candidate_ids:
        return text
    return None


def feed_breaker(card: T2ScoreCard, breaker) -> None:
    """Push this card's outcomes into a circuit breaker.

    Only *picks* carry information about correctness: an abstention cannot be
    right or wrong, so feeding it in would dilute precision with cases the
    breaker cannot learn from.
    """
    for scored in card.scored:
        if scored.outcome is Outcome.CORRECT:
            breaker.record_label(was_correct=True)
        elif scored.outcome is Outcome.INCORRECT:
            breaker.record_label(was_correct=False)


def score(
    corrections: list[Correction],
    review_items: dict[str, object],
) -> T2ScoreCard:
    """Join corrections to adjudications and score them.

    ``review_items`` maps chunk_id -> ReviewItem. Only authoritative owner
    corrections count: a verifier's guess is not ground truth, and scoring
    against it would measure agreement with a bystander.
    """
    scored: list[ScoredCorrection] = []
    excluded = 0

    for correction in corrections:
        if not correction.is_authoritative:
            excluded += 1
            continue

        item = review_items.get(correction.chunk_id)
        # Prefer what the label carries. The review item is a fallback for
        # corrections captured before labels were self-contained, and may be
        # gone entirely after a re-ingest.
        candidate_ids = correction.candidate_ids or tuple(
            getattr(item, "candidate_ids", ()) or ()
        )
        adjudicated = correction.adjudication_choice or getattr(
            item, "adjudicated_choice", None
        )

        if not candidate_ids:
            excluded += 1
            continue

        owner_choice = _owner_choice(correction, candidate_ids)

        if adjudicated is None:
            scored.append(
                ScoredCorrection(
                    chunk_id=correction.chunk_id,
                    field=correction.field,
                    outcome=Outcome.NO_PICK,
                    owner_choice=owner_choice,
                )
            )
            continue

        if owner_choice is None:
            # The owner's answer did not name a candidate. Without a ground-truth
            # choice there is nothing to compare T2 against -- and guessing that
            # the answer "sounds like" one candidate is how scoring systems start
            # flattering themselves.
            scored.append(
                ScoredCorrection(
                    chunk_id=correction.chunk_id,
                    field=correction.field,
                    outcome=Outcome.UNSCORABLE,
                    adjudicated_choice=adjudicated,
                    detail="owner answer did not name a candidate",
                )
            )
            continue

        outcome = Outcome.CORRECT if adjudicated == owner_choice else Outcome.INCORRECT
        scored.append(
            ScoredCorrection(
                chunk_id=correction.chunk_id,
                field=correction.field,
                outcome=outcome,
                adjudicated_choice=adjudicated,
                owner_choice=owner_choice,
            )
        )

    return T2ScoreCard(scored=tuple(scored), excluded=excluded)


__all__ = [
    "Outcome",
    "ScoredCorrection",
    "T2ScoreCard",
    "feed_breaker",
    "score",
]
