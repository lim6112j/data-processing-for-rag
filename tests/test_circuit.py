"""Circuit breaker behaviour, including the limits of each signal.

The central property: a confidently wrong pick is invisible to the fast signal.
Only owner labels catch it. These tests pin that, because the whole two-signal
design exists to cover it.
"""

from __future__ import annotations

from rag_ingest.adjudication import (
    AdjudicationDecision,
    Candidate,
    DisagreementKind,
    build_question,
)
from rag_ingest.circuit import BreakerState, T2CircuitBreaker, TripReason
from rag_ingest.docmodel import ProvenanceRef
from rag_ingest.scoring import Outcome, ScoredCorrection, T2ScoreCard, feed_breaker

BBOX = (10.0, 20.0, 300.0, 60.0)


def _question():
    return build_question(
        chunk_id="c1",
        doc_id="d1",
        kind=DisagreementKind.COLUMN_COUNT,
        prompt="How many columns?",
        candidates=[
            Candidate(candidate_id="c1:a", produced_by="docling", proposed_text="4"),
            Candidate(candidate_id="c1:b", produced_by="pdfplumber", proposed_text="5"),
        ],
        provenance=ProvenanceRef(page=1, char_start=0, char_end=5, bbox=BBOX),
    )


class _Backend:
    """Scripted adjudicator: returns queued decisions, or raises."""

    def __init__(self, decisions=None, raises: Exception | None = None) -> None:
        self._decisions = list(decisions or [])
        self._raises = raises
        self.calls = 0

    def adjudicate_with_evidence(self, question, *, image_png):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        if self._decisions:
            return self._decisions.pop(0)
        return AdjudicationDecision(kind="pick", choice_index=0)


def _pick() -> AdjudicationDecision:
    return AdjudicationDecision(kind="pick", choice_index=0)


def _abstain() -> AdjudicationDecision:
    return AdjudicationDecision(kind="abstain", reason="ambiguous")


class TestClosedState:
    def test_healthy_calls_pass_through(self):
        backend = _Backend()
        breaker = T2CircuitBreaker(backend)

        decision = breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")

        assert decision.kind == "pick"
        assert breaker.status.state is BreakerState.CLOSED

    def test_wrong_pick_is_invisible_to_the_fast_signal(self):
        # The critical limitation. A confidently wrong pick is a well-formed
        # decision, so call health stays perfect and the breaker stays closed.
        backend = _Backend(decisions=[_pick()] * 20)
        breaker = T2CircuitBreaker(backend)

        for _ in range(20):
            breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")

        assert breaker.status.state is BreakerState.CLOSED
        assert breaker.status.call_health == 1.0
        assert breaker.precision is None  # nothing labelled yet


class TestFastSignal:
    def test_repeated_backend_failure_opens_the_breaker(self):
        backend = _Backend(raises=TimeoutError("down"))
        breaker = T2CircuitBreaker(backend, min_calls_before_trip=5)

        for _ in range(5):
            breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")

        assert breaker.status.state is BreakerState.OPEN
        assert breaker.status.trip_reason is TripReason.UNHEALTHY_CALLS

    def test_a_single_failure_does_not_trip(self):
        # One transport blip is not a pattern; tripping on it makes the system flap.
        backend = _Backend(raises=TimeoutError("blip"))
        breaker = T2CircuitBreaker(backend, min_calls_before_trip=5)

        breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")

        assert breaker.status.state is BreakerState.CLOSED

    def test_failures_are_returned_as_abstentions_not_exceptions(self):
        backend = _Backend(raises=TimeoutError("down"))
        breaker = T2CircuitBreaker(backend)

        decision = breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")

        assert decision.kind == "abstain"
        assert "TimeoutError" in (decision.rationale or "")

    def test_incoherent_decision_counts_as_unhealthy(self):
        # An out-of-range index is a broken backend, not a judgement.
        backend = _Backend(decisions=[AdjudicationDecision(kind="pick", choice_index=9)] * 5)
        breaker = T2CircuitBreaker(backend, min_calls_before_trip=5)

        for _ in range(5):
            decision = breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")
            assert decision.kind == "abstain"

        assert breaker.status.state is BreakerState.OPEN


class TestSlowSignal:
    def _card(self, correct: int, incorrect: int) -> T2ScoreCard:
        scored = tuple(
            [
                ScoredCorrection(chunk_id=f"c{i}", field="f", outcome=Outcome.CORRECT)
                for i in range(correct)
            ]
            + [
                ScoredCorrection(chunk_id=f"w{i}", field="f", outcome=Outcome.INCORRECT)
                for i in range(incorrect)
            ]
        )
        return T2ScoreCard(scored=scored)

    def test_low_precision_opens_the_breaker(self):
        breaker = T2CircuitBreaker(_Backend(), min_labels_before_trip=10)

        feed_breaker(self._card(correct=5, incorrect=5), breaker)

        assert breaker.status.state is BreakerState.OPEN
        assert breaker.status.trip_reason is TripReason.LOW_PRECISION

    def test_high_precision_keeps_it_closed(self):
        breaker = T2CircuitBreaker(_Backend(), min_labels_before_trip=10)

        feed_breaker(self._card(correct=10, incorrect=0), breaker)

        assert breaker.status.state is BreakerState.CLOSED

    def test_too_few_labels_does_not_trip(self):
        breaker = T2CircuitBreaker(_Backend(), min_labels_before_trip=10)

        feed_breaker(self._card(correct=1, incorrect=4), breaker)

        assert breaker.status.state is BreakerState.CLOSED

    def test_abstentions_do_not_dilute_precision(self):
        # Abstaining is not being wrong. Feeding NO_PICK in would make a silent
        # but accurate T2 look bad.
        card = T2ScoreCard(
            scored=(ScoredCorrection(chunk_id="c", field="f", outcome=Outcome.NO_PICK),)
            * 20
        )
        breaker = T2CircuitBreaker(_Backend(), min_labels_before_trip=10)

        feed_breaker(card, breaker)

        assert breaker.status.state is BreakerState.CLOSED
        assert breaker.precision is None


class TestOpenState:
    def test_open_breaker_does_not_call_the_backend(self):
        backend = _Backend(raises=TimeoutError("down"))
        breaker = T2CircuitBreaker(backend, min_calls_before_trip=5, cooldown_calls=10)
        for _ in range(5):
            breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")
        calls_at_trip = backend.calls

        decision = breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")

        assert decision.kind == "abstain"
        assert backend.calls == calls_at_trip
        assert "circuit open" in (decision.rationale or "")

    def test_open_breaker_still_routes_to_the_queue(self):
        # Degraded T2 must produce a legitimate abstention, so the pipeline's
        # existing human path handles it with no special casing.
        breaker = T2CircuitBreaker(_Backend(raises=TimeoutError()), min_calls_before_trip=3)
        for _ in range(3):
            breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")

        decision = breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")

        assert decision.is_coherent(2)


class TestHalfOpen:
    def test_cooldown_allows_a_probe_and_a_good_probe_closes(self):
        backend = _Backend(decisions=[_pick()] * 5)
        breaker = T2CircuitBreaker(
            _Backend(decisions=[AdjudicationDecision(kind="pick", choice_index=9)] * 3),
            min_calls_before_trip=3,
            cooldown_calls=2,
        )
        for _ in range(3):
            breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")
        assert breaker.status.state is BreakerState.OPEN

        # Burn the cooldown.
        for _ in range(2):
            breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")
        assert breaker.status.state is BreakerState.OPEN

        # Swap in a healthy backend, then probe.
        breaker._inner = backend
        decision = breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")

        assert decision.kind == "pick"
        assert breaker.status.state is BreakerState.CLOSED

    def test_failed_probe_reopens(self):
        breaker = T2CircuitBreaker(
            _Backend(decisions=[AdjudicationDecision(kind="pick", choice_index=9)] * 10),
            min_calls_before_trip=3,
            cooldown_calls=1,
        )
        for _ in range(3):
            breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")

        breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")  # cooldown
        breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")  # probe, bad

        assert breaker.status.state is BreakerState.OPEN


class TestOperatorReset:
    def test_reset_returns_to_closed_with_clean_windows(self):
        breaker = T2CircuitBreaker(_Backend(raises=TimeoutError()), min_calls_before_trip=3)
        for _ in range(3):
            breaker.adjudicate_with_evidence(_question(), image_png=b"PNG")

        breaker.reset()

        assert breaker.status.state is BreakerState.CLOSED
        assert breaker.status.calls_seen == 0
        assert breaker.status.trip_reason is TripReason.NONE


class TestPassthroughContract:
    def test_inner_without_evidence_method_still_works(self):
        # The breaker must wrap the plain Adjudicator interface too.
        class _Plain:
            def adjudicate(self, question):
                return AdjudicationDecision(kind="pick", choice_index=1)

        breaker = T2CircuitBreaker(_Plain())

        decision = breaker.adjudicate(_question())

        assert decision.kind == "pick"
        assert decision.choice_index == 1

    def test_unimplemented_adjudicator_does_not_trip_the_breaker(self):
        # The stub raising NotImplementedError is configuration, not failure.
        breaker = T2CircuitBreaker(object())

        decision = breaker.adjudicate(_question())

        assert decision.kind == "abstain"
        assert breaker.status.state is BreakerState.CLOSED
