"""Circuit breaker for T2.

``scoring.py`` measures whether T2 is reliable; this module acts on that
measurement. Without it the metric is a report nobody reads at 3am.

**The design constraint that shapes everything here:** there are two signals
with very different latency, and a breaker needs both.

  * **Fast, self-observed** -- incoherent decisions, transport failures,
    unreadable output. Available on the same request, but blind to wrongness: a
    confidently wrong pick is an untroubled success as far as transport knows.
  * **Slow, label-based** -- owner corrections. Authoritative, but arrives hours
    or days later.

A breaker on labels alone keeps making the same wrong pick for the whole
review lag. A breaker on failures alone never notices wrongness at all. So both
feed it, and ``verdict`` names which one tripped.

The breaker wraps any ``Adjudicator`` rather than living inside a backend, so
swapping OpenRouter for a local model does not change the safety behaviour.

States (standard three-state breaker):

    CLOSED     -- pass through; collect signals.
    OPEN       -- every call returns an abstention without invoking the model.
    HALF_OPEN  -- after a cooldown, allow a limited number of probes; a good
                  probe closes the breaker, a bad one re-opens it.

When OPEN the breaker returns a *legitimate* abstention, not an exception -- a
degraded T2 must route to the human queue exactly as an ordinary abstention
does, because that is the safe path the pipeline already knows how to handle.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.adjudication import AdjudicationDecision, AdjudicationQuestion
from rag_ingest.ontology import AbstainReason

#: Below this share of well-formed calls, something is structurally wrong
#: (bad schema, broken model deployment) and continuing to call is pointless.
MIN_HEALTHY_CALL_RATE = 0.6
#: Sliding window of calls used for the fast signal.
CALL_WINDOW = 20
#: Minimum calls before the fast signal is trusted. One transport blip is not a
#: pattern, and tripping on the first failure makes the system flap.
MIN_CALLS_BEFORE_TRIP = 5
#: Below this precision on labelled cases, T2 is misleading humans.
#: Deliberately conservative: a wrong pick points a reviewer at the wrong
#: answer, which is worse than a wasted model call.
MIN_PRECISION = 0.85
#: Minimum labelled cases before the slow signal is trusted.
MIN_LABELS_BEFORE_TRIP = 10
#: Calls allowed while HALF_OPEN. Kept small: this is a probe, not a reopening.
PROBES_AFTER_COOLDOWN = 3


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class TripReason(StrEnum):
    UNHEALTHY_CALLS = "unhealthy_calls"
    LOW_PRECISION = "low_precision"
    NONE = "none"


class CallRecord(BaseModel):
    """One T2 invocation, as the breaker saw it."""

    model_config = ConfigDict(frozen=True)

    #: False when the backend failed, returned unreadable output, or produced an
    #: incoherent decision. Note this says nothing about *correctness*.
    well_formed: bool
    #: True when the decision was a pick (the only outcome with downside risk).
    committed: bool


class BreakerStatus(BaseModel):
    """Observable state, for metrics and for the report."""

    model_config = ConfigDict(frozen=True)

    state: BreakerState
    trip_reason: TripReason
    call_health: float | None
    calls_seen: int
    labels_seen: int
    probes_remaining: int = Field(ge=0)

    @property
    def is_degraded(self) -> bool:
        return self.state is BreakerState.OPEN


class T2CircuitBreaker:
    """Wraps an adjudicator and stops it when it stops being trustworthy."""

    def __init__(
        self,
        inner,
        *,
        min_healthy_call_rate: float = MIN_HEALTHY_CALL_RATE,
        min_precision: float = MIN_PRECISION,
        call_window: int = CALL_WINDOW,
        min_calls_before_trip: int = MIN_CALLS_BEFORE_TRIP,
        min_labels_before_trip: int = MIN_LABELS_BEFORE_TRIP,
        probes_after_cooldown: int = PROBES_AFTER_COOLDOWN,
        cooldown_calls: int = 50,
    ) -> None:
        self._inner = inner
        self._min_healthy = min_healthy_call_rate
        self._min_precision = min_precision
        self._window_size = call_window
        self._min_calls = min_calls_before_trip
        self._min_labels = min_labels_before_trip
        self._probe_budget = probes_after_cooldown
        self._cooldown_calls = cooldown_calls

        self._calls: list[CallRecord] = []
        self._labels_seen = 0
        self._labelled_correct = 0
        self._labelled_wrong = 0

        self._state = BreakerState.CLOSED
        self._trip_reason = TripReason.NONE
        self._probes_remaining = 0
        # Calls that must pass before an OPEN breaker is allowed to probe.
        self._calls_until_probe = 0

    # -- state ------------------------------------------------------------

    @property
    def status(self) -> BreakerStatus:
        return BreakerStatus(
            state=self._state,
            trip_reason=self._trip_reason,
            call_health=self.call_health,
            calls_seen=len(self._calls),
            labels_seen=self._labels_seen,
            probes_remaining=self._probes_remaining,
        )

    @property
    def call_health(self) -> float | None:
        """Share of recent calls that produced a usable decision."""
        if not self._calls:
            return None
        return sum(1 for call in self._calls if call.well_formed) / len(self._calls)

    @property
    def precision(self) -> float | None:
        """Precision on labelled cases; None until a pick has been scored."""
        decided = self._labelled_correct + self._labelled_wrong
        if decided == 0:
            return None
        return self._labelled_correct / decided

    # -- signals ----------------------------------------------------------

    def record_call(self, decision: AdjudicationDecision, *, well_formed: bool) -> None:
        """Feed the fast signal. Called for every T2 invocation."""
        self._calls.append(
            CallRecord(well_formed=well_formed, committed=decision.kind == "pick")
        )
        if len(self._calls) > self._window_size:
            self._calls.pop(0)
        self._evaluate_fast_signal()

    def record_label(self, *, was_correct: bool) -> None:
        """Feed the slow signal, one authoritative scored case at a time."""
        self._labels_seen += 1
        if was_correct:
            self._labelled_correct += 1
        else:
            self._labelled_wrong += 1
        self._evaluate_slow_signal()

    def _evaluate_fast_signal(self) -> None:
        if len(self._calls) < self._min_calls:
            return
        health = self.call_health
        if health is not None and health < self._min_healthy:
            self._trip(TripReason.UNHEALTHY_CALLS)

    def _evaluate_slow_signal(self) -> None:
        decided = self._labelled_correct + self._labelled_wrong
        if decided < self._min_labels:
            return
        precision = self.precision
        if precision is not None and precision < self._min_precision:
            self._trip(TripReason.LOW_PRECISION)

    # -- transitions ------------------------------------------------------

    def _trip(self, reason: TripReason) -> None:
        self._state = BreakerState.OPEN
        self._trip_reason = reason
        self._probes_remaining = 0
        self._calls_until_probe = self._cooldown_calls

    def _abstention(self, rationale: str) -> AdjudicationDecision:
        return AdjudicationDecision(
            kind="abstain",
            reason=AbstainReason.AMBIGUOUS,
            rationale=rationale,
        )

    def reset(self) -> None:
        """Return to CLOSED with fresh windows. For operator intervention."""
        self._state = BreakerState.CLOSED
        self._trip_reason = TripReason.NONE
        self._calls.clear()
        self._probes_remaining = 0
        self._calls_until_probe = 0

    # -- call path --------------------------------------------------------

    def adjudicate(self, question: AdjudicationQuestion) -> AdjudicationDecision:
        return self.adjudicate_with_evidence(question, image_png=None)

    def adjudicate_with_evidence(
        self, question: AdjudicationQuestion, *, image_png: bytes | None
    ) -> AdjudicationDecision:
        """Pass through when closed, abstain when open, probe when half-open."""
        if self._state is BreakerState.OPEN:
            if self._calls_until_probe > 0:
                self._calls_until_probe -= 1
                return self._abstention(
                    f"T2 circuit open ({self._trip_reason.value}); "
                    "routing to human review without calling the model."
                )
            # Cooldown elapsed: allow a limited number of probes.
            self._state = BreakerState.HALF_OPEN
            self._probes_remaining = self._probe_budget

        if self._state is BreakerState.HALF_OPEN and self._probes_remaining <= 0:
            return self._abstention("T2 circuit half-open; probe budget exhausted.")

        was_probationary = self._state is BreakerState.HALF_OPEN
        decision, well_formed = self._invoke(question, image_png)

        if was_probationary:
            # A probe is judged on its own result, never against the window that
            # caused the trip. Appending it to that window would re-trip the
            # breaker on stale evidence and it could never recover.
            self._probes_remaining -= 1
            if well_formed:
                self._state = BreakerState.CLOSED
                self._trip_reason = TripReason.NONE
                self._calls.clear()
                self._probes_remaining = 0
            else:
                self._trip(self._trip_reason)
            return decision

        self.record_call(decision, well_formed=well_formed)
        return decision

    def _invoke(
        self, question: AdjudicationQuestion, image_png: bytes | None
    ) -> tuple[AdjudicationDecision, bool]:
        """Call the inner adjudicator, classifying the result as well-formed or not.

        "Well-formed" means the backend answered coherently. It says nothing
        about correctness -- that is what the slow signal is for.
        """
        with_evidence = getattr(self._inner, "adjudicate_with_evidence", None)
        try:
            if with_evidence is not None:
                decision = with_evidence(question, image_png=image_png)
            else:
                decision = self._inner.adjudicate(question)
        except NotImplementedError:
            return self._abstention("No T2 adjudicator configured."), True
        except Exception as exc:
            return (
                self._abstention(f"T2 failed: {type(exc).__name__}"),
                False,
            )

        well_formed = decision.is_coherent(len(question.candidates))
        if not well_formed:
            return (
                self._abstention(f"T2 returned an incoherent decision: {decision.kind}"),
                False,
            )
        return decision, True


__all__ = [
    "MIN_HEALTHY_CALL_RATE",
    "MIN_PRECISION",
    "BreakerState",
    "BreakerStatus",
    "T2CircuitBreaker",
    "TripReason",
]
