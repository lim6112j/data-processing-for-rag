"""Deterministic semantic checks: the values agree with *each other*.

survey.md conclusion 1, ranked first among missing pieces: the pipeline verifies
"the value is in the source" but never "the values are consistent". Both an
implausible date (``2026-13-45``) and a self-consistent OCR error pass grounding
by construction -- the string IS verbatim in the span. These checks are cheap,
model-free, and catch a class of error the grounding gate cannot see.

Placement: checks run AFTER the grounding gate, and only on what the gate
admitted. A value must be provably present in the source before asking whether
it is plausible; otherwise a grounding failure would be reported as two
unrelated problems and the review queue would ask a double-barrelled question.

Failure is expressed as data (``CheckFailure``), never an exception -- same rule
as the gate's ``GroundingStatus``. The pipeline decides what a failure means for
admission; this module only establishes that the observation exists.

A check that cannot parse a value stays silent. Suspicion without a parse is
silence, not a finding: flagging "May 96" as a bad date would be the check
itself hallucinating structure. Same discipline as ``_numeric_key`` in the
gate: a wrong escalation is worse than a missed one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, ConfigDict

from rag_ingest.ontology import (
    Extraction,
    FieldAbstention,
    FieldValue,
    InvoiceHeader,
    ReportSection,
)

#: Newest issue date the ontology's seed corpus could legitimately carry.
#: Exists so an OCR-mangled year (e.g. 92026-01-01) fails as implausible
#: rather than passing as "far future". Not a tunable knob to stretch when the
#: queue gets busy -- it is a plausibility horizon, and a document legitimately
#: dated beyond it becomes a review question, which is the correct outcome.
MAX_REASONABLE_YEAR = 2100


class CheckFailure(BaseModel):
    """One deterministic check that disagreed with the extraction."""

    model_config = ConfigDict(frozen=True)

    field: str
    check_name: str
    message: str


@dataclass(frozen=True)
class _FieldObservation:
    """A named field value worth checking, abstentions included."""

    field: str
    value: FieldValue | FieldAbstention


def _observations(extraction: Extraction) -> tuple[_FieldObservation, ...]:
    if isinstance(extraction, InvoiceHeader):
        return tuple(
            _FieldObservation(field=name, value=field)
            for name, field in (
                ("invoice_number", extraction.invoice_number),
                ("issue_date", extraction.issue_date),
                ("total_amount", extraction.total_amount),
                ("currency", extraction.currency),
            )
        )
    if isinstance(extraction, ReportSection):
        return tuple(
            _FieldObservation(field=name, value=field)
            for name, field in (
                ("section_title", extraction.section_title),
                ("primary_finding", extraction.primary_finding),
            )
        )
    return ()


def _has_iso_shape(raw: str) -> bool:
    """True when a value claims the exact ISO shape the extractor emits."""
    return (
        len(raw) == 10
        and raw[4] == "-"
        and raw[7] == "-"
        and raw[:4].isdigit()
        and raw[5:7].isdigit()
        and raw[8:10].isdigit()
    )


def _parse_iso_date(raw: str) -> date | None:
    """Strict ISO YYYY-MM-DD, or None.

    ``date.fromisoformat`` accepts shapes the extractor never emits (``20260101``),
    and Python >= 3.11 accepts some non-zero-padded forms. The extractor's
    ``_DATE`` pattern guarantees ``\\d{4}-\\d{2}-\\d{2}``, so the check insists on
    exactly that shape too -- a looser parser here would turn an unparseable
    value into a finding the extraction never claimed.
    """
    if len(raw) != 10 or raw[4] != "-" or raw[7] != "-":
        return None
    if not (raw[:4].isdigit() and raw[5:7].isdigit() and raw[8:10].isdigit()):
        return None
    try:
        return date(int(raw[:4]), int(raw[5:7]), int(raw[8:10]))
    except ValueError:
        return None


def _check_date_plausibility(
    observation: _FieldObservation,
) -> CheckFailure | None:
    field = observation.field
    value = observation.value
    if isinstance(value, FieldAbstention):
        return None
    if not _has_iso_shape(value.value):
        # A value that does not claim to be an ISO date is not evidence of a
        # bad date. Flagging "May 96" would be the check hallucinating
        # structure the extraction never claimed.
        return None
    if _parse_iso_date(value.value) is None:
        # ISO shape, impossible calendar values (2026-13-45, 2026-02-30,
        # 1900-02-29). This is precisely the OCR-damage signature: the digits
        # are verbatim in the source, so grounding passes; only the calendar
        # can see that they cannot mean a day.
        return CheckFailure(
            field=field,
            check_name="date_plausibility",
            message=f"{value.value} has ISO shape but is not a real date",
        )
    parsed = _parse_iso_date(value.value)
    if parsed.year > MAX_REASONABLE_YEAR:
        return CheckFailure(
            field=field,
            check_name="date_plausibility",
            message=f"{value.value} is beyond the plausibility horizon "
            f"({MAX_REASONABLE_YEAR})",
        )
    return None


def check_extraction(extraction: Extraction) -> tuple[CheckFailure, ...]:
    """Every deterministic check that disagrees with this extraction.

    Returns *observations*, not a verdict: the caller decides what a failure
    means for admission. Runs after grounding, on gate-admitted extractions.
    """
    failures: list[CheckFailure] = []
    for observation in _observations(extraction):
        failure = _check_date_plausibility(observation)
        if failure is not None:
            failures.append(failure)
    return tuple(failures)


__all__ = ["MAX_REASONABLE_YEAR", "CheckFailure", "check_extraction"]
