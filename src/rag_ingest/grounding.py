"""The grounding gate: the check a schema cannot fake.

Plan §1. "Does this validate against the schema?" is answered by construction
under constrained decoding, so it carries no evidence of correctness. This
module re-derives each claimed value from the raw source at its provenance span
and compares. No model participates in the comparison.

Grounding failure is terminal. It is never fed back to a repair loop: a model
told "this value failed provenance" will produce a better-disguised
hallucination with a plausible invented span (plan §6).
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.docmodel import RawDocument
from rag_ingest.ontology import (
    AbstainReason,
    Extraction,
    FieldAbstention,
    FieldValue,
    InvoiceHeader,
    Unclassified,
)

#: Characters that OCR routinely confuses. Normalizing them before comparison
#: is legitimate; normalizing them *into* the value is not, so the gate flags
#: the substitution it had to make.
OCR_CONFUSIONS = str.maketrans({"l": "1", "I": "1", "O": "0", "S": "5", "B": "8"})

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[,\s]")
_CURRENCY = re.compile(r"[₩$€£¥]")


#: Minimum grounded fields before an extraction counts as verified.
#:
#: Not a tunable knob to lower when the queue gets busy: this is the line
#: between "we know what this chunk is" and "we found one thing that happened to
#: match". Two, because an identifier alone identifies nothing about the content
#: a query would retrieve it for.
MIN_EXTRACTED_FIELDS = 2


class GroundingStatus(StrEnum):
    GROUNDED = "grounded"
    #: Values that were found all grounded, but too few were found to describe
    #: the chunk. Distinct from ABSTAINED (nothing found) and from MISMATCH
    #: (something found that contradicts the source).
    INCOMPLETE = "incomplete"
    #: Value does not appear in the span it claims.
    MISMATCH = "mismatch"
    #: Span is malformed, out of range, or points at nothing.
    UNVERIFIABLE = "unverifiable"
    #: Extraction declined to produce a value; legitimate, not a failure.
    ABSTAINED = "abstained"
    #: Content is outside the ontology; legitimate, not a failure.
    UNCLASSIFIED = "unclassified"


class Issue(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    status: GroundingStatus
    message: str


class GroundingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: GroundingStatus
    issues: tuple[Issue, ...] = Field(default_factory=tuple)
    #: True when the gate had to normalize OCR confusions to make a match.
    needed_normalization: bool = False

    @property
    def verified(self) -> bool:
        """Only a clean grounding pass is verified.

        Abstentions and unclassified content are *legitimate* but unverified:
        they are stored so nothing is lost, and only the verified subset is
        retrievable (plan §5a -- fail closed for the chunk, not the file).
        """
        return self.status is GroundingStatus.GROUNDED


def _normalize(text: str, *, strip_currency: bool = False) -> str:
    collapsed = _WHITESPACE.sub(" ", text).strip()
    if strip_currency:
        collapsed = _CURRENCY.sub("", collapsed)
    return collapsed.casefold()


def _numeric_key(raw: str) -> Decimal | None:
    """Parse a number out of OCR-damaged text, or return None.

    Deliberately strict: a string that is not numeric after cleanup yields None
    rather than a best-effort guess, because a wrong number is worse than an
    escalated one.
    """
    candidate = _CURRENCY.sub("", raw)
    candidate = _PUNCT.sub("", candidate)
    if not candidate:
        return None
    try:
        return Decimal(candidate)
    except InvalidOperation:
        return None


def _loose(text: str) -> str:
    """Aggressively normalized form for the final comparison tier.

    Strips separators and maps OCR confusions, so ``l,234.56`` and ``1234.56``
    compare equal. Only used after the stricter tiers fail, and any match made
    here is reported as ``needed_normalization`` so it stays visible.
    """
    translated = _normalize(text.translate(OCR_CONFUSIONS))
    return re.sub(r"[,.\s|]", "", translated)


def _value_matches_span(value: str, span_text: str) -> tuple[bool, bool]:
    """Compare a claimed value against raw span text.

    Returns ``(matched, needed_normalization)``.

    Four tiers, cheapest first:
      1. exact substring of the normalized span;
      2. numeric equality after stripping separators and currency;
      3. OCR-confusion-normalized substring;
      4. OCR-confusion-normalized *and* separator-stripped substring.
    """
    needle = _normalize(value)
    haystack = _normalize(span_text)
    if needle and needle in haystack:
        return True, False

    claimed_number = _numeric_key(value)
    if claimed_number is not None and needle:
        # Re-derive every numeric token present in the span and compare.
        for token in re.split(r"[\s|]+", span_text):
            if _numeric_key(token) == claimed_number:
                return True, False

    confused_needle = _normalize(value.translate(OCR_CONFUSIONS))
    confused_haystack = _normalize(span_text.translate(OCR_CONFUSIONS))
    if confused_needle and confused_needle in confused_haystack:
        return True, True

    loose_needle = _loose(value)
    loose_haystack = _loose(span_text)
    if loose_needle and loose_needle in loose_haystack:
        return True, True

    return False, False


class GroundingGate:
    """Adjudicates extractions mechanically."""

    def __init__(self, raw: RawDocument) -> None:
        self._raw = raw

    def check(self, extraction: Extraction) -> GroundingResult:
        if isinstance(extraction, Unclassified):
            return self._check_unclassified(extraction)

        issues: list[Issue] = []
        normalized = False
        fields = (
            ("invoice_number", extraction.invoice_number),
            ("issue_date", extraction.issue_date),
            ("total_amount", extraction.total_amount),
            ("currency", extraction.currency),
        ) if isinstance(extraction, InvoiceHeader) else (
            ("section_title", extraction.section_title),
            ("primary_finding", extraction.primary_finding),
        )

        abstained = 0
        for name, field in fields:
            if isinstance(field, FieldAbstention):
                abstained += 1
                if field.reason is AbstainReason.ILLEGIBLE:
                    issues.append(
                        Issue(
                            field=name,
                            status=GroundingStatus.ABSTAINED,
                            message=f"declined to extract: {field.reason}",
                        )
                    )
                else:
                    issues.append(
                        Issue(
                            field=name,
                            status=GroundingStatus.ABSTAINED,
                            message=f"declined to extract: {field.reason}",
                        )
                    )
                continue

            issue, did_normalize = self._check_field(name, field)
            normalized = normalized or did_normalize
            if issue is not None:
                issues.append(issue)

        if any(issue.status is GroundingStatus.MISMATCH for issue in issues):
            return GroundingResult(
                status=GroundingStatus.MISMATCH,
                issues=tuple(issues),
                needed_normalization=normalized,
            )

        if any(issue.status is GroundingStatus.UNVERIFIABLE for issue in issues):
            return GroundingResult(
                status=GroundingStatus.UNVERIFIABLE,
                issues=tuple(issues),
                needed_normalization=normalized,
            )

        if abstained == len(fields):
            # Every field declined: there is nothing to retrieve, so this is
            # not a verified extraction even though nothing is wrong with it.
            return GroundingResult(
                status=GroundingStatus.ABSTAINED,
                issues=tuple(issues),
                needed_normalization=normalized,
            )

        # Partial extraction is not verification. An invoice where only the
        # identifier was found is not a verified invoice -- entering it into
        # retrieval would let a query surface a chunk that claims to be a total
        # but carries none. The threshold is a required-evidence rule, not a
        # style preference: a wrong or hollow record is worse than a queued one.
        extracted = len(fields) - abstained
        if extracted < MIN_EXTRACTED_FIELDS:
            return GroundingResult(
                status=GroundingStatus.INCOMPLETE,
                issues=tuple(issues),
                needed_normalization=normalized,
            )

        return GroundingResult(
            status=GroundingStatus.GROUNDED,
            issues=tuple(issues),
            needed_normalization=normalized,
        )

    def _check_field(
        self, name: str, field: FieldValue
    ) -> tuple[Issue | None, bool]:
        ref = field.provenance
        if not ref.is_well_formed():
            return (
                Issue(
                    field=name,
                    status=GroundingStatus.UNVERIFIABLE,
                    message="provenance span is malformed",
                ),
                False,
            )

        span_text = self._raw.span_text(ref)
        if span_text is None:
            return (
                Issue(
                    field=name,
                    status=GroundingStatus.UNVERIFIABLE,
                    message="provenance span does not resolve in raw source",
                ),
                False,
            )

        matched, did_normalize = _value_matches_span(field.value, span_text)
        if matched:
            if did_normalize:
                return (
                    Issue(
                        field=name,
                        status=GroundingStatus.GROUNDED,
                        message="matched only after OCR-confusion normalization",
                    ),
                    True,
                )
            return None, False

        return (
            Issue(
                field=name,
                status=GroundingStatus.MISMATCH,
                message=(
                    f"claimed value {field.value!r} absent from span "
                    f"p{ref.page}[{ref.char_start}:{ref.char_end}] "
                    f"({span_text[:80]!r})"
                ),
            ),
            False,
        )

    def _check_unclassified(self, extraction: Unclassified) -> GroundingResult:
        ref = extraction.provenance
        span_text = self._raw.span_text(ref) if ref.is_well_formed() else None
        if span_text is None:
            return GroundingResult(
                status=GroundingStatus.UNVERIFIABLE,
                issues=(
                    Issue(
                        field="observed_shape",
                        status=GroundingStatus.UNVERIFIABLE,
                        message="unclassified content has no resolvable span",
                    ),
                ),
            )
        if not span_text.strip():
            return GroundingResult(
                status=GroundingStatus.UNVERIFIABLE,
                issues=(
                    Issue(
                        field="observed_shape",
                        status=GroundingStatus.UNVERIFIABLE,
                        message="unclassified span is empty",
                    ),
                ),
            )
        return GroundingResult(status=GroundingStatus.UNCLASSIFIED)

__all__ = [
    "GroundingGate",
    "GroundingResult",
    "GroundingStatus",
    "Issue",
]
