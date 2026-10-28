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
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.docmodel import RawDocument
from rag_ingest.gate_tier import (
    MIN_EXTRACTED_FIELDS,
    MIN_FIELDS_BY_TIER,
    assert_tier_applies,
    select_gate_tier,
)

if TYPE_CHECKING:
    from rag_ingest.shapes import ShapeRegistry

from rag_ingest.ontology import (
    AbstainReason,
    Extraction,
    FieldAbstention,
    FieldValue,
    GateTier,
    InvoiceHeader,
    ProseKnowledge,
    Unclassified,
)

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[,\s]")
_CURRENCY = re.compile(r"[₩$€£¥]")


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
    """Parse a number out of a span token, or return None.

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


def _value_matches_span(value: str, span_text: str) -> bool:
    """Compare a claimed value against raw span text.

    Two tiers, cheapest first:
      1. exact substring of the normalized span;
      2. numeric equality after stripping separators and currency.

    There is deliberately no character-substitution tier: this pipeline
    ingests born-digital documents only, so a glyph that is not the claimed
    character is a mismatch, full stop. A substitution table (l->1, O->0, ...)
    is how an OCR pipeline forgives its own reader; keeping one here would
    silently admit values the document does not contain.
    """
    needle = _normalize(value)
    haystack = _normalize(span_text)
    if needle and needle in haystack:
        return True

    claimed_number = _numeric_key(value)
    if claimed_number is not None and needle:
        # Re-derive every numeric token present in the span and compare.
        for token in re.split(r"[\s|]+", span_text):
            if _numeric_key(token) == claimed_number:
                return True

    return False


def _abstention_for(field_name: str):
    """The abstention a missing claim stands for: not present, honest, unverified."""
    return FieldAbstention(reason=AbstainReason.NOT_PRESENT)


class GroundingGate:
    """Adjudicates extractions mechanically.

    ``shapes`` carries the runtime shape registry. It is optional: the class
    path (built-in ontology types) works unchanged, and a contract path is
    available for shapes the owner approved at runtime -- the thing that used
    to require a rebuild. Both paths share ``_check_field`` and the
    required-evidence rule; a dynamic contract earns the same gate, not a
    weaker one.
    """

    def __init__(self, raw: RawDocument, shapes: ShapeRegistry | None = None) -> None:
        self._raw = raw
        self._shapes = shapes

    def check(
        self, extraction: Extraction, *, tier: GateTier | None = None
    ) -> GroundingResult:
        """Check one extraction against the raw source.

        ``tier`` defaults to the bar implied by the extraction's own ontology
        type. Passing it explicitly is for callers that must record which bar was
        applied; it is not a knob for relaxing a typed record, which
        ``assert_tier_applies`` refuses.
        """
        if isinstance(extraction, Unclassified):
            return self._check_unclassified(extraction)

        resolved_tier = tier if tier is not None else select_gate_tier(extraction)
        assert_tier_applies(resolved_tier, extraction)

        if isinstance(extraction, ProseKnowledge):
            return self._check_prose(extraction, resolved_tier)

        issues: list[Issue] = []
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
                issues.append(
                    Issue(
                        field=name,
                        status=GroundingStatus.ABSTAINED,
                        message=f"declined to extract: {field.reason}",
                    )
                )
                continue

            issue = self._check_field(name, field)
            if issue is not None:
                issues.append(issue)

        if any(issue.status is GroundingStatus.MISMATCH for issue in issues):
            return GroundingResult(
                status=GroundingStatus.MISMATCH,
                issues=tuple(issues),
            )

        if any(issue.status is GroundingStatus.UNVERIFIABLE for issue in issues):
            return GroundingResult(
                status=GroundingStatus.UNVERIFIABLE,
                issues=tuple(issues),
            )

        if abstained == len(fields):
            # Every field declined: there is nothing to retrieve, so this is
            # not a verified extraction even though nothing is wrong with it.
            return GroundingResult(
                status=GroundingStatus.ABSTAINED,
                issues=tuple(issues),
            )

        # Partial extraction is not verification. An invoice where only the
        # identifier was found is not a verified invoice -- entering it into
        # retrieval would let a query surface a chunk that claims to be a total
        # but carries none. The threshold is a required-evidence rule, not a
        # style preference: a wrong or hollow record is worse than a queued one.
        extracted = len(fields) - abstained
        if extracted < MIN_FIELDS_BY_TIER[resolved_tier]:
            return GroundingResult(
                status=GroundingStatus.INCOMPLETE,
                issues=tuple(issues),
            )

        return GroundingResult(
            status=GroundingStatus.GROUNDED,
            issues=tuple(issues),
        )

    def _check_prose(
        self, extraction: ProseKnowledge, tier: GateTier
    ) -> GroundingResult:
        """The quoted text must appear verbatim at the span it cites.

        One field, so a LENIENT bar is satisfied by construction -- the real content
        of this check is ``_check_field``. Do not read a GROUNDED result here as
        "this is correct": it means "this document says this at this place",
        nothing more. That distinction is why ``ChunkRecord`` stamps the tier
        alongside the verified flag.

        The tier is still applied rather than assumed. Asking for a two-field bar on
        a one-field shape yields INCOMPLETE instead of silently receiving the lenient
        verdict, so a caller cannot get a relaxed answer by accident.

        A multi-page quote is verified as the ordered join of its per-page
        slices -- not as "some span matches". Verifying only the first page
        would admit an invented page two; verifying any single page would
        admit an edit on the other. The join keeps the fidelity guarantee
        whole across the boundary.
        """
        if extraction.text.multi_page_spans:
            issue = self._check_multi_page_quote(extraction.text)
        else:
            issue = self._check_field("text", extraction.text)
        if issue is not None:
            return GroundingResult(status=issue.status, issues=(issue,))
        required = MIN_FIELDS_BY_TIER[tier]
        if required > 1:
            return GroundingResult(
                status=GroundingStatus.INCOMPLETE,
                issues=(
                    Issue(
                        field="text",
                        status=GroundingStatus.INCOMPLETE,
                        message=(
                            f"quoted prose supplies 1 field; {tier.value} requires {required}"
                        ),
                    ),
                ),
            )
        return GroundingResult(status=GroundingStatus.GROUNDED)

    def _check_multi_page_quote(self, field: FieldValue) -> Issue | None:
        """The quoted text must equal the ordered join of its page slices."""
        assert field.multi_page_spans is not None
        joined = self._raw.span_text_multi(field.multi_page_spans)
        if joined is None:
            return Issue(
                field="text",
                status=GroundingStatus.UNVERIFIABLE,
                message="a page slice of the multi-page quote does not resolve",
            )
        if _normalize(field.value) != _normalize(joined):
            return Issue(
                field="text",
                status=GroundingStatus.MISMATCH,
                message=(
                    "quoted text does not match the pages it cites "
                    f"({len(field.multi_page_spans)} pages cited)"
                ),
            )
        return None

    def check_contract(
        self,
        shape_name: str,
        claimed: dict[str, FieldValue | FieldAbstention],
        *,
        tier: GateTier | None = None,
    ) -> GroundingResult:
        """Check a dict of claimed fields against a registered shape contract.

        This is the runtime path for owner-approved shapes. The contract
        supplies the field order and the required set; the check itself is the
        same span re-derivation as every other value, so a dynamically
        approved shape cannot enter retrieval by a route the static shapes
        do not share.
        """
        contract = self._shapes.lookup(shape_name) if self._shapes else None
        if contract is None:
            raise KeyError(f"shape contract {shape_name!r} is not registered")

        resolved_tier = tier if tier is not None else contract.tier
        issues: list[Issue] = []
        abstained = 0
        for spec in contract.fields:
            field = claimed.get(spec.name, _abstention_for(spec.name))
            if isinstance(field, FieldAbstention):
                abstained += 1
                issues.append(
                    Issue(
                        field=spec.name,
                        status=GroundingStatus.ABSTAINED,
                        message=f"declined to extract: {field.reason}",
                    )
                )
                continue
            if spec.kind == "enum" and field.value not in spec.values:
                issues.append(
                    Issue(
                        field=spec.name,
                        status=GroundingStatus.MISMATCH,
                        message=(
                            f"claimed value {field.value!r} is not one of "
                            f"{list(spec.values)}"
                        ),
                    )
                )
                continue
            issue = self._check_field(spec.name, field)
            if issue is not None:
                issues.append(issue)

        if any(i.status is GroundingStatus.MISMATCH for i in issues):
            return GroundingResult(status=GroundingStatus.MISMATCH, issues=tuple(issues))
        if any(i.status is GroundingStatus.UNVERIFIABLE for i in issues):
            return GroundingResult(
                status=GroundingStatus.UNVERIFIABLE, issues=tuple(issues)
            )
        if abstained == len(contract.fields):
            return GroundingResult(status=GroundingStatus.ABSTAINED, issues=tuple(issues))

        # Declared constraints, compiled per field. Span re-derivation proves
        # the value is IN the source; these prove the values agree with what
        # the owner declared they mean. A number field that cannot parse is
        # UNVERIFIABLE, not silent -- silence is for honest abstentions.
        from decimal import Decimal


        claimed_values: dict[str, str] = {}
        for spec in contract.fields:
            field = claimed.get(spec.name)
            if not isinstance(field, FieldValue):
                continue
            if spec.kind == "number":
                candidate = _numeric_key(field.value)
                if candidate is None:
                    issues.append(
                        Issue(
                            field=spec.name,
                            status=GroundingStatus.UNVERIFIABLE,
                            message=f"claimed value {field.value!r} is not a number",
                        )
                    )
                    continue
                claimed_values[spec.name] = field.value
                if spec.ge is not None and candidate < Decimal(str(spec.ge)):
                    issues.append(
                        Issue(
                            field=spec.name,
                            status=GroundingStatus.MISMATCH,
                            message=f"{field.value} is below the declared minimum {spec.ge}",
                        )
                    )
            else:
                claimed_values[spec.name] = field.value

        for spec in contract.fields:
            if spec.min_ref is None or spec.name not in claimed_values:
                continue
            ref_value = claimed_values.get(spec.min_ref)
            if ref_value is None:
                continue
            mine = _numeric_key(claimed_values[spec.name])
            theirs = _numeric_key(ref_value)
            if mine is None or theirs is None:
                continue
            if mine < theirs:
                issues.append(
                    Issue(
                        field=spec.name,
                        status=GroundingStatus.MISMATCH,
                        message=(
                            f"{spec.name} ({claimed_values[spec.name]}) is below "
                            f"{spec.min_ref} ({ref_value})"
                        ),
                    )
                )

        if any(i.status is GroundingStatus.MISMATCH for i in issues):
            return GroundingResult(status=GroundingStatus.MISMATCH, issues=tuple(issues))
        if any(i.status is GroundingStatus.UNVERIFIABLE for i in issues):
            return GroundingResult(
                status=GroundingStatus.UNVERIFIABLE, issues=tuple(issues)
            )

        # Two evidence bars, both must hold: the tier's minimum-field rule and
        # the contract's own required set. A required field missing is
        # INCOMPLETE even when enough optional fields were found -- the owner
        # declared those fields load-bearing, and the tier floor cannot
        # override a declaration the gate would otherwise honor.
        missing_required = [
            spec.name
            for spec in contract.fields
            if spec.required and any(
                i.field == spec.name and i.status is GroundingStatus.ABSTAINED
                for i in issues
            )
        ]
        extracted = len(contract.fields) - abstained
        unparseable = any(i.status is GroundingStatus.UNVERIFIABLE for i in issues)
        if unparseable:
            return GroundingResult(status=GroundingStatus.UNVERIFIABLE, issues=tuple(issues))
        if missing_required or extracted < MIN_FIELDS_BY_TIER[resolved_tier]:
            return GroundingResult(status=GroundingStatus.INCOMPLETE, issues=tuple(issues))
        return GroundingResult(status=GroundingStatus.GROUNDED, issues=tuple(issues))

    def _check_field(self, name: str, field: FieldValue) -> Issue | None:
        ref = field.provenance
        if not ref.is_well_formed():
            return Issue(
                field=name,
                status=GroundingStatus.UNVERIFIABLE,
                message="provenance span is malformed",
            )

        span_text = self._raw.span_text(ref)
        if span_text is None:
            return Issue(
                field=name,
                status=GroundingStatus.UNVERIFIABLE,
                message="provenance span does not resolve in raw source",
            )

        if not _value_matches_span(field.value, span_text):
            return Issue(
                field=name,
                status=GroundingStatus.MISMATCH,
                message=(
                    f"claimed value {field.value!r} absent from span "
                    f"p{ref.page}[{ref.char_start}:{ref.char_end}] "
                    f"({span_text[:80]!r})"
                ),
            )
        return None

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
    "MIN_EXTRACTED_FIELDS",
    "GroundingGate",
    "GroundingResult",
    "GroundingStatus",
    "Issue",
]
