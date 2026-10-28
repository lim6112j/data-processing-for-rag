"""Gate strictness: which bar a claimed record must clear to be retrievable.

Distinct from ``tiering.Tier``, which decides whether an element is worth
interpreting at all (OCR and layout quality). This module decides the bar a
record is held to *after* an extractor has claimed one. Same word, two different
questions; keeping them in separate modules keeps both honest.

survey.md (Appendix) states the criterion. DB-bound values can be held to a high
bar *because they can be checked* -- types, ranges, arithmetic, referential
integrity. Prose cannot be checked for meaning, so applying the same bar rejects
all of it. Measured on a real 646-page manual: 754 chunks, 0 admitted
(``corpus/README.md``). Rejecting every knowledge chunk is not strictness, it is
a pipeline that cannot ingest prose.

Tier assignment comes from the ontology type the extractor claimed, and nothing
else. survey.md is explicit that an LLM classifier is the wrong mechanism:

    a misclassified STRICT document silently becomes LENIENT, which is exactly
    the silent-corruption path this repo exists to prevent.

So the mapping is a lookup table, ``assert_tier_applies`` makes the dangerous
direction impossible to express, and no model participates.
"""

from __future__ import annotations

from rag_ingest.ontology import (
    Extraction,
    GateTier,
    InvoiceHeader,
    ProseKnowledge,
    ReportSection,
)

#: Minimum grounded fields before an extraction counts as verified.
#:
#: Not a tunable knob to lower when the queue gets busy: this is the line
#: between "we know what this chunk is" and "we found one thing that happened to
#: match". Two, because an identifier alone identifies nothing about the content
#: a query would retrieve it for.
#:
#: Lives here rather than in ``grounding`` because it is a *policy* threshold and
#: ``MIN_FIELDS_BY_TIER`` below is where policy now lives. ``grounding``
#: re-exports it so existing callers and docs keep working.
MIN_EXTRACTED_FIELDS = 2

#: Minimum grounded fields a record of each tier must supply.
#:
#: LENIENT requires one, because the quoted text *is* the record. The other two
#: keep the required-evidence rule: an identifier alone identifies nothing about
#: the content a query would retrieve it for. That constant is threaded through
#: rather than restated, so adding a tier cannot silently lower the bar for the
#: tiers that already exist.
MIN_FIELDS_BY_TIER: dict[GateTier, int] = {
    GateTier.STRICT: MIN_EXTRACTED_FIELDS,
    GateTier.STANDARD: MIN_EXTRACTED_FIELDS,
    GateTier.LENIENT: 1,
}


def select_gate_tier(extraction: Extraction) -> GateTier:
    """The bar this record must clear, from its claimed ontology type.

    ``Unclassified`` maps to STRICT deliberately. It never verifies either way,
    but if a future change ever let an unclassified record through, it should be
    the hardest bar that fails it -- not the easiest.
    """
    if isinstance(extraction, InvoiceHeader):
        return GateTier.STRICT
    if isinstance(extraction, ReportSection):
        return GateTier.STANDARD
    if isinstance(extraction, ProseKnowledge):
        return GateTier.LENIENT
    return GateTier.STRICT


def assert_tier_applies(tier: GateTier, extraction: Extraction) -> None:
    """Refuse a LENIENT bar on anything that is not quoted prose.

    This is the one invariant the design cannot afford to leave to convention.
    Every other mistake in this pipeline quarantines too much, which is visible in
    the queue and recoverable. Relaxing a typed record to LENIENT admits an
    unverified value into retrieval with a citation attached -- silently, and in
    the exact way this repo exists to prevent.

    Raising rather than silently upgrading to STRICT: a caller that asked for the
    wrong bar has a bug, and swallowing it would hide the bug that produced it.
    """
    if tier is GateTier.LENIENT and not isinstance(extraction, ProseKnowledge):
        raise ValueError(
            f"LENIENT gate cannot be applied to {type(extraction).__name__}: only a "
            "verbatim ProseKnowledge record may be admitted on quotation fidelity alone"
        )


__all__ = [
    "MIN_EXTRACTED_FIELDS",
    "MIN_FIELDS_BY_TIER",
    "assert_tier_applies",
    "select_gate_tier",
]
