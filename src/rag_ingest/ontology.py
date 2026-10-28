"""Ontology v1: small, Pydantic-native, and deliberately boring.

Plan §4: discover, don't design. This schema covers the two shapes a seed
corpus actually repeats (an invoice header, a figure-bearing report section).
It is not an attempt to model the domain -- it is the minimum needed to make
extraction testable. Promote new types only when a shape recurs in the cluster
report; do not add a class because it seems plausible.

Every field a model may populate is wrapped in ``GroundedValue`` so it cannot
be represented without provenance.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from rag_ingest.docmodel import ProvenanceRef

#: Stamped on every chunk. Schemas will change; mixed vintages degrade
#: retrieval silently, so the vintage must be recorded.
ONTOLOGY_VERSION = "1.1.0"

#: Stamped on every chunk. Parsers change and documents will be re-ingested.
PIPELINE_VERSION = "0.1.0"


class DefectCode(StrEnum):
    """A visible defect in text that was admitted anyway.

    The project's cleaning policy is mark, never repair: an LLM repairs almost
    nothing correctly (survey.md §1, max repair-match 0.028) and human edit
    systems introduce errors at nearly the rate they fix them (§6.1). So damaged
    text enters retrieval as what the source says, carrying a mark, and the
    correction decision stays with a human instead of moving silently to the
    consumer.

    Closed enum, like ``AbstainReason``: a defect code is a triageable fact, not
    a free-text note.
    """

    SUSPECTED_OCR_DAMAGE = "suspected_ocr_damage"


class AbstainReason(StrEnum):
    """Why an extraction declined. Closed enum: the queue must be triageable."""

    NOT_PRESENT = "not_present"
    ILLEGIBLE = "illegible"
    AMBIGUOUS = "ambiguous"
    OUT_OF_ONTOLOGY = "out_of_ontology"


class FieldValue(BaseModel):
    """A successfully extracted value, inseparable from its provenance span."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["value"] = "value"
    value: str
    provenance: ProvenanceRef


class FieldAbstention(BaseModel):
    """An explicit refusal.

    Plan §1: this must be a *legal* output. If the schema offers no way to say
    "not legible", a constrained decoder's only legal move is invention. An
    Optional field set to None is not equivalent -- structured generation
    prefers filling optional fields over omitting them.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["abstain"] = "abstain"
    reason: AbstainReason
    note: str | None = None


class Unclassified(BaseModel):
    """Content that matches no defined type.

    Plan §3, failure class D. A closed ontology cannot detect its own blind
    spots, so this must be a first-class outcome that is *not* counted as a
    validation failure.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["unclassified"] = "unclassified"
    observed_shape: str
    provenance: ProvenanceRef


GroundedField = Annotated[
    FieldValue | FieldAbstention,
    Field(discriminator="kind"),
]


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class LineItem(BaseModel):
    """One reconcilable row of an invoice.

    Exists for the arithmetic reconciliation check (``checks.py``): a total that
    contradicts its own line items is the canonical error grounding cannot see
    -- both numbers are verbatim in the source. The amount is a GroundedField
    for the same reason every other value is: without a span the check cannot
    run and the row should not exist.
    """

    model_config = ConfigDict(frozen=True)

    amount: GroundedField
    description: GroundedField | None = None


class InvoiceHeader(BaseModel):
    """Shape 1: a document header block with an identifier, date and total."""

    model_config = ConfigDict(frozen=True)

    doc_type: Literal["invoice"] = "invoice"
    invoice_number: GroundedField
    issue_date: GroundedField
    total_amount: GroundedField
    currency: GroundedField
    #: Reconcilable rows, when the extractor found labelled ones. Default empty
    #: so existing producers stay valid; reconciliation only fires when the
    #: check can see complete rows, so an empty tuple is "unknown", not "zero".
    line_items: tuple[LineItem, ...] = ()


class ReportSection(BaseModel):
    """Shape 2: a narrative section, optionally with a numeric finding."""

    model_config = ConfigDict(frozen=True)

    doc_type: Literal["report"] = "report"
    section_title: GroundedField
    primary_finding: GroundedField


class ProseKnowledge(BaseModel):
    """Shape 3: narrative content, quoted verbatim from the source.

    Admission for this shape is LENIENT: the only check is that the quoted text
    appears verbatim at the span it cites. That is still a real check -- it
    catches chunker damage, truncation and span drift -- but be precise about
    what it proves. It verifies **fidelity of quotation**, not correctness.

    survey.md: "A prose sentence that is verbatim in the source may be wrong, but
    the pipeline still conveys 'this document said this', which is what retrieval
    is for. It is quoting, not summarising." Never report a LENIENT pass as
    "verified" without saying which tier it passed.

    This shape exists because the alternative was measured and is unacceptable:
    a 646-page manual produced 754 chunks and 0 admissions, every one rejected
    as ``unclassified`` (``corpus/README.md``).
    """

    model_config = ConfigDict(frozen=True)

    doc_type: Literal["prose"] = "prose"
    text: GroundedField


class GateTier(StrEnum):
    """How strict the grounding gate was for one record.

    Lives here rather than in ``gate_tier`` to keep ``ChunkRecord`` (which stamps
    it) free of a circular import. The selection logic -- which must inspect the
    extraction types below -- lives in ``gate_tier``.

    STRICT and STANDARD currently apply the same checks. That is deliberate
    honesty rather than an oversight: the arithmetic and range checks that are
    supposed to separate them are the highest-value missing piece per survey.md
    and are not built yet. The enum is complete so the checks attach to a tier
    when they arrive, instead of a silent reclassification later.
    """

    STRICT = "strict"
    STANDARD = "standard"
    LENIENT = "lenient"


Extraction = InvoiceHeader | ReportSection | ProseKnowledge | Unclassified


class ChunkRecord(BaseModel):
    """A chunk as it enters the store, carrying its versions and verification."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    embed_text: str
    verified: bool
    ontology_version: str = ONTOLOGY_VERSION
    pipeline_version: str = PIPELINE_VERSION
    extraction: Extraction | None = None
    #: Which bar this record was held to. A ``verified`` flag without this is
    #: ambiguous: "quoted verbatim" and "grounded, complete and consistent" are
    #: not the same claim, and mixing them silently is the ambiguity this project
    #: exists to remove.
    gate_tier: GateTier | None = None
    #: Defects observed in the admitted text, marked rather than repaired.
    #: Populated for verified records only -- a mark is a statement about text
    #: that entered retrieval, and stamping quarantined chunks would blur the one
    #: distinction the store keeps.
    defects: tuple[DefectCode, ...] = ()


__all__ = [
    "ONTOLOGY_VERSION",
    "PIPELINE_VERSION",
    "AbstainReason",
    "ChunkRecord",
    "DefectCode",
    "Extraction",
    "FieldAbstention",
    "FieldValue",
    "GateTier",
    "GroundedField",
    "InvoiceHeader",
    "LineItem",
    "ProseKnowledge",
    "ReportSection",
    "Unclassified",
]
