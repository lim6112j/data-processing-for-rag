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
ONTOLOGY_VERSION = "1.0.0"

#: Stamped on every chunk. Parsers change and documents will be re-ingested.
PIPELINE_VERSION = "0.1.0"


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


class InvoiceHeader(BaseModel):
    """Shape 1: a document header block with an identifier, date and total."""

    model_config = ConfigDict(frozen=True)

    doc_type: Literal["invoice"] = "invoice"
    invoice_number: GroundedField
    issue_date: GroundedField
    total_amount: GroundedField
    currency: GroundedField


class ReportSection(BaseModel):
    """Shape 2: a narrative section, optionally with a numeric finding."""

    model_config = ConfigDict(frozen=True)

    doc_type: Literal["report"] = "report"
    section_title: GroundedField
    primary_finding: GroundedField


Extraction = InvoiceHeader | ReportSection | Unclassified


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


__all__ = [
    "ONTOLOGY_VERSION",
    "PIPELINE_VERSION",
    "AbstainReason",
    "ChunkRecord",
    "Extraction",
    "FieldAbstention",
    "FieldValue",
    "GroundedField",
    "InvoiceHeader",
    "ReportSection",
    "Unclassified",
]
