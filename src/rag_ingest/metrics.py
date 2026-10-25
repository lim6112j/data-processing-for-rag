"""Escalation metrics -- the health signal, reported per document class.

Plan §5b: a 5% global escalation rate can hide 100% on one high-volume class.
A class that does not improve release over release means the *ontology* is
wrong for that class, not the model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.escalation import EscalationReason, ReviewItem


class ClassReport(BaseModel):
    """Health of one document class.

    ``chunks`` is the denominator and must count *all* chunks that reached the
    store, verified or not. Counting only verified chunks would make a class
    whose chunks all failed look like a perfect score with no traffic.
    """

    model_config = ConfigDict(frozen=True)

    document_class: str
    chunks: int = Field(ge=0)
    escalated: int = Field(ge=0)
    by_reason: dict[EscalationReason, int] = Field(default_factory=dict)

    @property
    def escalation_rate(self) -> float:
        return 0.0 if self.chunks == 0 else self.escalated / self.chunks

    @property
    def suspect_ontology(self) -> bool:
        """Flag a class whose failure mode looks systemic.

        A majority of schema-shaped failures means the type definition is wrong
        for this class, not that the extractor was unlucky. Grounding failures
        are excluded: those are a parser/OCR problem, not a schema problem.
        """
        if self.escalated == 0:
            return False
        schema_reasons = (
            EscalationReason.UNCLASSIFIED,
            EscalationReason.SCHEMA_MISMATCH,
            EscalationReason.STRUCTURE_AMBIGUOUS,
        )
        schema_failures = sum(self.by_reason.get(reason, 0) for reason in schema_reasons)
        return schema_failures / self.escalated >= 0.5


class EscalationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    classes: tuple[ClassReport, ...] = Field(default_factory=tuple)

    @property
    def global_rate(self) -> float:
        total = sum(report.chunks for report in self.classes)
        if total == 0:
            return 0.0
        return sum(report.escalated for report in self.classes) / total

    @property
    def classes_to_rework(self) -> tuple[str, ...]:
        return tuple(r.document_class for r in self.classes if r.suspect_ontology)


def build_report(
    document_class: str, items: list[ReviewItem], *, chunks: int
) -> ClassReport:
    by_reason: dict[EscalationReason, int] = {}
    for item in items:
        by_reason[item.reason] = by_reason.get(item.reason, 0) + 1
    return ClassReport(
        document_class=document_class,
        chunks=chunks,
        escalated=len(items),
        by_reason=by_reason,
    )
