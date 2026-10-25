"""T2 adjudication contract: the shape, not the model.

Plan §2: T2 resolves disagreements between extractors. It does not extract.
The distinction is enforced structurally here:

  * a candidate carries a *reading* of a region, never a field value;
  * a decision is one of three legal outcomes, and two of them are refusals;
  * an adjudication can never mark content verified -- only T1 can, and the
    chosen candidate must survive a fresh T1 pass.

Everything in this module is provider-agnostic. Wiring a model in means
implementing ``Adjudicator``; swapping models never changes this file.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.docmodel import BBox, ProvenanceRef
from rag_ingest.ontology import AbstainReason


class DisagreementKind(StrEnum):
    """What kind of conflict T2 is being asked to settle.

    Every one of these is a *structural* disagreement. Grounding failures are
    deliberately absent: those never reach T2 (plan §6), because a model told
    "this value failed provenance" invents a plausible span rather than
    conceding.
    """

    COLUMN_COUNT = "column_count"
    CELL_BOUNDARY = "cell_boundary"
    READING_ORDER = "reading_order"
    CHARACTER_IDENTITY = "character_identity"
    HEADER_SCOPE = "header_scope"


class Candidate(BaseModel):
    """One extractor's reading of a region.

    ``proposed_text`` is what the extractor read -- not a parsed field. T2
    chooses between readings; turning a reading into a field value is T0/T1's
    job.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    produced_by: str
    proposed_text: str
    bbox: BBox | None = None
    note: str | None = None


class AdjudicationQuestion(BaseModel):
    """Everything T2 is given for one decision."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    chunk_id: str
    doc_id: str
    kind: DisagreementKind
    #: A concrete, closed question. "Which reading is correct here?" gets a
    #: decision; an open prompt invites invention.
    prompt: str
    candidates: tuple[Candidate, ...]
    provenance: ProvenanceRef
    #: Raw page text around the region. Context only -- it cannot settle a
    #: character-identity question that the image can.
    context_text: str | None = None

    @property
    def has_visual_evidence(self) -> bool:
        """Whether T2 can actually see the region it is judging.

        False means T2 must abstain. Answering from text alone would be a guess
        dressed as an adjudication.
        """
        return self.provenance.has_visual_evidence


class AdjudicationDecision(BaseModel):
    """T2's answer. Exactly one of three outcomes.

    Note what is absent: there is no ``value`` field. Adding one would reopen a
    free-form generation channel that bypasses the provenance gate.
    ``extra="forbid"`` makes that structural rather than conventional -- a model
    that emits a stray ``value`` gets a validation error instead of having it
    silently dropped, so the violation is visible in the label set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["pick", "abstain", "escalate"]
    #: Set only when kind == "pick". Indexes into ``question.candidates``.
    choice_index: int | None = Field(default=None, ge=0)
    #: Set only when kind == "abstain".
    reason: AbstainReason | None = None
    #: Set only when kind == "escalate": a specific question for the document
    #: owner.
    owner_question: str | None = None
    #: Free-text justification, recorded for the label set but never trusted.
    rationale: str | None = None

    def is_coherent(self, candidate_count: int) -> bool:
        """Whether the decision is internally consistent.

        Guards against the malformed-but-parseable answers a model produces
        when its schema allows a union: a "pick" with no index, an index out
        of range, or an abstention with no reason.
        """
        if self.kind == "pick":
            return self.choice_index is not None and 0 <= self.choice_index < candidate_count
        if self.kind == "abstain":
            return self.reason is not None
        if self.kind == "escalate":
            return bool(self.owner_question and self.owner_question.strip())
        return False


def build_question(
    *,
    chunk_id: str,
    doc_id: str,
    kind: DisagreementKind,
    prompt: str,
    candidates: list[Candidate],
    provenance: ProvenanceRef,
    context_text: str | None = None,
) -> AdjudicationQuestion:
    if len(candidates) < 2:
        raise ValueError("a disagreement needs at least two candidates")
    return AdjudicationQuestion(
        question_id=f"{chunk_id}:{kind.value}",
        chunk_id=chunk_id,
        doc_id=doc_id,
        kind=kind,
        prompt=prompt,
        candidates=tuple(candidates),
        provenance=provenance,
        context_text=context_text,
    )


__all__ = [
    "AdjudicationDecision",
    "AdjudicationQuestion",
    "Candidate",
    "DisagreementKind",
    "build_question",
]
