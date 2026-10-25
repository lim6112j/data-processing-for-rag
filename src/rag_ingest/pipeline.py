"""Pipeline orchestration: T0 extraction -> T1 grounding -> quarantine.

Plan §2 tiering:
  T0  deterministic parsers + template registry  : extract, emit confidence
  T1  grounding validator                        : adjudicate mechanically
  T2  LLM adjudicator                            : only disagreements (interface
                                                   defined, not implemented)
  T3  human, on quarantine

No model is in T0 or T1. That is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from rag_ingest.adjudication import (
    AdjudicationDecision,
    AdjudicationQuestion,
    DisagreementKind,
    build_question,
)
from rag_ingest.candidates import CandidateProducer, NullProducer, describe_disagreement
from rag_ingest.chunking import Chunk, Chunker
from rag_ingest.docmodel import Element, ProvenanceRef, RawDocument
from rag_ingest.escalation import (
    Correction,
    EscalationReason,
    ReviewItem,
    ReviewQueue,
)
from rag_ingest.grounding import GroundingGate, GroundingResult, GroundingStatus
from rag_ingest.ontology import (
    ONTOLOGY_VERSION,
    PIPELINE_VERSION,
    AbstainReason,
    ChunkRecord,
    Extraction,
)
from rag_ingest.tiering import Tier, TierDecision, select_tier


@dataclass(frozen=True)
class IngestOutcome:
    """Everything that happened to one document, for reporting and metrics."""

    doc_id: str
    verified_chunks: tuple[ChunkRecord, ...]
    quarantined_chunks: tuple[ChunkRecord, ...]
    review_items: tuple[ReviewItem, ...]
    tier_decisions: tuple[tuple[str, TierDecision], ...]
    #: T2 decisions taken during this document, for the label set.
    adjudications: tuple[tuple[AdjudicationQuestion, AdjudicationDecision], ...] = ()


class Adjudicator:
    """T2 interface: resolve a *structural* disagreement between extractors.

    Intentionally unimplemented. Two invariants live in this contract and are
    enforced by ``IngestPipeline`` rather than trusted to implementations:

    1. T2 is never handed a grounding failure. It may only see questions whose
       evidence is structural (column count, cell boundaries, reading order,
       character identity). A model told a value failed provenance invents a
       better-disguised span.
    2. T2 cannot verify anything. A ``pick`` narrows the candidate set; the
       chosen reading is re-checked by T1 before it can enter retrieval.
    """

    def adjudicate(self, question: AdjudicationQuestion) -> AdjudicationDecision:
        raise NotImplementedError(
            "T2 adjudication is deliberately not implemented; wire a model here "
            "only for parser disagreements and low-confidence cases."
        )


class IngestPipeline:
    """Runs one document through the cascade."""

    def __init__(
        self,
        queue: ReviewQueue,
        extractor,
        chunker: Chunker | None = None,
        today: date | None = None,
        adjudicator: Adjudicator | None = None,
        producer: CandidateProducer | None = None,
    ) -> None:
        self._queue = queue
        self._extractor = extractor
        self._chunker = chunker or Chunker()
        self._today = today or date.today()
        # Default is the unimplemented seam: a pipeline with no T2 configured
        # escalates rather than guessing.
        self._adjudicator = adjudicator or Adjudicator()
        # Default produces a single reading, so no disagreement reaches T2.
        self._producer: CandidateProducer = producer or NullProducer()

    def ingest(
        self,
        raw: RawDocument,
        elements: list[Element],
        *,
        owner_id: str,
        retrieval_demand: int = 0,
    ) -> IngestOutcome:
        tier_decisions: list[tuple[str, TierDecision]] = []
        extractable: list[Element] = []
        pre_escalations: list[Element] = []

        for element in elements:
            decision = select_tier(element)
            tier_decisions.append((element.element_id, decision))
            if decision.tier is Tier.ESCALATE:
                pre_escalations.append(element)
            elif decision.tier is Tier.RETRY_OCR:
                # Plan section 3: never attempt semantic extraction on garbage
                # text. The text layer failed, so chunking it would put
                # unreadable characters into the store under a confident
                # heading. Queue it for re-scan instead.
                pre_escalations.append(element)
            else:
                # EXTRACT and STRUCTURE_UNKNOWN both carry usable text;
                # STRUCTURE_UNKNOWN is chunked as flat text only.
                extractable.append(element)

        chunks = self._chunker.chunk(raw.doc_id, extractable)
        gate = GroundingGate(raw)

        verified: list[ChunkRecord] = []
        quarantined: list[ChunkRecord] = []
        review_items: list[ReviewItem] = []
        adjudications: list[tuple[AdjudicationQuestion, AdjudicationDecision]] = []

        for chunk in chunks:
            extraction = self._extractor.extract(chunk)
            result = gate.check(extraction) if extraction is not None else None
            record = _to_record(chunk, extraction, result)

            if result is not None and result.verified:
                verified.append(record)
                continue

            quarantined.append(record)
            item = self._review_item(
                chunk=chunk,
                extraction=extraction,
                result=result,
                owner_id=owner_id,
                retrieval_demand=retrieval_demand,
                # A chunk built from low-layout-confidence elements has
                # unreliable structure -- exactly what T2 exists to settle.
                structure_ambiguous=_has_unreliable_structure(chunk, tier_decisions),
            )

            # T2 is consulted only for structural ambiguity. Grounding failures
            # skip it entirely: they are terminal, and handing one to a model
            # invites a better-disguised hallucination.
            if item.reason is EscalationReason.STRUCTURE_AMBIGUOUS:
                question = self._adjudication_question(chunk)
                if question is not None:
                    decision = self._call_adjudicator(question, raw)
                    adjudications.append((question, decision))
                    item = _apply_adjudication(item, question, decision)

            review_items.append(item)

        for element in pre_escalations:
            retry_ocr = any(
                element.element_id == element_id and decision.tier is Tier.RETRY_OCR
                for element_id, decision in tier_decisions
            )
            if retry_ocr:
                question = (
                    f"Page {element.page} was read at low OCR confidence. Is a higher-resolution "
                    "scan or a text version of this passage available?"
                )
            else:
                question = (
                    f"Page {element.page} could not be read as text. "
                    "Is a clearer scan available, or can the content be supplied as text?"
                )
            review_items.append(
                ReviewItem(
                    item_id=f"{raw.doc_id}:{element.element_id}:triage",
                    doc_id=raw.doc_id,
                    chunk_id=element.element_id,
                    reason=EscalationReason.EXTRACTION_FAILED,
                    question=question,
                    owner_id=owner_id,
                    created_on=self._today,
                    retrieval_demand=retrieval_demand,
                )
            )

        return IngestOutcome(
            doc_id=raw.doc_id,
            verified_chunks=tuple(verified),
            quarantined_chunks=tuple(quarantined),
            review_items=tuple(review_items),
            tier_decisions=tuple(tier_decisions),
            adjudications=tuple(adjudications),
        )

    def _review_item(
        self,
        *,
        chunk: Chunk,
        extraction: Extraction | None,
        result: GroundingResult | None,
        owner_id: str,
        retrieval_demand: int,
        structure_ambiguous: bool = False,
    ) -> ReviewItem:
        # Precedence matters. Grounding mismatch is terminal and always wins:
        # it must never reach T2. Structural ambiguity comes next, because a
        # chunk whose layout is unreliable has no trustworthy "no known type"
        # reading either -- T2 may still be able to settle it from the image.
        if result is not None and result.status is GroundingStatus.MISMATCH:
            reason = EscalationReason.UNGROUNDED
            bad = ", ".join(
                issue.field for issue in result.issues if issue.status is GroundingStatus.MISMATCH
            )
            question = (
                f"The extracted value for {bad} does not appear in the source. "
                "What should it be?"
            )
            issues = result.issues
        elif result is not None and result.status is GroundingStatus.INCOMPLETE:
            reason = EscalationReason.INCOMPLETE_EXTRACTION
            issue_by_field = {
                issue.field: issue
                for issue in result.issues
                if issue.status is GroundingStatus.ABSTAINED
            }
            # "Not legible" and "not present" need different actions from the
            # reviewer: one is a re-scan, the other is new information. Collapsing
            # them into "missing" told them nothing about what to do.
            illegible = [
                name for name in issue_by_field if "illegible" in issue_by_field[name].message
            ]
            absent = [name for name in issue_by_field if name not in illegible]
            found = [
                name
                for name in _field_names(extraction)
                if name not in issue_by_field
            ]

            parts = []
            where = f" under '{chunk.heading_path}'" if chunk.heading_path.titles else ""
            if found:
                parts.append(f"read {_join(found)}")
            if illegible:
                parts.append(f"could not read {_join(illegible)} (damaged or illegible)")
            if absent:
                parts.append(f"did not find {_join(absent)}")

            question = (
                f"Partially read the record{where}: " + "; ".join(parts) + "."
            )
            if illegible:
                question += (
                    " For the damaged fields, is a clearer scan available,"
                    " and what are the correct values?"
                )
            if absent:
                question += " For the missing fields, what should they be?"
            issues = result.issues
        elif structure_ambiguous:
            # Structural conflict -- the one case T2 exists for.
            reason = EscalationReason.STRUCTURE_AMBIGUOUS
            question = (
                f"How is the content under '{chunk.heading_path}' structured? "
                "Two parsers read it differently."
            )
            issues = result.issues if result is not None else ()
        elif extraction is None or result is None or result.status is GroundingStatus.UNCLASSIFIED:
            reason = EscalationReason.UNCLASSIFIED
            question = (
                f"What is the content under '{chunk.heading_path}'? "
                "It matched no known document type."
            )
            issues = ()
        else:
            reason = EscalationReason.LOW_CONFIDENCE
            question = f"Please confirm the extracted fields for '{chunk.heading_path}'."
            issues = result.issues

        return ReviewItem(
            item_id=f"{chunk.doc_id}:{chunk.chunk_id}",
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            reason=reason,
            issues=issues,
            question=question,
            owner_id=owner_id,
            created_on=self._today,
            retrieval_demand=retrieval_demand,
        )

    def _adjudication_question(
        self, chunk: Chunk
    ) -> AdjudicationQuestion | None:
        """Build the T2 question, or return None when nothing is in dispute.

        Candidates come from the configured ``CandidateProducer``. With the
        default ``NullProducer`` there is one reading, hence no disagreement,
        hence no T2 call -- which is the honest outcome for a single-parser
        deployment rather than a fabricated second opinion.
        """
        candidates = self._producer.produce(chunk)
        if len(candidates) < 2:
            return None
        return build_question(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            kind=DisagreementKind.READING_ORDER,
            prompt=(
                "Two parsers read this region differently. Which reading correctly "
                "preserves its structure? Answer from the page image. If the image "
                "does not settle it, abstain."
            ),
            candidates=candidates,
            provenance=ProvenanceRef(
                page=chunk.page_start,
                char_start=0,
                char_end=1,
                element_id=chunk.element_ids[0] if chunk.element_ids else None,
                bbox=chunk.bbox,
            ),
            context_text=chunk.text,
        )

    def _call_adjudicator(
        self, question: AdjudicationQuestion, raw: RawDocument
    ) -> AdjudicationDecision:
        """Call T2, supplying the evidence crop when the backend wants one.

        The default ``Adjudicator`` takes only the question (it is a stub). A
        visual backend additionally needs the image, which lives on the raw
        document -- so it is offered via ``adjudicate_with_evidence`` when the
        backend supports it. Backends that raise are contained here: a failed
        adjudication must degrade to the human queue, never abort ingestion.
        """
        evidence = raw.visual_evidence(question.provenance)
        with_evidence = getattr(self._adjudicator, "adjudicate_with_evidence", None)
        try:
            if with_evidence is not None:
                return with_evidence(question, image_png=evidence)
            return self._adjudicator.adjudicate(question)
        except NotImplementedError:
            # No T2 configured: escalate rather than guess.
            return AdjudicationDecision(
                kind="abstain",
                reason=AbstainReason.AMBIGUOUS,
                rationale="No T2 adjudicator configured.",
            )
        except Exception as exc:
            return AdjudicationDecision(
                kind="abstain",
                reason=AbstainReason.AMBIGUOUS,
                rationale=f"T2 failed: {type(exc).__name__}",
            )


def _field_names(extraction: Extraction | None) -> list[str]:
    """Field names of an extraction, in declaration order.

    Derived rather than hardcoded so adding an ontology field cannot silently
    drop it from the review question.
    """
    if extraction is None:
        return []
    return [
        name
        for name in type(extraction).model_fields
        if name not in ("doc_type", "kind")
    ]


def _join(names: list[str]) -> str:
    """Readable list: 'a, b and c'."""
    human = [name.replace("_", " ") for name in names]
    if len(human) <= 1:
        return "".join(human)
    return f"{', '.join(human[:-1])} and {human[-1]}"


def _has_unreliable_structure(
    chunk: Chunk, tier_decisions: list[tuple[str, TierDecision]]
) -> bool:
    """True when any element in the chunk was routed STRUCTURE_UNKNOWN.

    That tier means layout confidence was below the floor: the text is readable
    but its arrangement is not trustworthy. Routing only these to T2 keeps the
    model off every chunk that parsed cleanly.
    """
    unreliable = {
        element_id
        for element_id, decision in tier_decisions
        if decision.tier is Tier.STRUCTURE_UNKNOWN
    }
    return any(element_id in unreliable for element_id in chunk.element_ids)


def _apply_adjudication(
    item: ReviewItem, question: AdjudicationQuestion, decision: AdjudicationDecision
) -> ReviewItem:
    """Fold a T2 decision into the review item.

    A ``pick`` does NOT verify anything -- it only narrows the question. The
    chosen reading still has to pass T1, and until it does the chunk stays
    quarantined. ``abstain`` and ``escalate`` leave the item human-facing, with
    the T2 rationale recorded so the queue shows why automation stopped.
    """
    candidate_ids = tuple(candidate.candidate_id for candidate in question.candidates)

    if not decision.is_coherent(len(question.candidates)):
        # Malformed-but-parseable answers are treated as an abstention, never
        # as a silent pick.
        return item.model_copy(
            update={
                "reason": EscalationReason.STRUCTURE_AMBIGUOUS,
                "question": item.question + " (T2 returned an incoherent decision.)",
                "candidate_ids": candidate_ids,
            }
        )

    if decision.kind == "pick":
        chosen = question.candidates[decision.choice_index or 0]
        disagreement = describe_disagreement(list(question.candidates))
        return item.model_copy(
            update={
                "question": (
                    f"{item.question} Readings: {disagreement}. "
                    f"T2 selected '{chosen.produced_by}' from the page image. "
                    "Confirm it before it becomes retrievable."
                ),
                "candidate_ids": candidate_ids,
                "adjudicated_choice": chosen.candidate_id,
            }
        )

    note = decision.rationale or ""
    if decision.kind == "escalate" and decision.owner_question:
        return item.model_copy(
            update={
                "question": decision.owner_question,
                "candidate_ids": candidate_ids,
            }
        )
    return item.model_copy(
        update={
            "question": f"{item.question} (T2 could not settle it. {note})".strip(),
            "candidate_ids": candidate_ids,
        }
    )


def _to_record(
    chunk: Chunk, extraction: Extraction | None, result: GroundingResult | None
) -> ChunkRecord:
    return _chunk_to_record(
        chunk, verified=bool(result is not None and result.verified), extraction=extraction
    )


def _chunk_to_record(
    chunk: Chunk, verified: bool, extraction: Extraction | None
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        embed_text=chunk.embed_text,
        verified=verified,
        ontology_version=ONTOLOGY_VERSION,
        pipeline_version=PIPELINE_VERSION,
        extraction=extraction,
    )


__all__ = [
    "Adjudicator",
    "Correction",
    "IngestOutcome",
    "IngestPipeline",
    "_chunk_to_record",
]
