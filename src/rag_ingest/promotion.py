"""Owner approval closes the quarantine loop -- but never replaces the gate.

A quarantined chunk is a question the owner must answer. Once answered, the
chunk may enter retrieval, yet the answer's *authority* is strictly scoped:

- The owner can confirm that an extraction is right. The gate still re-checks
  the cited values against the source, because meaning is the owner's domain
  and citation is the machine's. An approval cannot make an uncited value
  cited -- a chunk the gate still refuses stays quarantined, and the refusal
  names itself so the owner knows the chunk needs a fix, not a rubber stamp.
- An unclassified chunk has no values to confirm. The only honest claim
  available is the verbatim quote, which the LENIENT tier checks mechanically.
- The promoted record carries the approval's history (who, when), so a
  downstream store can distinguish machine-verified from owner-approved
  rather than mixing the two silently.

Promotion re-issues records; nothing is mutated. The frozen outcome is the
audit trail."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date

from rag_ingest.docmodel import Element, RawDocument
from rag_ingest.extractors import ProseExtractor
from rag_ingest.grounding import GroundingGate, GroundingStatus, select_gate_tier
from rag_ingest.pipeline import IngestOutcome


@dataclass(frozen=True)
class Approval:
    """Who approved the chunk, and when -- the record that rides downstream."""

    reviewer: str
    approved_on: date


class RefusalReason:
    """Why an approval could not promote the chunk. Named, not raised: an
    approval refusal is a pipeline outcome, not an error condition."""

    NOT_CITABLE = "not_citable"
    GATE_STILL_REFUSES = "gate_still_refuses"
    NO_CHUNK = "no_chunk"


@dataclass(frozen=True)
class ApprovalRefusal:
    """The chunk stays quarantined; the reason says what would change it."""

    reason: str
    detail: str


@dataclass(frozen=True)
class PromotionResult:
    """The new outcome if promoted, or the honest refusal if not."""

    outcome: IngestOutcome | None
    refusal: ApprovalRefusal | None
    approval: Approval


def promote_chunk(
    outcome: IngestOutcome,
    chunk_id: str,
    approval: Approval,
    *,
    gate: GroundingGate,
    raw: RawDocument,
    elements: list[Element],
) -> PromotionResult:
    """Promote one approved chunk out of quarantine, through the gate.

    The gate runs on whatever claim the chunk can honestly make: its existing
    extraction (values cited from the source) or, for an unclassified chunk,
    a fresh verbatim quote. Either way the machine check is the same one any
    ingested chunk faces -- approval changes who is asking, not what is
    checked."""
    quarantined = next(
        (r for r in outcome.quarantined_chunks if r.chunk_id == chunk_id), None
    )
    if quarantined is None:
        return _refused(approval, RefusalReason.NO_CHUNK,
                        f"chunk {chunk_id} is not quarantined in {outcome.doc_id}")

    extraction = quarantined.extraction
    if extraction is None:
        extraction = _quote(raw, elements, chunk_id, quarantined.text)
        if extraction is None:
            return _refused(approval, RefusalReason.NOT_CITABLE,
                            "the chunk text cannot be cited from the source")
    elif _is_unclassified(extraction):
        quoted = _quote(raw, elements, chunk_id, quarantined.text)
        if quoted is not None:
            extraction = quoted

    tier = select_gate_tier(extraction)
    result = gate.check(extraction, tier=tier)
    if result.status != GroundingStatus.GROUNDED:
        return _refused(
            approval, RefusalReason.GATE_STILL_REFUSES,
            "approval cannot make an uncited value cited; the chunk needs a fix",
        )

    promoted = quarantined.model_copy(
        update={"verified": True, "extraction": extraction, "gate_tier": tier}
    )
    new_outcome = dataclasses.replace(
        outcome,
        verified_chunks=(*outcome.verified_chunks, promoted),
        quarantined_chunks=tuple(
            r for r in outcome.quarantined_chunks if r.chunk_id != chunk_id
        ),
    )
    return PromotionResult(
        outcome=new_outcome, refusal=None, approval=approval
    )


def _quote(raw: RawDocument, elements: list[Element], chunk_id: str, text: str):
    """Build the verbatim-quote extraction for a chunk's text, or None when
    the text cannot be cited -- a quote that fails fidelity is refused, never
    passed by the owner's authority.

    The chunk is rebuilt from its source element so the quote's provenance
    points at the raw page buffer: the segment spans carry the real char
    offsets the gate will re-derive against."""
    from rag_ingest.chunking import Chunk, ChunkKind, ChunkSegment, HeadingPath

    # Chunk ids may be namespaced (doc:kind:element); match the element the
    # chunk was built from, which is the only citation source we have.
    element = next(
        (
            e
            for e in elements
            if e.element_id == chunk_id or chunk_id.endswith(":" + e.element_id)
        ),
        None,
    )
    if element is None:
        return None
    rendered = element.rendered()
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=raw.doc_id,
        kind=ChunkKind.TEXT,
        text=rendered,
        heading_path=HeadingPath(),
        page_start=element.page,
        page_end=element.page,
        element_ids=(element.element_id,),
        estimated_tokens=len(rendered) // 4,
        segments=(
            ChunkSegment(
                element_id=element.element_id,
                page=element.page,
                chunk_start=0,
                chunk_end=len(rendered),
                raw_start=element.char_start,
                raw_end=element.char_end,
                bbox=element.bbox,
            ),
        ),
    )
    return ProseExtractor().extract(chunk)


def _is_unclassified(extraction) -> bool:
    from rag_ingest.extractors import Unclassified

    return isinstance(extraction, Unclassified)


def _refused(approval: Approval, reason: str, detail: str) -> PromotionResult:
    return PromotionResult(
        outcome=None,
        refusal=ApprovalRefusal(reason=reason, detail=detail),
        approval=approval,
    )


__all__ = [
    "Approval",
    "ApprovalRefusal",
    "PromotionResult",
    "RefusalReason",
    "promote_chunk",
]
