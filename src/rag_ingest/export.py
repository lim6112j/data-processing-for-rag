"""Verified-chunk export: the pipeline's honest output is the only thing
that may leave for retrieval.

Export is a mechanical projection of records that already passed the gate --
no re-judgment happens here. A quarantined record cannot be projected into
retrieval no matter who asks: the reviewer's question is still open, and
answering retrieval with an open question would launder withholding into
knowledge. The tier rides along so a downstream store can distinguish
"verified by contract arithmetic" from "quoted verbatim" -- mixing those
silently is the ambiguity this project exists to remove."""

from __future__ import annotations

import json
from dataclasses import dataclass

from rag_ingest.ontology import ChunkRecord, GateTier

#: Bumped when the row shape changes; consumers key their validation on it.
EXPORT_FORMAT = "rag_export_v1"


@dataclass(frozen=True)
class ExportRow:
    """One verified chunk as it leaves the pipeline for a retrieval store."""

    chunk_id: str
    doc_id: str
    text: str
    gate_tier: GateTier

    def to_jsonl(self) -> str:
        """One JSON line carrying identity, text, and the tier that admitted it."""
        return json.dumps(
            {
                "export_format": EXPORT_FORMAT,
                "chunk_id": self.chunk_id,
                "doc_id": self.doc_id,
                "text": self.text,
                "gate_tier": self.gate_tier.value,
            },
            ensure_ascii=False,
        )


def export_verified(records: list[ChunkRecord] | tuple[ChunkRecord, ...]) -> tuple[ExportRow, ...]:
    """Project verified records into export rows, sorted for a stable push order.

    Unverified records vanish here rather than raising: a bulk caller mixing
    verified and quarantined records is exporting a batch, and the batch's
    honest content is exactly its verified part. The sort makes repeated
    pushes -- and diffs of pushed data -- reproducible."""
    rows = [
        ExportRow(
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            text=r.text,
            gate_tier=r.gate_tier,
        )
        for r in records
        if r.verified
    ]
    rows.sort(key=lambda row: (row.doc_id, row.chunk_id))
    return tuple(rows)


__all__ = ["EXPORT_FORMAT", "ExportRow", "export_verified"]
