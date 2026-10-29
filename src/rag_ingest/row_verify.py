"""Row-level verification: an adopted row contract verifies real rows.

Adoption makes a contract effective; the point of effectiveness is the next
document. When a quarantined table chunk's rows carry a shape an adopted
contract describes (the contract name carries the row signature it was
proposed from), each row is verified mechanically: every cell becomes a
claimed field value with its own provenance span inside the chunk, and the
standard gate -- the same span re-derivation that runs everywhere else --
decides. No model is involved; this is T1 at row granularity.

Rows whose shape matches no adopted contract are left untouched: they stay
discovery observations. Verifying them would require inventing a mapping
from columns to fields, which is exactly the guess this pipeline refuses.

A refusal is per-row and named: the row's text, the contract, the status.
A wrong value cannot disappear into a chunk-level blob.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rag_ingest.chunking import Chunk
from rag_ingest.grounding import GroundingGate
from rag_ingest.shapes import ShapeRegistry
from rag_ingest.table_rows import RowObservation


class RowVerification(BaseModel):
    """The gate's verdict on one table row against one adopted contract."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    page: int
    row_text: str
    contract_name: str
    status: str
    cells: tuple[str, ...]


def _cell_span(chunk: Chunk, row: RowObservation, cell: str, cursor: int):
    """A chunk-local provenance span for one cell, searched left to right.

    The cursor prevents a value that repeats across columns (a unit, a zero)
    from grounding every column to its first occurrence: each cell is found
    after the previous one, in reading order. A cell that cannot be located
    yields no span -- the field will abstain, which is the honest outcome
    for text the parser rendered differently than it reported.
    """
    idx = row.row_text.find(cell, cursor)
    if idx < 0:
        return None, cursor
    start = row.chunk_start + idx
    return chunk.span_for(start, start + len(cell)), idx + len(cell)


def verify_rows(
    rows: tuple[RowObservation, ...] | list[RowObservation],
    chunk: Chunk,
    gate: GroundingGate,
    registry: ShapeRegistry,
) -> tuple[RowVerification, ...]:
    """Verify rows against adopted row contracts; pass unmatched rows through.

    Returns only rows that matched a contract -- verified or refused. Rows
    with no matching contract are absent from the result, which is their
    honest state: still waiting for the owner, not silently passed.
    """
    from rag_ingest.ontology import FieldValue

    contracts = {
        name[len("row:"):-len("_spec")]: contract
        for name, contract in registry._contracts.items()
        if name.startswith("row:") and name.endswith("_spec")
    }
    if not contracts:
        return ()

    from rag_ingest.discovery import row_signatures

    signatures = row_signatures(rows)
    verifications: list[RowVerification] = []
    for row in rows:
        signature = signatures[id(row)][len("row:"):]
        contract = contracts.get(signature)
        if contract is None or len(contract.fields) != len(row.cells):
            continue
        claimed: dict[str, FieldValue] = {}
        cursor = 0
        for field, cell in zip(contract.fields, row.cells, strict=False):
            span, cursor = _cell_span(chunk, row, cell, cursor)
            if span is not None:
                claimed[field.name] = FieldValue(value=cell, provenance=span)
        result = gate.check_contract(contract.name, claimed)
        verifications.append(
            RowVerification(
                doc_id=row.doc_id,
                page=row.page,
                row_text=row.row_text,
                contract_name=contract.name,
                status=result.status.value,
                cells=row.cells,
            )
        )
    return tuple(verifications)


__all__ = ["RowVerification", "verify_rows"]
