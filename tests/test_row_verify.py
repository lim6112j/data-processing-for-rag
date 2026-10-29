"""Row-level verification: an adopted row contract verifies real rows.

Adoption makes a contract effective; effectiveness means the next document
whose table rows carry the same shape is verified mechanically, row by row.
This module pins that loop:

  * a row whose cells match the adopted contract grounds -- status
    verified, per-cell provenance, no model anywhere;
  * a row with a wrong value is refused as a specific row, not silently
    absorbed -- the refusal names the row, the contract, and the status;
  * rows whose shape matches no adopted contract are left alone: an
    unverified row stays a discovery observation, never a guess.
"""

from __future__ import annotations

from rag_ingest.chunking import Chunker
from rag_ingest.docmodel import Element, ElementKind, RawDocument
from rag_ingest.grounding import GroundingGate
from rag_ingest.shapes import ShapeContract, ShapeRegistry
from rag_ingest.table_rows import row_observations

INVENTORY_ROWS = (
    ("품목코드", "품목명", "규격", "단위", "재고", "안전재고"),
    ("IV-001", "육안검사기", "KS-A 12mm", "개", "150", "30"),
    ("IV-002", "베어링6204", "20x47x14", "개", "2,400", "500"),
)


def _contract() -> ShapeContract:
    from rag_ingest.shapes import FieldSpec

    return ShapeContract(
        name="row:text,text,text,text,number,number_spec",
        tier="strict",
        fields=tuple(
            FieldSpec(name=f"field_{i}", kind=kind, required=True)
            for i, kind in enumerate(("text", "text", "text", "text", "number", "number"))
        ),
    )


def _doc_and_rows(rows=INVENTORY_ROWS):
    elements = [
        Element(element_id="p1", kind=ElementKind.PARAGRAPH, page=1,
                text="부품 재고 관리 대장. " * 20),
        Element(element_id="t1", kind=ElementKind.TABLE, page=1, text="",
                table_rows=rows),
    ]
    raw, located = RawDocument.from_elements("kr", elements)
    chunks = Chunker().chunk("kr", located)
    table_chunk = next(c for c in chunks if c.kind.value == "table")
    rows = row_observations(table_chunk, located)
    return raw, table_chunk, rows


class TestVerifyRows:
    def test_adopted_contract_verifies_matching_rows(self):
        from rag_ingest.row_verify import verify_rows

        raw, chunk, rows = _doc_and_rows()
        registry = ShapeRegistry()
        registry.register(_contract())
        gate = GroundingGate(raw, shapes=registry)

        verifications = verify_rows(rows, chunk, gate, registry)

        data_rows = [v for v in verifications if v.status == "grounded"]
        assert len(data_rows) == 2, [v.status for v in verifications]
        assert data_rows[0].cells[0] == "IV-001"
        # The refusal path is per-row: a row that cannot ground is named.
        grounded = [v for v in verifications if v.status == "grounded"]
        assert all(v.contract_name == _contract().name for v in grounded)

    def test_an_enum_violation_refuses_as_a_specific_row(self):
        """The constraint that bites across the board is the declared closed
        set: an out-of-enum status keeps the row's shape (both are text to
        the classifier) but violates the contract, so the gate refuses that
        one row by name."""
        from rag_ingest.row_verify import verify_rows
        from rag_ingest.shapes import FieldSpec

        contract = ShapeContract(
            name="row:text,text,text,text_spec",
            tier="strict",
            fields=(
                FieldSpec(name="field_0", kind="text", required=True),
                FieldSpec(name="field_1", kind="enum", required=True,
                          values=("PASS", "FAIL", "HOLD")),
                FieldSpec(name="field_2", kind="text", required=False),
                FieldSpec(name="field_3", kind="text", required=False),
            ),
        )
        rows = (
            ("부품코드", "상태", "판정", "메모"),
            ("AN310-4", "PASS", "합격", "정상"),
            ("MS20365-3", "폐기", "불합격", "손상"),
        )
        raw, chunk, observations = _doc_and_rows(rows)
        registry = ShapeRegistry()
        registry.register(contract)
        gate = GroundingGate(raw, shapes=registry)

        verifications = verify_rows(observations, chunk, gate, registry)

        refused = [v for v in verifications if v.status != "grounded"]
        # The header row is refused too: read as data, its status cell is
        # not in the closed set. That is the gate being honest about what
        # it was asked, not noise to suppress.
        assert any(v.cells[0] == "MS20365-3" and v.status == "mismatch" for v in refused)
        assert all("폐기" in v.row_text or "상태" in v.row_text for v in refused)

    def test_rows_matching_no_contract_are_left_alone(self):
        from rag_ingest.row_verify import verify_rows

        raw, chunk, rows = _doc_and_rows()
        gate = GroundingGate(raw, shapes=ShapeRegistry.with_builtins())

        # Builtins only: no row contract matches these Korean inventory rows.
        assert verify_rows(rows, chunk, gate, ShapeRegistry.with_builtins()) == ()


class TestContractMatchingByDeclaredKinds:
    """The grammar proposer can describe the same column structure under two
    names -- 'row:..._spec' from a routed row cluster and 'tabular_spec' from
    the blob cluster whose lines parse as rows. Matching by name would make
    the tabular adoption silently inert; matching by declared field kinds
    makes both drive row verification, because both ARE the same shape."""

    def _blob_contract(self):
        from rag_ingest.shapes import FieldSpec, ShapeContract

        return ShapeContract(
            name="tabular_spec",
            tier="strict",
            fields=tuple(
                FieldSpec(name=f"field_{i}", kind=kind, required=True)
                for i, kind in enumerate(("text", "text", "text", "text", "number", "number"))
            ),
        )

    def test_tabular_named_contract_verifies_rows(self):
        from rag_ingest.row_verify import verify_rows

        raw, chunk, rows = _doc_and_rows()
        registry = ShapeRegistry()
        registry.register(self._blob_contract())
        gate = GroundingGate(raw, shapes=registry)

        verifications = verify_rows(rows, chunk, gate, registry)

        grounded = [v for v in verifications if v.status == "grounded"]
        assert len(grounded) == 2
        assert all(v.contract_name == "tabular_spec" for v in grounded)

    def test_builtins_never_verify_table_rows(self):
        from rag_ingest.row_verify import verify_rows

        raw, chunk, rows = _doc_and_rows()
        gate = GroundingGate(raw, shapes=ShapeRegistry.with_builtins())

        assert verify_rows(rows, chunk, gate, ShapeRegistry.with_builtins()) == ()
