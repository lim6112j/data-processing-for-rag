"""Verified-chunk export: the pipeline's honest output is the only thing
that may leave for retrieval. Export is a mechanical projection of records
that already passed the gate -- no re-judgment, no filtering beyond tier."""

import json

from rag_ingest.export import export_verified
from rag_ingest.ontology import ChunkRecord, GateTier


def _record(text: str, tier: GateTier | None = GateTier.STRICT,
            chunk_id: str = "c1", doc_id: str = "d1") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id, doc_id=doc_id, text=text, embed_text=text,
        verified=True, gate_tier=tier,
    )


class TestExportVerified:
    def test_verified_chunks_project_to_rows_with_tier(self):
        rows = export_verified([_record("본 대장은 재고 현황을 기록한다.")])
        assert len(rows) == 1
        assert rows[0].chunk_id == "c1"
        assert rows[0].doc_id == "d1"
        assert rows[0].text == "본 대장은 재고 현황을 기록한다."
        assert rows[0].gate_tier == GateTier.STRICT

    def test_quarantined_records_are_never_exportable(self):
        """The invariant in exportable form: an unverified record cannot be
        projected into retrieval, no matter who asks."""
        record = _record("요청자의 비밀 노트", tier=None).model_copy(
            update={"verified": False}
        )
        assert export_verified([record]) == ()

    def test_jsonl_roundtrip_is_stable_and_metadata_rides_along(self):
        rows = export_verified([
            _record("재고가 안전재고 이하로 내려가면 발주를 개시한다.",
                    tier=GateTier.LENIENT, chunk_id="c2"),
        ])
        line = json.loads(rows[0].to_jsonl())
        assert line["chunk_id"] == "c2"
        assert line["gate_tier"] == "lenient"
        assert "발주" in line["text"]
        assert line["export_format"] == "rag_export_v1"

    def test_rows_are_sorted_for_reproducible_push(self):
        rows = export_verified([
            _record("b", chunk_id="c2"),
            _record("a", chunk_id="c1", doc_id="d2"),
        ])
        keys = [(r.doc_id, r.chunk_id) for r in rows]
        assert keys == sorted(keys)


class TestApprovalInExport:
    """Owner-approved chunks ride into export with their history, so the
    downstream store can tell machine-verified from owner-approved instead
    of mixing the two silently."""

    def test_row_carries_approval_when_present(self):
        from datetime import date

        from rag_ingest.promotion import Approval

        record = _record("본 대장은 기록한다.")
        rows = export_verified(
            [record],
            approvals={"c1": Approval(reviewer="owner-1", approved_on=date(2026, 11, 1))},
        )
        line = json.loads(rows[0].to_jsonl())
        assert line["approval"] == {
            "reviewer": "owner-1", "approved_on": "2026-11-01",
        }

    def test_machine_verified_rows_have_no_approval_key(self):
        rows = export_verified([_record("no approval here")])
        assert "approval" not in json.loads(rows[0].to_jsonl())
