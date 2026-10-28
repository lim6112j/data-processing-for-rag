# ruff: noqa: RUF001  -- Korean spec notation uses the real multiplication sign

"""The LLM proposer layer: proposals for clusters the grammar cannot read.

Contract with the rest of the system:

  * same ``Proposal`` shape the grammar proposer emits, labelled
    ``source="llm"`` -- the owner sees who proposed what;
  * the model only *claims*; every claim is checked by the adoption gate's
    deterministic cases before anything is registered;
  * an unparseable, off-schema, or transport-failed response is an honest
    ``cannot propose`` -- never a guessed schema, never a fabricated field;
  * evidence must point at real sample rows: a claimed field with no
    supporting value in the samples is refused, because a field nobody can
    ground is worse than no field.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from rag_ingest.llm_propose import LlmProposer

KOREAN_INVENTORY_ROWS = (
    ("품목명", "규격", "단위", "재고", "안전재고"),
    ("육안검사기", "KS-A 12mm", "개", "150", "30"),
    ("베어링6204", "20×47×14", "개", "2,400", "500"),
    ("윤활유ISO32", "드럼", "통", "12", "3"),
)

# The model's answer for the Korean inventory cluster: per-field claims,
# each with the column index and a sample value the field was seen holding.
INVENTORY_PAYLOAD = {
    "propose": True,
    "note": "Korean inventory table; specification column is composite text",
    "fields": [
        {"name": "item_name", "kind": "text", "column": 0,
         "sample_value": "육안검사기", "row_index": 1, "required": True},
        {"name": "spec", "kind": "text", "column": 1,
         "sample_value": "20×47×14", "row_index": 2, "required": True},
        {"name": "unit", "kind": "enum", "column": 2, "values": ["개", "통"],
         "sample_value": "개", "row_index": 1, "required": True},
        {"name": "stock", "kind": "number", "column": 3,
         "sample_value": "150", "row_index": 1, "required": True},
        {"name": "reorder_point", "kind": "number", "column": 4,
         "sample_value": "30", "row_index": 1, "required": True,
         "min_ref": "stock"},
    ],
}


class _FakeCompletions:
    def __init__(self, payload: str | None = None, raises: Exception | None = None):
        self._payload = payload
        self._raises = raises
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._payload))]
        )


class _FakeClient:
    def __init__(self, payload: str | None = None, raises: Exception | None = None):
        self.chat = SimpleNamespace(completions=_FakeCompletions(payload, raises))


def _proposer(payload=None, raises=None) -> LlmProposer:
    client = _FakeClient(
        payload=json.dumps(payload) if payload is not None else None, raises=raises
    )
    return LlmProposer(client=client, model="test-model")


class TestLlmProposer:
    def test_proposes_fields_for_a_korean_inventory_cluster(self):
        proposal = _proposer(INVENTORY_PAYLOAD).propose(
            "row:text,text,empty,empty,empty", KOREAN_INVENTORY_ROWS
        )

        assert proposal.parseable
        assert proposal.source == "llm"
        names = [f.name for f in proposal.fields]
        assert names == ["item_name", "spec", "unit", "stock", "reorder_point"]
        # The enum constraint came from the model's claim about a closed set.
        unit = next(f for f in proposal.fields if f.name == "unit")
        assert unit.kind == "enum"
        assert set(unit.values) == {"개", "통"}
        # The cross-field constraint is carried the same way the grammar
        # proposer expresses it, so the gate checks it identically.
        reorder = next(f for f in proposal.fields if f.name == "reorder_point")
        assert reorder.min_ref == "stock"

    def test_every_claimed_field_is_backed_by_a_real_sample_value(self):
        proposal = _proposer(INVENTORY_PAYLOAD).propose(
            "row:text,text,empty,empty,empty", KOREAN_INVENTORY_ROWS
        )

        # Each field's evidence must be locatable in the claimed row.
        for field in proposal.fields:
            evidence = [e for e in proposal.evidence if e.sample_value]
            assert evidence, f"field {field.name} has no evidence"
        row_2 = KOREAN_INVENTORY_ROWS[2]
        assert "20×47×14" in row_2  # the sample the model cited exists

    def test_a_claim_about_a_cell_that_is_not_there_is_refused(self):
        # The model cites a value that does not exist in the samples:
        # fabricated evidence -> cannot propose, not a guessed schema.
        payload = {
            "propose": True,
            "note": "",
            "fields": [
                {"name": "made_up", "kind": "text", "column": 0,
                 "sample_value": "이런값은없다", "row_index": 1, "required": True},
            ],
        }
        proposal = _proposer(payload).propose(
            "row:text,text,empty,empty,empty", KOREAN_INVENTORY_ROWS
        )

        assert not proposal.parseable
        assert proposal.fields == ()
        assert "evidence" in proposal.note.lower()

    def test_model_declining_is_an_honest_refusal(self):
        payload = {
            "propose": False,
            "note": "sample rows are too few to distinguish column roles",
            "fields": [],
        }
        proposal = _proposer(payload).propose(
            "row:text,text,empty,empty,empty", KOREAN_INVENTORY_ROWS
        )

        assert not proposal.parseable
        assert proposal.fields == ()
        assert "too few" in proposal.note

    def test_malformed_response_is_an_honest_refusal(self):
        proposal = _proposer("not json at all").propose(
            "row:text,text,empty,empty,empty", KOREAN_INVENTORY_ROWS
        )

        assert not proposal.parseable
        assert proposal.fields == ()

    def test_transport_failure_is_an_honest_refusal(self):
        proposal = _proposer(raises=RuntimeError("network down")).propose(
            "row:text,text,empty,empty,empty", KOREAN_INVENTORY_ROWS
        )

        assert not proposal.parseable
        assert proposal.fields == ()
        assert "transport" in proposal.note.lower()

    def test_field_outside_the_sample_width_is_dropped(self):
        # A claimed column beyond the row width is unfounded structure.
        payload = {
            "propose": True,
            "note": "",
            "fields": [
                {"name": "beyond", "kind": "text", "column": 9,
                 "sample_value": "x", "row_index": 1, "required": True},
            ],
        }
        proposal = _proposer(payload).propose(
            "row:text,text,empty,empty,empty", KOREAN_INVENTORY_ROWS
        )

        assert not proposal.parseable

    def test_declared_enum_values_must_occur_in_the_samples(self):
        # The model claims an enum with a value the samples never show:
        # the claim cannot be grounded, so it is refused.
        payload = {
            "propose": True,
            "note": "",
            "fields": [
                {"name": "unit", "kind": "enum", "column": 2,
                 "values": ["개", "통", "롤"], "sample_value": "개",
                 "row_index": 1, "required": True},
            ],
        }
        proposal = _proposer(payload).propose(
            "row:text,text,empty,empty,empty", KOREAN_INVENTORY_ROWS
        )

        assert not proposal.parseable
