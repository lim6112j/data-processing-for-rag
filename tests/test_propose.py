"""Schema proposals: claims with evidence, never adopted rules.

The grammar proposer must derive fields from what actually recurs in the
samples, and an under-evidenced column is not a field. The contract built
from a proposal must pass the gate -- the proposal is the claim, the
contract-path gate is its verification.
"""

from __future__ import annotations

from rag_ingest.propose import propose
from rag_ingest.shapes import ShapeRegistry

TORQUE_ROWS = (
    ("1/4-28", "50-70", "110-160", "Coarse"),
    ("5/16-24", "100-140", "160-190", "Coarse"),
    ("3/8-24", "160-190", "270-300", "Fine"),
)


class TestGrammarProposer:
    def test_range_columns_become_a_constrained_pair(self):
        proposal = propose("tabular", TORQUE_ROWS, shape_name="torque_spec")

        names = [f.name for f in proposal.fields]
        assert "torque_min" in names
        max_spec = next(f for f in proposal.fields if f.name == "torque_max")
        assert max_spec.min_ref == "torque_min"

    def test_thread_column_is_text_required(self):
        proposal = propose("tabular", TORQUE_ROWS, shape_name="torque_spec")

        thread = next(f for f in proposal.fields if f.name == "thread_size")
        assert thread.kind == "text"
        assert thread.required is True

    def test_evidence_points_at_real_rows(self):
        proposal = propose("tabular", TORQUE_ROWS, shape_name="torque_spec")

        sample_values = [e.sample_value for e in proposal.evidence]
        assert "50-70" in sample_values       # a real cell from row 0
        assert "100-140" in sample_values     # a real cell from row 1

    def test_proposal_compiles_to_a_valid_contract(self):
        proposal = propose("tabular", TORQUE_ROWS, shape_name="torque_spec")

        contract = proposal.to_contract()

        registry = ShapeRegistry.with_builtins()
        registry.register(contract)
        assert registry.lookup("torque_spec") is contract

    def test_no_samples_is_an_honest_refusal(self):
        proposal = propose("tabular", (), shape_name="x")

        assert proposal.parseable is False
        assert proposal.fields == ()

    def test_unstructured_text_is_an_honest_refusal(self):
        # Prose rows: no recurring column pattern -- proposing a schema here
        # would be the check hallucinating structure. Refuse instead.
        prose_rows = (
            ("Castle nuts should be inspected", "for cracks"),
            ("Torque wrenches require", "periodic calibration"),
        )

        proposal = propose("prose", prose_rows, shape_name="prose_shape")

        assert proposal.parseable is False
        assert proposal.fields == ()

    def test_single_value_column_becomes_a_number_field(self):
        rows = (
            ("1/4-28", "100"),
            ("5/16-24", "225"),
        )

        proposal = propose("tabular", rows, shape_name="limits")

        assert any(
            f.kind == "number" and f.name != "torque_min" for f in proposal.fields
        )
