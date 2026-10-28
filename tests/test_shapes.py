"""Declarative shape contracts: a user-approved shape runs without a reboot.

The old constraint: a shape was a Python class, so its identity was fixed at
import time -- isinstance chains in the gate, the tier table and the checks all
referenced classes that cannot exist until a rebuild. A user approving a new
shape in the UI therefore could not see it take effect in the running process.

The fix keeps every invariant and changes the representation:

* A shape becomes a ``ShapeContract`` -- tier and fields declared as data,
  approved by the owner, never chosen by a model.
* The contract lives in a ``ShapeRegistry``; registering it makes it effective
  for the rest of the process lifetime, no restart.
* Persistence is explicit: a contract that has passed its gold cases is also
  serialized (``to_definition``/``from_definition``), so a restart restores the
  recorded contract instead of losing the capability.
* Tier assignment stays a lookup -- by contract name now, which is exactly the
  discipline that prevents silent reclassification. The dangerous direction
  (LENIENT on a typed record) is still refused structurally.

What must NOT change: the gate still re-derives every value from its span, and
nothing reaches retrieval without that check. A dynamic contract earns access
to the same gate, not a weaker one.
"""

from __future__ import annotations

import pytest

from rag_ingest.chunking import Chunker
from rag_ingest.docmodel import Element, ElementKind, ProvenanceRef, RawDocument
from rag_ingest.gate_tier import GateTier
from rag_ingest.ontology import (
    AbstainReason,
    FieldAbstention,
    FieldValue,
)
from rag_ingest.shapes import (
    FieldSpec,
    ShapeContract,
    ShapeRegistry,
)

BBOX = (10.0, 20.0, 400.0, 60.0)


def _ref(page: int = 1) -> ProvenanceRef:
    return ProvenanceRef(page=page, char_start=0, char_end=40, element_id="e")


def _value(value: str) -> FieldValue:
    return FieldValue(value=value, provenance=_ref())


def _abstain() -> FieldAbstention:
    return FieldAbstention(reason=AbstainReason.NOT_PRESENT)


TORQUE_CONTRACT = ShapeContract(
    name="torque_spec",
    tier=GateTier.STRICT,
    fields=(
        FieldSpec(name="thread_size", kind="text", required=True),
        FieldSpec(name="torque_min", kind="number", required=True, ge=0),
        FieldSpec(name="torque_max", kind="number", required=True, min_ref="torque_min"),
        FieldSpec(name="series", kind="enum", values=("Coarse", "Fine"), required=True),
    ),
)


class TestContractDefinition:
    def test_contract_holds_its_declaration(self):
        assert TORQUE_CONTRACT.name == "torque_spec"
        assert TORQUE_CONTRACT.tier is GateTier.STRICT
        assert TORQUE_CONTRACT.required_field_names == (
            "thread_size", "torque_min", "torque_max", "series",
        )

    def test_contract_serializes_and_restores(self):
        # The restart story: the approved contract is recorded, a fresh process
        # restores it, and the restored contract is equal to the original.
        payload = TORQUE_CONTRACT.to_definition()

        restored = ShapeContract.from_definition(payload)

        assert restored == TORQUE_CONTRACT

    def test_tier_declaration_cannot_be_lenient_for_a_typed_shape(self):
        # The structural invariant, now at declaration time: a LENIENT tier is
        # only meaningful for verbatim prose, so a typed contract declaring it
        # is an invalid contract, not a running hazard.
        with pytest.raises(ValueError):
            ShapeContract(
                name="bad",
                tier=GateTier.LENIENT,
                fields=(FieldSpec(name="x", kind="text", required=True),),
            )


class TestRegistry:
    def test_registration_makes_the_shape_effective_immediately(self):
        registry = ShapeRegistry()

        assert registry.lookup("torque_spec") is None

        registry.register(TORQUE_CONTRACT)

        assert registry.lookup("torque_spec") is TORQUE_CONTRACT

    def test_registration_is_rejected_for_a_duplicate_name(self):
        # Two contracts with one name would make tier assignment ambiguous --
        # exactly the ambiguity the lookup-table discipline exists to prevent.
        registry = ShapeRegistry()
        registry.register(TORQUE_CONTRACT)

        with pytest.raises(ValueError):
            registry.register(TORQUE_CONTRACT)

    def test_built_in_shapes_are_present(self):
        # The existing ontology shapes are registered as contracts at startup,
        # so the pipeline's behaviour is identical before and after the pivot.
        registry = ShapeRegistry.with_builtins()

        assert registry.lookup("invoice").tier is GateTier.STRICT
        assert registry.lookup("report").tier is GateTier.STANDARD
        assert registry.lookup("prose").tier is GateTier.LENIENT


class TestContractDrivenGate:
    """The gate honours a dynamically registered contract -- no restart."""

    def _registry(self) -> ShapeRegistry:
        registry = ShapeRegistry.with_builtins()
        registry.register(TORQUE_CONTRACT)
        return registry

    def _ingest_with_contract(self, text: str, field_values: dict[str, str]):
        from rag_ingest.grounding import GroundingGate

        elements = [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1, text=text)]
        raw, located = RawDocument.from_elements("d", elements)
        chunk = Chunker().chunk("d", located)[0]
        ref = ProvenanceRef(
            page=1,
            char_start=0,
            char_end=len(raw.pages[0].text),
            element_id="e1",
        )
        # A contract-driven extraction claims exactly the declared fields,
        # each grounded against the whole page -- same convention as the
        # gold-set replay extractor.
        claimed = {
            name: (
                FieldValue(value=val, provenance=ref) if val else _abstain()
            )
            for name, val in field_values.items()
        }
        gate = GroundingGate(raw, shapes=self._registry())
        return gate, chunk, claimed

    def test_registered_contract_field_set_is_used_by_the_gate(self):
        gate, _chunk, claimed = self._ingest_with_contract(
            "1/4-28  50-70  Coarse",
            {
                "thread_size": "1/4-28",
                "torque_min": "50",
                "torque_max": "70",
                "series": "Coarse",
            },
        )

        result = gate.check_contract("torque_spec", claimed)

        assert result.verified

    def test_contract_required_fields_are_enforced(self):
        # Drop torque_max: a contract with required fields must produce
        # INCOMPLETE, not a pass -- same required-evidence rule as the
        # built-in invoice shape.
        gate, _chunk, claimed = self._ingest_with_contract(
            "1/4-28  50-70  Coarse",
            {
                "thread_size": "1/4-28",
                "torque_min": "50",
                "torque_max": None,
                "series": "Coarse",
            },
        )

        from rag_ingest.grounding import GroundingStatus

        result = gate.check_contract("torque_spec", claimed)

        assert result.status is GroundingStatus.INCOMPLETE

    def test_unregistered_contract_name_is_refused(self):
        gate, _chunk, claimed = self._ingest_with_contract(
            "1/4-28  50-70  Coarse", {"thread_size": "1/4-28"}
        )

        with pytest.raises(KeyError):
            gate.check_contract("unknown_shape", claimed)


class TestConstraintCompilation:
    """Declared constraints become deterministic checks -- no per-shape code.

    This is the piece that makes torque tables safe: min <= max is a
    declaration on the contract, and the gate compiles it. Before this, a
    wrong range (max < min) passed grounding (both numbers are verbatim) and
    entered the database silently.
    """

    def _claim(self, values: dict[str, str | None]) -> dict:
        ref = ProvenanceRef(page=1, char_start=0, char_end=60, element_id="e")
        return {
            name: (FieldValue(value=v, provenance=ref) if v is not None else _abstain())
            for name, v in values.items()
        }

    def _gate_for(self, contract: ShapeContract):
        from rag_ingest.grounding import GroundingGate, GroundingStatus

        elements = [
            Element(
                element_id="e1",
                kind=ElementKind.PARAGRAPH,
                page=1,
                text="1/4-28  70-50  Coarse",
            )
        ]
        raw, _ = RawDocument.from_elements("d", elements)
        return GroundingGate(raw, shapes=ShapeRegistry.with_builtins()), GroundingStatus

    def _register(self, contract: ShapeContract):
        registry = ShapeRegistry.with_builtins()
        registry.register(contract)
        return registry

    def test_min_ref_violation_is_mismatch(self):
        # The document verbatim says "70-50" -- grounding passes both numbers.
        # Only the declared max>=min constraint can see the contradiction.
        contract = ShapeContract(
            name="torque_spec",
            tier=GateTier.STRICT,
            fields=(
                FieldSpec(name="thread_size", kind="text", required=True),
                FieldSpec(name="torque_min", kind="number", required=True),
                FieldSpec(
                    name="torque_max", kind="number", required=True, min_ref="torque_min"
                ),
            ),
        )
        registry = self._register(contract)
        from rag_ingest.grounding import GroundingGate, GroundingStatus

        elements = [
            Element(
                element_id="e1",
                kind=ElementKind.PARAGRAPH,
                page=1,
                text="1/4-28  70-50  Coarse",
            )
        ]
        raw, _ = RawDocument.from_elements("d", elements)
        gate = GroundingGate(raw, shapes=registry)
        ref = ProvenanceRef(page=1, char_start=0, char_end=len(raw.pages[0].text), element_id="e1")

        result = gate.check_contract(
            "torque_spec",
            {
                "thread_size": FieldValue(value="1/4-28", provenance=ref),
                "torque_min": FieldValue(value="70", provenance=ref),
                "torque_max": FieldValue(value="50", provenance=ref),
            },
        )

        assert result.status is GroundingStatus.MISMATCH
        bad = [i for i in result.issues if i.status is GroundingStatus.MISMATCH]
        assert bad[0].field == "torque_max"
        assert "torque_min" in bad[0].message

    def test_ge_constraint_is_enforced(self):
        contract = ShapeContract(
            name="pressure",
            tier=GateTier.STRICT,
            fields=(
                FieldSpec(name="psi", kind="number", required=True, ge=0),
            ),
        )
        registry = self._register(contract)
        from rag_ingest.grounding import GroundingGate, GroundingStatus

        elements = [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1, text="psi -5")]
        raw, _ = RawDocument.from_elements("d", elements)
        gate = GroundingGate(raw, shapes=registry)
        ref = ProvenanceRef(page=1, char_start=0, char_end=len(raw.pages[0].text), element_id="e1")

        result = gate.check_contract(
            "pressure", {"psi": FieldValue(value="-5", provenance=ref)}
        )

        assert result.status is GroundingStatus.MISMATCH

    def test_enum_violation_is_mismatch(self):
        registry = self._register(TORQUE_CONTRACT)
        from rag_ingest.grounding import GroundingGate, GroundingStatus

        elements = [
            Element(
                element_id="e1", kind=ElementKind.PARAGRAPH, page=1,
                text="1/4-28 50-70 Medium",
            )
        ]
        raw, _ = RawDocument.from_elements("d", elements)
        gate = GroundingGate(raw, shapes=registry)
        ref = ProvenanceRef(page=1, char_start=0, char_end=len(raw.pages[0].text), element_id="e1")

        result = gate.check_contract(
            "torque_spec",
            {
                "thread_size": FieldValue(value="1/4-28", provenance=ref),
                "torque_min": FieldValue(value="50", provenance=ref),
                "torque_max": FieldValue(value="70", provenance=ref),
                "series": FieldValue(value="Medium", provenance=ref),
            },
        )

        assert result.status is GroundingStatus.MISMATCH
        bad = [i for i in result.issues if i.status is GroundingStatus.MISMATCH]
        assert bad[0].field == "series"

    def test_satisfied_constraints_still_verify(self):
        registry = self._register(TORQUE_CONTRACT)
        from rag_ingest.grounding import GroundingGate

        elements = [
            Element(
                element_id="e1", kind=ElementKind.PARAGRAPH, page=1,
                text="1/4-28 50-70 Coarse",
            )
        ]
        raw, _ = RawDocument.from_elements("d", elements)
        gate = GroundingGate(raw, shapes=registry)
        ref = ProvenanceRef(page=1, char_start=0, char_end=len(raw.pages[0].text), element_id="e1")

        result = gate.check_contract(
            "torque_spec",
            {
                "thread_size": FieldValue(value="1/4-28", provenance=ref),
                "torque_min": FieldValue(value="50", provenance=ref),
                "torque_max": FieldValue(value="70", provenance=ref),
                "series": FieldValue(value="Coarse", provenance=ref),
            },
        )

        assert result.verified

    def test_unparseable_number_is_unverifiable_not_passed(self):
        # A number field whose value cannot parse ("TBD") must not slip
        # through the numeric constraints by failing to parse -- silence is
        # for abstentions, not for unparseable claimed values.
        contract = ShapeContract(
            name="pressure",
            tier=GateTier.STRICT,
            fields=(FieldSpec(name="psi", kind="number", required=True, ge=0),),
        )
        registry = self._register(contract)
        from rag_ingest.grounding import GroundingGate, GroundingStatus

        elements = [
            Element(
                element_id="e1", kind=ElementKind.PARAGRAPH, page=1,
                text="psi is fine",
            )
        ]
        raw, _ = RawDocument.from_elements("d", elements)
        gate = GroundingGate(raw, shapes=registry)
        ref = ProvenanceRef(page=1, char_start=0, char_end=len(raw.pages[0].text), element_id="e1")

        result = gate.check_contract(
            "pressure", {"psi": FieldValue(value="fine", provenance=ref)}
        )

        assert result.status is GroundingStatus.UNVERIFIABLE


class TestAdoptionGate:
    """Adoption is gated, not immediate: the contract must pass its gold cases
    in the same process, against the same gate, before it registers.

    '승인 ≠ 채택' in code: approval creates the contract and its cases; the
    cases run; only a full pass registers. A failed case is recorded as a
    falsification the owner can act on, and the registry stays untouched.
    """

    def _contract(self) -> ShapeContract:
        return ShapeContract(
            name="torque_spec",
            tier=GateTier.STRICT,
            fields=(
                FieldSpec(name="thread_size", kind="text", required=True),
                FieldSpec(name="torque_min", kind="number", required=True),
                FieldSpec(name="torque_max", kind="number", required=True,
                          min_ref="torque_min"),
            ),
        )

    def _gate(self, text: str = "1/4-28  50-70  Coarse"):
        from rag_ingest.grounding import GroundingGate
        elements = [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1, text=text)]
        raw, _located = RawDocument.from_elements("d", elements)
        ref = ProvenanceRef(page=1, char_start=0, char_end=len(raw.pages[0].text),
                            element_id="e1")
        return GroundingGate(raw, shapes=ShapeRegistry.with_builtins()), ref

    def test_passing_cases_register_the_contract(self):
        from rag_ingest.shapes import GoldCaseSpec, adopt_contract

        gate, _ref = self._gate()
        cases = GoldCaseSpec.from_claims("torque_spec", [
            {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70"},
        ])
        registry = ShapeRegistry.with_builtins()

        result = adopt_contract(gate, registry, self._contract(), cases)

        assert result.registered is True
        assert registry.lookup("torque_spec") is not None
        assert result.failed_cases == ()

    def test_enum_case_confirms_enforcement_and_registers(self):
        # The owner declares an enum; a case asserts the gate MUST refuse an
        # out-of-enum value. The gate refuses, the case's expectation is met,
        # and adoption proceeds -- the constraint is proven, not assumed.
        from rag_ingest.shapes import GoldCaseSpec, adopt_contract

        gate, _ref = self._gate()
        contract = ShapeContract(
            name="torque_spec",
            tier=GateTier.STRICT,
            fields=(
                FieldSpec(name="thread_size", kind="text", required=True),
                FieldSpec(name="torque_min", kind="number", required=True),
                FieldSpec(name="torque_max", kind="number", required=True,
                          min_ref="torque_min"),
                FieldSpec(name="series", kind="enum",
                          values=("Coarse", "Fine"), required=True),
            ),
        )
        cases = GoldCaseSpec.from_claims("torque_spec", [
            {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70",
             "series": "Coarse"},
            {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70",
             "series": "Medium", "expected": "mismatch"},  # gate MUST refuse
        ])
        registry = ShapeRegistry.with_builtins()

        result = adopt_contract(gate, registry, contract, cases)

        assert result.registered is True
        assert registry.lookup("torque_spec") is contract
        assert result.cases_run == 2

    def test_case_expecting_refusal_but_gate_passing_is_a_recorded_failure(self):
        # The falsification direction: the case EXPECTS the gate to refuse,
        # the gate passes it -- that is a broken contract, and adoption must
        # block with the failure recorded as a falsification.
        from rag_ingest.shapes import GoldCaseSpec, adopt_contract

        gate, _ref = self._gate(text="1/4-28  50-70  Medium  Coarse")
        contract = ShapeContract(
            name="no_enum",
            tier=GateTier.STRICT,
            fields=(
                FieldSpec(name="thread_size", kind="text", required=True),
                FieldSpec(name="series", kind="text", required=True),
            ),
        )
        # 'Medium' IS verbatim in the span, so grounding passes it -- the
        # contract declares no enum, so nothing refuses it. The case expected
        # a refusal: that is a falsified contract.
        cases = GoldCaseSpec.from_claims("no_enum", [
            {"thread_size": "1/4-28", "series": "Medium",
             "expected": "mismatch"},  # gate will pass it
        ])
        registry = ShapeRegistry.with_builtins()

        result = adopt_contract(gate, registry, contract, cases)

        assert result.registered is False
        assert registry.lookup("no_enum") is None
        assert len(result.failed_cases) == 1
        failure = result.failed_cases[0]
        assert failure.case_index == 0
        assert failure.expected == "mismatch"
        assert failure.got == "grounded"

    def test_expected_quarantine_case_passes_when_the_gate_refuses(self):
        # The falsification direction: a case that EXPECTS refusal (an
        # inverted range must be caught). If the gate failed to refuse, that
        # is a hole in the contract, and adoption is blocked.
        from rag_ingest.shapes import GoldCaseSpec, adopt_contract

        gate, _ref = self._gate()
        contract = self._contract()
        cases = GoldCaseSpec.from_claims("torque_spec", [
            {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70"},
            {"thread_size": "1/4-28", "torque_min": "70", "torque_max": "50",
             "expected": "mismatch"},  # inverted: the gate MUST refuse
        ])
        registry = ShapeRegistry.with_builtins()

        result = adopt_contract(gate, registry, contract, cases)

        assert result.registered is True
        assert result.failed_cases == ()

    def test_gate_that_fails_to_refuse_blocks_adoption(self):
        # The critical adversarial case: the schema declares a constraint,
        # a gold case expects it to be refused, and the gate passes it.
        # That is a broken contract -- registering it would silently admit
        # the value class the owner declared load-bearing.
        from rag_ingest.shapes import GoldCaseSpec, adopt_contract

        gate, _ref = self._gate()
        # A contract with NO constraints: an inverted range cannot be caught.
        contract = ShapeContract(
            name="loose_range",
            tier=GateTier.STRICT,
            fields=(
                FieldSpec(name="lo", kind="number", required=True),
                FieldSpec(name="hi", kind="number", required=True),
            ),
        )
        cases = GoldCaseSpec.from_claims("loose_range", [
            {"lo": "70", "hi": "50", "expected": "mismatch"},  # will PASS -> FAIL case
        ])
        registry = ShapeRegistry.with_builtins()

        result = adopt_contract(gate, registry, contract, cases)

        assert result.registered is False
        assert len(result.failed_cases) == 1
        assert result.failed_cases[0].expected == "mismatch"
        assert result.failed_cases[0].got == "grounded"

    def test_default_expectation_is_verified(self):
        from rag_ingest.shapes import GoldCaseSpec

        cases = GoldCaseSpec.from_claims("x", [{"a": "1"}, {"a": "2", "expected": "mismatch"}])

        assert cases[0].expected == "verified"
        assert cases[1].expected == "mismatch"
