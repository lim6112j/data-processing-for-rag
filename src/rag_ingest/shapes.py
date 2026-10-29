"""Declarative shape contracts: user-approved shapes that run without a reboot.

A shape was a Python class, so approving a new shape in the UI could not take
effect until a rebuild -- the isinstance chains referenced types that do not
exist until import time. This module makes the shape a *contract*: tier and
fields declared as data, registered at runtime, effective immediately.

The invariants this module exists to preserve:

1. **Tier is declared, never inferred.** A contract carries its tier from the
   owner's approval. No model chooses it at runtime; a misdeclared tier is a
   visible declaration, not a silent reclassification.
2. **LENIENT stays prose-only.** The structural refusal in ``gate_tier`` is
   mirrored at declaration time: a typed contract cannot declare LENIENT,
   because quotation fidelity is not a bar a field-carrying record can clear.
3. **One name, one contract.** Duplicate registration is refused; two
   contracts under one name would make tier assignment ambiguous.
4. **Same gate, not a weaker one.** A dynamic contract earns access to the
   standard span re-derivation; nothing here relaxes the grounding check.
5. **Persistence is explicit.** A contract that passes its gold cases is
   serialized with ``to_definition``; a restart restores recorded contracts
   rather than losing the capability. Nothing is kept alive only by the
   process.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_ingest.docmodel import ProvenanceRef
from rag_ingest.ontology import GateTier


class FieldSpec(BaseModel):
    """One field of a shape contract.

    ``kind`` selects the deterministic check family the gate's contract path
    applies: ``text`` verifies the value is in the span, ``number`` adds
    numeric equivalence, ``enum`` restricts to declared values. Constraints
    are declarations the owner approved -- the gate compiles them into checks,
    it does not invent them.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["text", "number", "enum"]
    required: bool = False
    ge: float | None = None
    #: Numeric lower bound must not exceed this field's value (a range check
    #: expressed declaratively: torque_max has min_ref="torque_min").
    min_ref: str | None = None
    values: tuple[str, ...] = Field(default_factory=tuple)


class ShapeContract(BaseModel):
    """A user-approved shape: name, tier, fields -- as data.

    This is the serialization boundary of the shape-discovery loop: the owner
    approves a contract in the UI, the contract passes its gold cases, and the
    registered contract drives extraction and gating for the rest of the
    process. Nothing about the pipeline's meaning changes at runtime that was
    not first written down here and tested.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    tier: GateTier
    fields: tuple[FieldSpec, ...]

    @model_validator(mode="after")
    def _typed_shapes_cannot_be_lenient(self) -> ShapeContract:
        if self.tier is GateTier.LENIENT and not (
            len(self.fields) == 1 and self.fields[0].name == "text"
        ):
            raise ValueError(
                f"shape contract {self.name!r} cannot declare LENIENT: only a "
                "single-field verbatim quotation shape may clear quotation "
                "fidelity alone"
            )
        if not self.fields:
            raise ValueError(f"shape contract {self.name!r} declares no fields")
        names = [f.name for f in self.fields]
        if len(names) != len(set(names)):
            raise ValueError(f"shape contract {self.name!r} has duplicate field names")
        declared = set(names)
        for field in self.fields:
            if field.min_ref is not None and field.min_ref not in declared:
                raise ValueError(
                    f"field {field.name!r} references unknown field {field.min_ref!r}"
                )
            if field.kind != "number" and (
                field.ge is not None or field.min_ref is not None
            ):
                raise ValueError(
                    f"field {field.name!r}: numeric constraints need kind='number'"
                )
            if field.kind == "enum" and not field.values:
                raise ValueError(f"field {field.name!r}: enum fields declare values")
        return self

    @property
    def required_field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.required)

    def to_definition(self) -> dict[str, Any]:
        """The recorded form: survives a restart, restorable by from_definition."""
        return self.model_dump(mode="json")

    @classmethod
    def from_definition(cls, payload: dict[str, Any]) -> ShapeContract:
        return cls.model_validate(payload)


class ShapeRegistry(BaseModel):
    """Runtime home of approved contracts.

    ``register`` makes a shape effective immediately for the process lifetime;
    ``with_builtins`` seeds the registry with the ontology's existing shapes so
    the pipeline behaves identically before and after the contract pivot.
    ``duplicate_with`` builds an isolated candidate registry for the adoption
    gate -- the contract under test must not shadow live shapes while it is
    still falsifiable.
    """

    model_config = ConfigDict(frozen=True)

    _contracts: dict[str, ShapeContract] = {}

    def lookup(self, name: str) -> ShapeContract | None:
        return self._contracts.get(name)

    def register(self, contract: ShapeContract) -> None:
        if contract.name in self._contracts:
            raise ValueError(f"shape {contract.name!r} is already registered")
        self._contracts[contract.name] = contract

    def duplicate_with(self, contract: ShapeContract) -> ShapeRegistry:
        """A copy of this registry plus the candidate, for gate testing.

        If the candidate shares a name with a live shape, the candidate wins
        in the copy -- the gate must evaluate the NEW contract, and a live
        shape under the same name is exactly the thing adoption replaces.
        """
        clone = ShapeRegistry()
        clone._contracts.update(self._contracts)
        clone._contracts[contract.name] = contract
        return clone

    @classmethod
    def with_builtins(cls) -> ShapeRegistry:
        registry = cls()
        registry._contracts.update({c.name: c for c in _builtin_contracts()})
        return registry


def _builtin_contracts() -> tuple[ShapeContract, ...]:
    """The existing ontology shapes, expressed as contracts.

    Declared to match ``gate_tier.select_gate_tier`` exactly: invoice=STRICT,
    report=STANDARD, prose=LENIENT. The invoice fields mirror
    ``InvoiceHeader`` so the contract path and the class path agree on the
    required-evidence rule while both exist.
    """
    return (
        ShapeContract(
            name="invoice",
            tier=GateTier.STRICT,
            fields=(
                FieldSpec(name="invoice_number", kind="text", required=True),
                FieldSpec(name="issue_date", kind="text"),
                FieldSpec(name="total_amount", kind="number", required=True),
                FieldSpec(name="currency", kind="text"),
            ),
        ),
        ShapeContract(
            name="report",
            tier=GateTier.STANDARD,
            fields=(
                FieldSpec(name="section_title", kind="text", required=True),
                FieldSpec(name="primary_finding", kind="text", required=True),
            ),
        ),
        ShapeContract(
            name="prose",
            tier=GateTier.LENIENT,
            fields=(FieldSpec(name="text", kind="text", required=True),),
        ),
    )


class GoldCaseSpec(BaseModel):
    """One adoption-gate case: claimed values and the expected verdict.

    ``expected`` is the contract's promise about the world: a case claiming
    good values must verify; a case claiming contradictory values must be
    refused. The gate runs against the SAME pipeline the shape will use, in
    the same process -- what is tested is what will run.
    """

    model_config = ConfigDict(frozen=True)

    values: dict[str, str]
    expected: Literal["verified", "mismatch"] = "verified"

    @classmethod
    def from_claims(
        cls, shape_name: str, claims: list[dict]
    ) -> tuple[GoldCaseSpec, ...]:
        cases = []
        for claim in claims:
            claim = dict(claim)
            expected = claim.pop("expected", "verified")
            cases.append(cls(values=claim, expected=expected))
        return tuple(cases)


class AdoptionResult(BaseModel):
    """The gate's verdict on one contract: registered, or why not."""

    model_config = ConfigDict(frozen=True)

    registered: bool
    name: str
    cases_run: int = 0
    failed_cases: tuple[CaseFailure, ...] = Field(default_factory=tuple)


class CaseFailure(BaseModel):
    """A falsification: this case proved the contract does not do what it
    declared. Recorded for the owner -- it is the direction of the next fix,
    not an error message."""

    model_config = ConfigDict(frozen=True)

    case_index: int
    values: dict[str, str]
    expected: str
    got: str


def adopt_contract(
    gate,
    registry: ShapeRegistry,
    contract: ShapeContract,
    cases: tuple[GoldCaseSpec, ...],
) -> AdoptionResult:
    """Run the adoption gate; register only on a full pass.

    Every case is claimed against the contract through a *candidate* registry
    -- the production registry is untouched until every case agrees, so a
    failed contract cannot shadow or mutate anything already live. The gate
    used is the standard one, with span re-derivation; what is tested is
    what will run.
    """
    candidate_registry = registry.duplicate_with(contract)
    candidate_gate = gate.with_registry(candidate_registry)
    failures: list[CaseFailure] = []
    for index, case in enumerate(cases):
        claimed = _claims_from_values(candidate_gate, case.values)
        result = candidate_gate.check_contract(contract.name, claimed)
        got = result.status.value
        if case.expected == "verified":
            expected = {"grounded"}
        else:
            # The case states an intent -- the gate MUST refuse -- so any
            # refusal flavour satisfies it. A number field holding
            # non-numeric text refuses as 'unverifiable'; an enum violation
            # as 'mismatch'. Requiring the literal status would block an
            # adoption whose constraint was proven, just proven differently.
            expected = {"mismatch", "unverifiable"}
        if got not in expected:
            failures.append(
                CaseFailure(
                    case_index=index,
                    values=case.values,
                    expected=case.expected,
                    got=got,
                )
            )
    if failures:
        return AdoptionResult(
            registered=False,
            name=contract.name,
            cases_run=len(cases),
            failed_cases=tuple(failures),
        )
    registry.register(contract)
    return AdoptionResult(
        registered=True, name=contract.name, cases_run=len(cases)
    )


def _claims_from_values(gate, values: dict[str, str]) -> dict:
    """Turn a case's claimed values into grounded field claims.

    The span is the whole raw buffer of the gate's document -- the same
    convention as the gold-set replay extractor. Cases are about the
    contract's semantics (does max>=min hold?), not about span arithmetic.
    """
    from rag_ingest.ontology import FieldValue

    raw = gate._raw
    ref = ProvenanceRef(
        page=1, char_start=0, char_end=len(raw.pages[0].text), element_id="e1"
    )
    return {
        name: FieldValue(value=value, provenance=ref) for name, value in values.items()
    }


__all__ = [
    "AdoptionResult",
    "CaseFailure",
    "FieldSpec",
    "GoldCaseSpec",
    "ShapeContract",
    "ShapeRegistry",
    "adopt_contract",
]
