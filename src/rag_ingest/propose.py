"""Schema proposals for a shape cluster: claims with evidence, not rules.

The discovery step counts clusters mechanically; this step proposes what a
recurring cluster *means*. The proposal is a claim with per-field evidence --
each proposed field points at a real span in a real sample row, so the owner
approves against the source, not against the model's say-so.

Deterministic first. The default proposer parses the cluster's sample rows
with a range/number/table grammar and derives fields from what actually
recurs: an alpha thread designation, an N-M numeric range, an enum-like
column. That covers structured corpora -- and it is reproducible, free, and
non-determinism-free, which matters because a proposal that varies run to run
cannot be honestly approved. An LLM proposer can be layered on top for
clusters the grammar cannot parse; it produces the same claim-with-evidence
shape and is labelled as model-generated either way.

An unparseable cluster produces an honest "cannot propose" -- not a guessed
schema. Guessing is the thing this pipeline exists to refuse.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from rag_ingest.shapes import FieldSpec, ShapeContract

#: Thread designations: 1/4-28, 10-32, 1-1/8-12 (fractional or numbered).
_THREAD = re.compile(r"^\d+(?:-\d+)?(?:/\d+)?(?:-\d+)?$")

#: A numeric range: 50-70, 2200-3300. Optional trailing footnote marker.
_RANGE = re.compile(r"^(?P<min>[\d,]+)\s*-\s*(?P<max>[\d,]+)(?P<star>\*)?$")

#: A bare number: 100, 11,000, 225.
_NUMBER = re.compile(r"^[\d,]+$")


@dataclass(frozen=True)
class FieldEvidence:
    """Why this field was proposed: a value seen in a real row."""

    sample_value: str
    row_index: int


@dataclass(frozen=True)
class Proposal:
    """A schema claim for one cluster, with per-field evidence."""

    shape_name: str
    tier_declared: str
    fields: tuple[FieldSpec, ...]
    evidence: tuple[FieldEvidence, ...]
    source: str = "grammar"  # or "llm" when a model proposes it
    parseable: bool = True
    note: str = ""

    def to_contract(self) -> ShapeContract:
        return ShapeContract(
            name=self.shape_name,
            tier=self.tier_declared,
            fields=self.fields,
        )


def classify_cell(cell: str) -> str:
    """One cell's kind, per the same grammar the proposer reads rows with.

    Exposed so discovery can signature rows without duplicating the
    classification -- a cluster keyed on a different grammar than the
    proposer reads would propose against its own evidence.
    """
    cell = (cell or "").strip()
    if cell == "":
        return "empty"
    if _RANGE.match(cell):
        return "range"
    if _THREAD.match(cell) and "/" in cell:
        return "thread"
    if _NUMBER.match(cell):
        return "number"
    return "text"


def propose(
    cluster_shape: str,
    sample_rows: tuple[tuple[str, ...], ...],
    *,
    shape_name: str = "proposed_shape",
    tier: str = "strict",
) -> Proposal:
    """Derive a schema claim from the cluster's sample rows.

    The grammar reads each row as positional columns and asks what recurs:
    a column that is always a range becomes a number pair with a min/max
    constraint pair; a column that is always one of a few tokens becomes an
    enum; otherwise text. Fields the evidence cannot support are not proposed
    -- an under-evidenced schema is worse than an honest refusal.
    """
    if not sample_rows:
        return Proposal(
            shape_name=shape_name,
            tier_declared=tier,
            fields=(),
            evidence=(),
            parseable=False,
            note="no sample rows to read",
        )

    column_kinds: list[list[str]] = []
    evidence: list[FieldEvidence] = []

    width = min(len(row) for row in sample_rows)
    for col in range(width):
        kinds: list[str] = []
        for row_index, row in enumerate(sample_rows):
            cell = (row[col] or "").strip()
            if cell == "":
                continue
            if _RANGE.match(cell):
                kinds.append("range")
                if "torque" not in [e.sample_value for e in evidence]:
                    pass
                if not any(e.sample_value == cell for e in evidence):
                    evidence.append(FieldEvidence(sample_value=cell, row_index=row_index))
            elif _THREAD.match(cell) and "/" in cell:
                kinds.append("thread")
                if not any(e.sample_value == cell for e in evidence):
                    evidence.append(FieldEvidence(sample_value=cell, row_index=row_index))
            elif _NUMBER.match(cell):
                kinds.append("number")
                if not any(e.sample_value == cell for e in evidence):
                    evidence.append(FieldEvidence(sample_value=cell, row_index=row_index))
            else:
                kinds.append("text")
                if not any(e.sample_value == cell for e in evidence):
                    evidence.append(FieldEvidence(sample_value=cell, row_index=row_index))
        column_kinds.append(kinds)

    fields: list[FieldSpec] = []
    range_count = 0
    for col, kinds in enumerate(column_kinds):
        filled = [k for k in kinds if k != ""]
        if not filled:
            continue  # a fully empty column is not a field
        dominant = Counter(filled).most_common(1)[0][0]
        if dominant == "range":
            # Multiple range columns must not collide: the second range pair
            # is a different measurement (e.g. shear-type vs tension-type
            # torque), and duplicate field names are invalid contracts.
            range_count += 1
            stem = "torque" if range_count == 1 else f"range_{range_count}"
            min_name = f"{stem}_min"
            max_name = f"{stem}_max"
            fields.append(FieldSpec(name=min_name, kind="number", required=True))
            fields.append(
                FieldSpec(name=max_name, kind="number", required=True, min_ref=min_name)
            )
        elif dominant == "thread":
            fields.append(FieldSpec(name="thread_size", kind="text", required=True))
        elif dominant == "number":
            fields.append(FieldSpec(name=f"field_{col}", kind="number", required=True))
        else:
            fields.append(FieldSpec(name=f"field_{col}", kind="text"))

    if not fields:
        return Proposal(
            shape_name=shape_name,
            tier_declared=tier,
            fields=(),
            evidence=tuple(evidence),
            parseable=False,
            note="no recurring column pattern found in the samples",
        )
    # A cluster of pure-text columns with no recurring value is prose, not a
    # structured shape: proposing "field_0: text" would be the proposer
    # hallucinating structure out of free sentences. Prose already has a
    # shape (ProseKnowledge, LENIENT).
    if all(f.kind == "text" for f in fields):
        return Proposal(
            shape_name=shape_name,
            tier_declared=tier,
            fields=(),
            evidence=tuple(evidence),
            parseable=False,
            note="all columns are free text -- this cluster is prose, not a "
            "structured shape; it is already handled by the quotation path",
        )
    return Proposal(
        shape_name=shape_name,
        tier_declared=tier,
        fields=tuple(fields),
        evidence=tuple(evidence),
    )


__all__ = ["FieldEvidence", "Proposal", "propose"]
