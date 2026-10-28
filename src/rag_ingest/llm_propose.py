"""The LLM proposer layer: claims for clusters the grammar cannot read.

The grammar proposer in ``propose.py`` reads cells with three regexes; a
column like ``KS-A 12mm`` or ``PASS/FAIL/PENDING`` is "text" to it, and an
all-text cluster is honestly refused as prose. That refusal is correct for
the grammar but leaves real structure on the table: an inventory table with
Korean specifications is not prose, and pretending it is hides a shape the
owner would adopt in one glance.

This layer lets a model *claim* that structure. The claim arrives in the
same ``Proposal`` form the grammar emits, labelled ``source="llm"`` so the
owner knows who proposed what -- and it is a claim, not a decision: every
field must be backed by a value that verifiably occurs in a cited sample
row, and the proposal only becomes effective after passing the adoption
gate's deterministic gold cases. A model that guesses is caught by the
gate, not served to retrieval.

Failure is refusal, everywhere: a decline, a malformed response, a
transport error, a fabricated value, a column beyond the sample width, an
enum value the samples never show -- all become ``parseable=False`` with a
reason, never a guessed schema.
"""

from __future__ import annotations

import json
import os
from typing import Any

from rag_ingest.propose import FieldEvidence, Proposal
from rag_ingest.shapes import FieldSpec

PROPOSAL_SCHEMA = {
    "name": "shape_proposal",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "propose": {"type": "boolean"},
            "note": {"type": "string"},
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": ["text", "number", "enum"]},
                        "column": {"type": "integer"},
                        "values": {"type": "array", "items": {"type": "string"}},
                        "sample_value": {"type": "string"},
                        "row_index": {"type": "integer"},
                        "required": {"type": "boolean"},
                        "min_ref": {"type": ["string", "null"]},
                    },
                    # strict mode: required must list every property. Optional
                    # meaning is carried by nullable types, not absent keys.
                    "required": [
                        "name", "kind", "column", "values", "sample_value",
                        "row_index", "required", "min_ref",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["propose", "note", "fields"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You propose table schemas for recurring row shapes.

You are given sample rows from one cluster of a document corpus. Propose a
schema only when the evidence supports it; otherwise decline with a reason.
Rules:
- Every field must cite a sample_value that literally appears in the row at
  row_index (0 is the first row given, which may be a header).
- kind must be text, number, or enum. enum only for a genuinely closed set;
  list every value you saw, never a value the samples do not contain.
- Do not invent required constraints you cannot justify from the samples.
- If the rows are prose, or too few to distinguish column roles, decline.
"""


def _build_messages(shape: str, sample_rows: tuple[tuple[str, ...], ...]) -> list[dict]:
    rows_text = "\n".join(
        f"row {i}: {json.dumps(list(row), ensure_ascii=False)}"
        for i, row in enumerate(sample_rows)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Cluster shape: {shape}\nSample rows:\n{rows_text}",
        },
    ]


class LlmProposer:
    """Propose a schema claim for one cluster, via an OpenRouter-style client.

    Same dependency posture as the T2 adjudicator: the client is injected
    (or resolved lazily), failures become abstentions, and nothing here
    decides what enters retrieval -- the adoption gate does.
    """

    def __init__(
        self,
        *,
        model: str = "openai/gpt-4o-mini",
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._client = client
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "LLM proposals need the 'openai' package: uv add openai"
            ) from exc
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            # OpenRouter's key lives in OPENROUTER_API_KEY, which the OpenAI
            # SDK does not read -- resolve it here, like the T2 adjudicator.
            api_key=self._api_key or os.environ.get("OPENROUTER_API_KEY"),
        )
        return self._client

    def propose(
        self,
        shape: str,
        sample_rows: tuple[tuple[str, ...], ...],
        *,
        tier: str = "strict",
    ) -> Proposal:
        def refusal(note: str) -> Proposal:
            return Proposal(
                shape_name=f"{shape}_spec",
                tier_declared=tier,
                fields=(),
                evidence=(),
                source="llm",
                parseable=False,
                note=note,
            )

        try:
            response = self._get_client().chat.completions.create(
                model=self._model,
                messages=_build_messages(shape, sample_rows),
                response_format={"type": "json_schema", "json_schema": PROPOSAL_SCHEMA},
            )
            content = response.choices[0].message.content
            if content is None:
                return refusal("model returned no content")
            payload = json.loads(content)
        except Exception as exc:
            return refusal(f"LLM proposal transport failure: {type(exc).__name__}")

        if not isinstance(payload, dict):
            return refusal("proposal refused: response is not a JSON object")
        try:
            return self._validate(payload, sample_rows, tier, shape)
        except ValueError as exc:
            return refusal(f"proposal refused: {exc}")

    # -- validation --------------------------------------------------------

    def _validate(
        self,
        payload: dict,
        sample_rows: tuple[tuple[str, ...], ...],
        tier: str,
        shape: str,
    ) -> Proposal:
        note = str(payload.get("note", ""))
        if not payload.get("propose"):
            return Proposal(
                shape_name="declined",
                tier_declared=tier,
                fields=(),
                evidence=(),
                source="llm",
                parseable=False,
                note=note or "model declined",
            )
        if not sample_rows:
            raise ValueError("no sample rows to check claims against")

        width = min(len(row) for row in sample_rows)
        fields: list[FieldSpec] = []
        evidence: list[FieldEvidence] = []
        for spec in payload.get("fields", []):
            column = int(spec.get("column", -1))
            row_index = int(spec.get("row_index", -1))
            sample_value = str(spec.get("sample_value", ""))
            if not (0 <= column < width):
                raise ValueError(
                    f"field {spec.get('name')!r}: column {column} is outside the "
                    f"sample width ({width})"
                )
            if not (0 <= row_index < len(sample_rows)):
                raise ValueError(
                    f"field {spec.get('name')!r}: row_index {row_index} does not "
                    f"exist in {len(sample_rows)} sample rows"
                )
            cited_row = sample_rows[row_index]
            if sample_value not in cited_row:
                raise ValueError(
                    f"field {spec.get('name')!r}: claimed sample value "
                    f"{sample_value!r} does not occur in sample row {row_index} "
                    "-- evidence is fabricated"
                )
            kind = spec.get("kind", "text")
            values = tuple(str(v) for v in spec.get("values") or ())
            if kind == "enum":
                for value in values:
                    if not any(value in row for row in sample_rows):
                        raise ValueError(
                            f"field {spec.get('name')!r}: enum value {value!r} "
                            "does not occur in any sample row -- the closed set "
                            "is not grounded"
                        )
                if not values:
                    raise ValueError(
                        f"field {spec.get('name')!r}: enum without values"
                    )
            fields.append(
                FieldSpec(
                    name=str(spec["name"]),
                    kind=kind,
                    values=values if kind == "enum" else (),
                    required=bool(spec.get("required", False)),
                    min_ref=spec.get("min_ref") or None,
                )
            )
            evidence.append(FieldEvidence(sample_value=sample_value, row_index=row_index))
        if not fields:
            raise ValueError("the model proposed an empty schema")
        return Proposal(
            shape_name=f"{shape}_spec",
            tier_declared=tier,
            fields=tuple(fields),
            evidence=tuple(evidence),
            source="llm",
            note=note,
        )


__all__ = ["PROPOSAL_SCHEMA", "LlmProposer"]
