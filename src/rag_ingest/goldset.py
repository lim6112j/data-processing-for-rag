"""Gold-set loader and evaluator.

Plan §7-2: without a gold set every threshold in the pipeline is vibes. This
module turns the YAML cases into concrete documents, runs them through the
pipeline, and reports pass/fail per expected decision.

The evaluator deliberately asserts on *decisions* (verified / quarantined /
why), not on exact confidence numbers -- the numbers are implementation detail,
the decisions are the contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.docmodel import Element, ElementKind, ElementScores, ProvenanceRef, RawDocument
from rag_ingest.ontology import (
    AbstainReason,
    FieldAbstention,
    FieldValue,
    InvoiceHeader,
    Unclassified,
)


class Expected(BaseModel):
    model_config = ConfigDict(frozen=True)

    verified: int | None = None
    quarantined: int | None = None
    reason: str | None = None
    not_retrievable: tuple[str, ...] = Field(default_factory=tuple)
    table_chunks: int | None = None


class GoldCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    description: str
    class_name: str = Field(alias="class")
    scores: dict[str, float] = Field(default_factory=dict)
    pages: tuple[str, ...] = Field(default_factory=tuple)
    extractor_claim: dict[str, Any] | str | None = None
    table_rows: int | None = None
    sections: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    expected: Expected


class GoldSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    cases: tuple[GoldCase, ...]


def load_goldset(path: Path) -> GoldSet:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "cases" not in payload:
        raise ValueError(f"{path} is not a gold-set file: missing 'cases'")
    return GoldSet.model_validate(payload)


def _element_scores(raw: dict[str, float] | None) -> ElementScores:
    if not raw:
        return ElementScores()
    return ElementScores(
        layout_score=raw.get("layout"),
        ocr_score=raw.get("ocr"),
        parse_score=raw.get("parse"),
    )


def build_document(case: GoldCase) -> tuple[RawDocument, list[Element]]:
    """Construct the raw document and elements a gold case describes."""
    elements: list[Element] = []

    if case.sections:
        for index, section in enumerate(case.sections):
            elements.append(
                Element(
                    element_id=f"s{index}",
                    kind=ElementKind.PARAGRAPH,
                    page=1,
                    text=str(section["text"]),
                    scores=_element_scores(section.get("scores")),
                )
            )
    else:
        body = "\n".join(case.pages)
        elements.append(
            Element(
                element_id="body",
                kind=ElementKind.PARAGRAPH,
                page=1,
                text=body,
                scores=_element_scores(case.scores),
            )
        )

    if case.table_rows:
        header = ("item", "amount")
        body_rows = tuple((f"row{i}", f"{i * 10}") for i in range(case.table_rows))
        rows = (header, *body_rows)
        elements.append(
            Element(
                element_id="table",
                kind=ElementKind.TABLE,
                page=1,
                text="",
                table_rows=rows,
                scores=_element_scores(case.scores),
            )
        )

    return RawDocument.from_elements(case.id, elements)


class GoldExtractor:
    """Replays the claim recorded in the gold case.

    ``extractor_claim`` of ``"unclassified"`` yields an Unclassified result; a
    dict of field -> claimed value yields an InvoiceHeader with those values
    grounded against the whole page. Everything unclaimed is an honest
    abstention, which is the correct behaviour for a field that is absent.
    """

    def __init__(self, case: GoldCase, raw: RawDocument) -> None:
        self._case = case
        self._raw = raw

    def extract(self, chunk):
        claim = self._case.extractor_claim
        page = self._raw.page(chunk.page_start)
        if page is None:
            return None
        ref = ProvenanceRef(page=chunk.page_start, char_start=0, char_end=page.char_len)

        if claim == "unclassified":
            return Unclassified(observed_shape="unknown", provenance=ref)
        if not isinstance(claim, dict):
            # No claim recorded: this case is about parsing, not extraction.
            return None

        def field(name: str):
            if name in claim:
                return FieldValue(value=str(claim[name]), provenance=ref)
            return FieldAbstention(reason=AbstainReason.NOT_PRESENT)

        return InvoiceHeader(
            invoice_number=field("invoice_number"),
            issue_date=field("issue_date"),
            total_amount=field("total_amount"),
            currency=field("currency"),
        )


class CaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    passed: bool
    failures: tuple[str, ...] = Field(default_factory=tuple)


def check_case(case: GoldCase, outcome, verified_hits: list[str]) -> CaseResult:
    failures: list[str] = []
    expected = case.expected

    if expected.verified is not None and len(outcome.verified_chunks) != expected.verified:
        failures.append(
            f"expected {expected.verified} verified, got {len(outcome.verified_chunks)}"
        )

    if (
        expected.quarantined is not None
        and len(outcome.quarantined_chunks) != expected.quarantined
    ):
        failures.append(
            f"expected {expected.quarantined} quarantined, got {len(outcome.quarantined_chunks)}"
        )

    if expected.reason is not None:
        reasons = {item.reason.value for item in outcome.review_items}
        if expected.reason not in reasons:
            failures.append(
                f"expected escalation reason {expected.reason!r}, got {sorted(reasons)}"
            )

    for term in expected.not_retrievable:
        if any(term in hit for hit in verified_hits):
            failures.append(f"forbidden term {term!r} appeared in verified retrieval")

    if expected.table_chunks is not None:
        all_chunks = outcome.verified_chunks + outcome.quarantined_chunks
        table_chunks = [c for c in all_chunks if c.chunk_id.startswith(f"{case.id}:table:")]
        if len(table_chunks) != expected.table_chunks:
            failures.append(
                f"expected {expected.table_chunks} table chunk(s), got {len(table_chunks)}"
            )

    return CaseResult(case_id=case.id, passed=not failures, failures=tuple(failures))


__all__ = [
    "CaseResult",
    "GoldCase",
    "GoldExtractor",
    "GoldSet",
    "build_document",
    "check_case",
    "load_goldset",
]
