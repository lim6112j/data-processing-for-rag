"""OpenRouter T2 adapter, exercised against a fake client.

No network, no credentials. The behaviours pinned here are the ones that decide
whether T2 is safe to run unattended:

  * no visual evidence -> abstain, never a text-only guess;
  * transport/malformed failures -> abstain, never an exception;
  * the wire schema has no value field;
  * a pick never verifies anything downstream.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

from rag_ingest.adapters.openrouter import (
    DECISION_SCHEMA,
    OpenRouterAdjudicator,
)
from rag_ingest.adjudication import Candidate, DisagreementKind, build_question
from rag_ingest.candidates import AlternateParseProducer, SecondParse
from rag_ingest.docmodel import Element, ElementKind, ElementScores, ProvenanceRef, RawDocument
from rag_ingest.escalation import ReviewQueue
from rag_ingest.ontology import AbstainReason
from rag_ingest.pipeline import IngestPipeline

TODAY = date(2026, 10, 25)
BBOX = (10.0, 20.0, 300.0, 60.0)


class _FakeCompletions:
    def __init__(self, payload: str | None = None, raises: Exception | None = None) -> None:
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
    def __init__(self, payload: str | None = None, raises: Exception | None = None) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(payload, raises))


def _question(*, bbox=BBOX):
    return build_question(
        chunk_id="c1",
        doc_id="d1",
        kind=DisagreementKind.COLUMN_COUNT,
        prompt="How many columns?",
        candidates=[
            Candidate(candidate_id="a", produced_by="pdfplumber", proposed_text="4"),
            Candidate(candidate_id="b", produced_by="tableformer", proposed_text="5"),
        ],
        provenance=ProvenanceRef(page=1, char_start=0, char_end=5, bbox=bbox),
        context_text="table body",
    )


def _adjudicator(client) -> OpenRouterAdjudicator:
    return OpenRouterAdjudicator(client=client)


class TestVisualEvidenceRequirement:
    def test_no_bbox_abstains_without_calling_the_api(self):
        client = _FakeClient(json.dumps({"kind": "pick", "choice_index": 0}))

        decision = _adjudicator(client).adjudicate_with_evidence(
            _question(bbox=None), image_png=b"PNG"
        )

        assert decision.kind == "abstain"
        assert decision.reason is AbstainReason.AMBIGUOUS
        assert client.chat.completions.calls == []

    def test_no_image_abstains_without_calling_the_api(self):
        client = _FakeClient(json.dumps({"kind": "pick", "choice_index": 0}))

        decision = _adjudicator(client).adjudicate_with_evidence(
            _question(), image_png=None
        )

        assert decision.kind == "abstain"
        assert decision.reason is AbstainReason.AMBIGUOUS
        assert client.chat.completions.calls == []


class TestDecisionParsing:
    def test_valid_pick_is_returned(self):
        client = _FakeClient(json.dumps({"kind": "pick", "choice_index": 1}))

        decision = _adjudicator(client).adjudicate_with_evidence(_question(), image_png=b"PNG")

        assert decision.kind == "pick"
        assert decision.choice_index == 1

    def test_out_of_range_index_becomes_abstention(self):
        client = _FakeClient(json.dumps({"kind": "pick", "choice_index": 7}))

        decision = _adjudicator(client).adjudicate_with_evidence(_question(), image_png=b"PNG")

        assert decision.kind == "abstain"

    def test_pick_without_index_becomes_abstention(self):
        client = _FakeClient(json.dumps({"kind": "pick"}))

        assert (
            _adjudicator(client).adjudicate_with_evidence(_question(), image_png=b"PNG").kind
            == "abstain"
        )

    def test_malformed_json_becomes_abstention(self):
        client = _FakeClient("this is not json")

        decision = _adjudicator(client).adjudicate_with_evidence(_question(), image_png=b"PNG")

        assert decision.kind == "abstain"
        assert "unreadable" in (decision.rationale or "")

    def test_transport_error_becomes_abstention_not_exception(self):
        client = _FakeClient(raises=TimeoutError("slow"))

        decision = _adjudicator(client).adjudicate_with_evidence(_question(), image_png=b"PNG")

        assert decision.kind == "abstain"
        assert "TimeoutError" in (decision.rationale or "")

    def test_abstain_and_escalate_round_trip(self):
        abstain = _FakeClient(json.dumps({"kind": "abstain", "reason": "illegible"}))
        escalate = _FakeClient(
            json.dumps({"kind": "escalate", "owner_question": "Is column 3 a currency?"})
        )

        assert (
            _adjudicator(abstain)
            .adjudicate_with_evidence(_question(), image_png=b"PNG")
            .reason
            is AbstainReason.AMBIGUOUS
        )
        assert (
            _adjudicator(escalate)
            .adjudicate_with_evidence(_question(), image_png=b"PNG")
            .owner_question
            == "Is column 3 a currency?"
        )


class TestWireSchema:
    def test_schema_has_no_value_field(self):
        # The §1 contract, enforced at the wire level rather than by convention.
        assert "value" not in DECISION_SCHEMA["properties"]
        assert DECISION_SCHEMA["additionalProperties"] is False

    def test_schema_enumerates_the_three_outcomes(self):
        assert DECISION_SCHEMA["properties"]["kind"]["enum"] == [
            "pick",
            "abstain",
            "escalate",
        ]

    def test_request_sends_schema_and_image(self):
        client = _FakeClient(json.dumps({"kind": "pick", "choice_index": 0}))

        _adjudicator(client).adjudicate_with_evidence(_question(), image_png=b"PNG")

        call = client.chat.completions.calls[0]
        assert call["response_format"]["json_schema"]["strict"] is True
        assert call["response_format"]["json_schema"]["schema"] is DECISION_SCHEMA
        content = call["messages"][1]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_text_precedes_image(self):
        # OpenRouter recommends this ordering for reliable parsing.
        client = _FakeClient(json.dumps({"kind": "pick", "choice_index": 0}))

        _adjudicator(client).adjudicate_with_evidence(_question(), image_png=b"PNG")

        content = client.chat.completions.calls[0]["messages"][1]["content"]
        assert content[0]["type"] == "text"


class TestPipelineIntegration:
    def test_visual_failure_degrades_to_the_human_queue(self):
        # The whole chain: low-layout element -> STRUCTURE_AMBIGUOUS -> T2 with
        # a failing backend -> abstention -> still queued, never verified.
        element = Element(
            element_id="e1",
            kind=ElementKind.TABLE,
            page=1,
            text="",
            table_rows=(("a", "b"), ("c", "d")),
            scores=ElementScores(layout_score=0.2, parse_score=0.95),
            bbox=BBOX,
        )
        raw, elements = RawDocument.from_elements("d1", [element], page_images={1: b"PNG"})
        adjudicator = _adjudicator(_FakeClient(raises=TimeoutError("slow")))
        # A genuine second parse is required for T2 to have anything to judge.
        alternate = SecondParse(
            name="pdfplumber",
            elements=(
                Element(
                    element_id="alt1",
                    kind=ElementKind.PARAGRAPH,
                    page=1,
                    text="a b c d",
                    bbox=BBOX,
                ),
            ),
        )

        outcome = IngestPipeline(
            ReviewQueue(),
            _NoOpExtractor(),
            today=TODAY,
            adjudicator=adjudicator,
            producer=AlternateParseProducer(primary_name="docling", alternate=alternate),
        ).ingest(raw, elements, owner_id="o")

        assert outcome.verified_chunks == ()
        assert outcome.review_items
        assert len(outcome.adjudications) == 1
        assert outcome.adjudications[0][1].kind == "abstain"


class _NoOpExtractor:
    def extract(self, chunk):
        return None
