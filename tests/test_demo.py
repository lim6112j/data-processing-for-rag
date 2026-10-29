"""Demo layer: the HTTP contract and the state behind it.

The UI itself is verified by hand in a browser; these tests pin the parts that
break silently -- routing, id handling, and the review loop's effect on the
scorecard.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient
from helpers.pdf_fixture import make_pdf

from rag_ingest.demo.session import DemoSession
from rag_ingest.docmodel import Element, ElementKind, RawDocument
from rag_ingest.extractors import TemplateExtractor
from rag_ingest.grounding import GroundingStatus

TODAY = date(2026, 10, 25)


def _quarantine_session():
    """A session with NO prose fallback: single-line text chunks stay
    unclassified and quarantine, which is what the review-flow tests need
    a steady supply of. The production default layers ProseExtractor in --
    prose is quotable -- so tests that specifically exercise the review
    loop ask for this narrower extractor explicitly."""
    return DemoSession(today=TODAY, extractor=TemplateExtractor())


def _source(text: str, doc_id: str = "d1"):
    raw, located = RawDocument.from_elements(
        doc_id,
        [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1, text=text)],
        page_sizes={1: (612.0, 792.0)},
    )
    return raw, located


class TestSessionIngest:
    def test_verified_document_reports_full_rate(self):
        raw, elements = _source("Invoice INV-2041 total 1234.56 USD")
        session = DemoSession(today=TODAY)

        view = session.ingest(
            doc_id="d1", filename="a.pdf", parser="test", raw=raw, elements=elements
        )

        assert view.verified_count == 1
        assert view.quarantined_count == 0

    def test_incomplete_extraction_is_withheld(self):
        raw, elements = _source("Invoice INV-3003 total l,234.56 USD")
        session = DemoSession(today=TODAY)

        view = session.ingest(
            doc_id="d1", filename="a.pdf", parser="test", raw=raw, elements=elements
        )

        assert view.verified_count == 0
        assert view.chunks[0].reason == "incomplete_extraction"

    def test_review_items_are_enqueued_not_just_listed(self):
        # A view that lists items the queue does not hold would look alive and
        # accept nothing -- this was a real bug.
        raw, elements = _source("A haiku about procurement")
        session = _quarantine_session()

        view = session.ingest(
            doc_id="d1", filename="a.pdf", parser="test", raw=raw, elements=elements
        )

        assert view.review_items
        assert len(session.open_items()) == len(view.review_items)

    def test_page_size_is_carried_for_the_overlay(self):
        raw, elements = _source("Invoice INV-1 total 10.00 USD")
        session = DemoSession(today=TODAY)

        view = session.ingest(
            doc_id="d1", filename="a.pdf", parser="test", raw=raw, elements=elements
        )

        assert view.pages[0].width == 612.0
        assert view.pages[0].height == 792.0


class TestReviewLoop:
    def _ingested(self):
        raw, elements = _source("A haiku about procurement")
        session = _quarantine_session()
        session.ingest(
            doc_id="d1", filename="a.pdf", parser="test", raw=raw, elements=elements
        )
        return session

    def test_resolving_removes_the_item_from_the_queue(self):
        session = self._ingested()
        item = session.open_items()[0]

        session.resolve(item.item_id, after="prose")

        assert session.open_items() == []

    def test_resolution_creates_a_label(self):
        session = self._ingested()
        item = session.open_items()[0]

        session.resolve(item.item_id, after="prose")

        assert len(session.queue.corrections()) == 1

    def test_unknown_item_raises(self):
        session = self._ingested()

        with pytest.raises(KeyError):
            session.resolve("nope", after="x")

    def test_verifier_correction_is_recorded_but_not_authoritative(self):
        from rag_ingest.escalation import ReviewerRole

        session = self._ingested()
        item = session.open_items()[0]

        session.resolve(item.item_id, after="guess", reviewer_role=ReviewerRole.VERIFIER)

        assert session.queue.labels() == ()


class TestOverlay:
    def test_overlay_returns_boxes_for_chunks_and_elements(self):
        raw, elements = _source("Invoice INV-2041 total 1234.56 USD")
        elements[0] = elements[0].model_copy(update={"bbox": (10.0, 20.0, 300.0, 40.0)})
        session = DemoSession(today=TODAY)
        session.ingest(
            doc_id="d1", filename="a.pdf", parser="test", raw=raw, elements=elements
        )

        overlay = session.overlay("d1")

        assert overlay["chunks"]
        assert overlay["chunks"][0]["bbox"] == [10.0, 20.0, 300.0, 40.0]
        assert overlay["elements"]

    def test_unknown_document_yields_empty_overlay(self):
        assert DemoSession().overlay("nope") == {"chunks": [], "elements": []}


class TestHttpContract:
    @pytest.fixture
    def client(self):
        from rag_ingest.demo import server

        # Fresh session per test: the module-level one is shared state.
        server._session = _quarantine_session()
        return TestClient(server.app)

    def test_index_serves_html(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "rag-ingest" in response.text

    def test_ingest_rejects_an_empty_upload(self, client):
        response = client.post(
            "/api/ingest", files={"file": ("x.pdf", b"", "application/pdf")}
        )

        assert response.status_code == 400

    def test_ingest_rejects_a_non_pdf(self, client):
        response = client.post(
            "/api/ingest", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}
        )

        assert response.status_code == 422

    def test_ingest_accepts_a_real_pdf(self, client, tmp_path):
        path = tmp_path / "a.pdf"
        make_pdf(path)

        with path.open("rb") as handle:
            response = client.post(
                "/api/ingest?parser=pdfplumber",
                files={"file": ("a.pdf", handle, "application/pdf")},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["pages"]
        assert body["chunks"]

    def test_page_image_is_png(self, client, tmp_path):
        path = tmp_path / "a.pdf"
        make_pdf(path)
        with path.open("rb") as handle:
            client.post(
                "/api/ingest?parser=pdfplumber",
                files={"file": ("a.pdf", handle, "application/pdf")},
            )

        response = client.get("/api/page/a/1")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_missing_page_image_is_404(self, client):
        assert client.get("/api/page/nope/1").status_code == 404

    def test_review_id_travels_in_the_body(self, client):
        # Chunk ids contain ':' and '/', which a path parameter cannot carry.
        # This was a real failure: every Submit returned 404.
        from rag_ingest.demo import server
        from rag_ingest.escalation import EscalationReason, ReviewItem

        item_id = "doc:table:#/tables/0"
        server._session.queue.enqueue(
            ReviewItem(
                item_id=item_id,
                doc_id="doc",
                chunk_id="c1",
                reason=EscalationReason.UNCLASSIFIED,
                question="q",
                owner_id="o",
                created_on=TODAY,
            )
        )

        response = client.post(
            "/api/review",
            json={"item_id": item_id, "after": "prose", "role": "owner"},
        )

        assert response.status_code == 200
        assert server._session.open_items() == []

    def test_unknown_item_is_404(self, client):
        response = client.post(
            "/api/review", json={"item_id": "nope", "after": "x", "role": "owner"}
        )

        assert response.status_code == 404

    def test_bad_role_is_400(self, client):
        response = client.post(
            "/api/review",
            json={"item_id": "x", "after": "y", "role": "wizard"},
        )

        assert response.status_code == 400

    def test_config_toggles_t2(self, client):
        assert client.post("/api/config?t2=true").json()["t2_enabled"] is True
        assert client.get("/api/config").json()["t2_enabled"] is True

    def test_scorecard_is_available_before_any_ingest(self, client):
        body = client.get("/api/scorecard").json()

        assert body["verdict"] == "no labelled cases yet"

    def test_view_reflects_resolution(self, client, tmp_path):
        path = tmp_path / "a.pdf"
        make_pdf(path)
        with path.open("rb") as handle:
            view = client.post(
                "/api/ingest?parser=pdfplumber",
                files={"file": ("a.pdf", handle, "application/pdf")},
            ).json()

        item = view["review_items"][0]
        client.post(
            "/api/review",
            json={"item_id": item["item_id"], "after": "prose", "role": "owner"},
        )

        refreshed = client.get(f"/api/view/{view['doc_id']}").json()
        resolved = {i["chunk_id"]: i["resolved"] for i in refreshed["review_items"]}
        assert resolved[item["chunk_id"]] is True
        # Other items stay open -- resolving one question must not close the rest.
        assert sum(resolved.values()) == 1


class TestT2DemoSwitch:
    def test_adjudicator_without_a_producer_never_fires(self):
        # T2 needs a disagreement to adjudicate. An adjudicator alone is inert,
        # which is why the demo configures both together.
        from rag_ingest.demo.fake_adjudicator import ScriptedAdjudicator

        raw, elements = _source("Item | Amount")
        session = DemoSession(
            extractor=TemplateExtractor(),
            adjudicator=ScriptedAdjudicator(),
            force_structure_ambiguous=True,
            today=TODAY,
        )

        view = session.ingest(
            doc_id="d1", filename="a.pdf", parser="test", raw=raw, elements=elements
        )

        assert view.review_items[0].candidate_ids == ()
        assert view.review_items[0].adjudicated_choice is None

    def test_force_ambiguous_reaches_the_adjudicator(self):
        from rag_ingest.demo.fake_adjudicator import ScriptedAdjudicator
        from rag_ingest.demo.producers import DoubleReadingProducer

        raw, elements = _source("Item | Amount\nWidget | 1200")
        session = DemoSession(
            extractor=TemplateExtractor(),
            adjudicator=ScriptedAdjudicator(prefer="pdfplumber"),
            producer=DoubleReadingProducer(),
            force_structure_ambiguous=True,
            today=TODAY,
        )

        view = session.ingest(
            doc_id="d1", filename="a.pdf", parser="test", raw=raw, elements=elements
        )

        tiers = {t.tier for t in view.tiers}
        assert "structure_unknown" in tiers
        assert view.review_items
        assert view.review_items[0].adjudicated_choice is not None

    def test_without_the_switch_a_readable_document_never_reaches_t2(self):
        # Honest behaviour, and the reason the demo needs the switch.
        from rag_ingest.demo.fake_adjudicator import ScriptedAdjudicator

        raw, elements = _source("Invoice INV-2041 total 1234.56 USD")
        session = DemoSession(adjudicator=ScriptedAdjudicator(), today=TODAY)

        view = session.ingest(
            doc_id="d1", filename="a.pdf", parser="test", raw=raw, elements=elements
        )

        assert view.review_items == ()

    def test_scripted_abstention_is_reachable(self):
        from rag_ingest.demo.fake_adjudicator import ScriptedAdjudicator
        from rag_ingest.demo.producers import DoubleReadingProducer

        raw, elements = _source("Item | Amount")
        session = DemoSession(
            extractor=TemplateExtractor(),
            adjudicator=ScriptedAdjudicator(always_abstain=True),
            producer=DoubleReadingProducer(),
            force_structure_ambiguous=True,
            today=TODAY,
        )

        view = session.ingest(
            doc_id="d1", filename="a.pdf", parser="test", raw=raw, elements=elements
        )

        assert view.review_items[0].adjudicated_choice is None


def test_incomplete_status_exists():
    assert GroundingStatus.INCOMPLETE.value == "incomplete"


class TestReingest:
    """Re-uploading the same file must not 500.

    Item ids derive from doc_id + chunk id, so a second parse produced identical
    ids and tripped the queue's duplicate guard. That was a real server error.
    """

    def _ingest(self, session, doc_id="damaged"):
        raw, elements = _source("Invoice INV-3003 total l,234.56 USD", doc_id=doc_id)
        return session.ingest(
            doc_id=doc_id, filename="d.pdf", parser="test", raw=raw, elements=elements
        )

    def test_second_ingest_does_not_raise(self):
        session = DemoSession(today=TODAY)

        self._ingest(session)
        self._ingest(session)  # must not raise

        assert len(session.open_items()) == 1

    def test_reingest_replaces_stale_questions(self):
        session = DemoSession(today=TODAY)

        first = self._ingest(session)
        second = self._ingest(session)

        assert first.review_items[0].item_id == second.review_items[0].item_id
        assert len(session.open_items()) == 1

    def test_resolved_label_survives_reingest(self):
        session = DemoSession(today=TODAY)
        self._ingest(session)
        session.resolve(session.open_items()[0].item_id, after="1234.56")

        self._ingest(session)

        assert len(session.queue.labels()) == 1

    def test_collected_labels_stay_scorable_after_the_item_is_gone(self):
        # Items are discarded on re-ingest; a label that needs the item to be
        # scored is not a label.
        from rag_ingest.demo.fake_adjudicator import ScriptedAdjudicator
        from rag_ingest.demo.producers import DoubleReadingProducer

        session = DemoSession(
            extractor=TemplateExtractor(),
            adjudicator=ScriptedAdjudicator(prefer="pdfplumber"),
            producer=DoubleReadingProducer(),
            force_structure_ambiguous=True,
            today=TODAY,
        )
        self._ingest(session)
        item = session.open_items()[0]
        session.resolve(
            item.item_id, after=item.adjudicated_choice, choice=item.adjudicated_choice
        )
        before = session.scorecard()

        self._ingest(session)
        after = session.scorecard()

        assert before.total_scored == 1
        assert after.total_scored == 1
        assert after.precision == before.precision


class TestStructureAndIncompleteCoexist:
    """The two escalation reasons are not mutually exclusive.

    A chunk with unreliable layout very often also has an incomplete extraction.
    An earlier ordering let INCOMPLETE win outright, which made the T2 branch
    unreachable -- a regression that no existing test caught.
    """

    def _session(self):
        from rag_ingest.demo.fake_adjudicator import ScriptedAdjudicator
        from rag_ingest.demo.producers import DoubleReadingProducer

        return DemoSession(
            adjudicator=ScriptedAdjudicator(prefer="pdfplumber"),
            producer=DoubleReadingProducer(),
            force_structure_ambiguous=True,
            today=TODAY,
        )

    def test_t2_is_reachable_when_extraction_is_also_incomplete(self):
        session = self._session()
        raw, elements = _source("Invoice INV-3003 total l,234.56 USD")

        view = session.ingest(
            doc_id="d1", filename="d.pdf", parser="test", raw=raw, elements=elements
        )

        item = view.review_items[0]
        assert item.reason == "structure_ambiguous"
        assert item.candidate_ids
        assert item.adjudicated_choice is not None

    def test_incomplete_detail_is_kept_in_the_structural_question(self):
        session = self._session()
        raw, elements = _source("Invoice INV-3003 total l,234.56 USD")

        view = session.ingest(
            doc_id="d1", filename="d.pdf", parser="test", raw=raw, elements=elements
        )

        question = view.review_items[0].question
        assert "Two parsers read it differently" in question
        # The structural question must keep the grounding detail: it names the
        # missing fields so the reviewer can answer both questions at once.
        assert "did not find issue date, total amount and currency" in question

    def test_grounding_mismatch_still_never_reaches_t2(self):
        from rag_ingest.adjudication import AdjudicationDecision
        from rag_ingest.docmodel import Element, ElementKind, ProvenanceRef, RawDocument
        from rag_ingest.ontology import AbstainReason, FieldAbstention, FieldValue, InvoiceHeader

        class _Recorder:
            def __init__(self):
                self.calls = 0

            def adjudicate_with_evidence(self, question, *, image_png):
                self.calls += 1
                return AdjudicationDecision(kind="pick", choice_index=0)

        class _Liar:
            def extract(self, chunk):
                ref = ProvenanceRef(page=1, char_start=0, char_end=len(chunk.text))
                return InvoiceHeader(
                    invoice_number=FieldValue(value="INV-1", provenance=ref),
                    issue_date=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
                    total_amount=FieldValue(value="9999.99", provenance=ref),
                    currency=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
                )

        from rag_ingest.demo.producers import DoubleReadingProducer

        recorder = _Recorder()
        raw, elements = RawDocument.from_elements(
            "d1",
            [
                Element(
                    element_id="e1",
                    kind=ElementKind.PARAGRAPH,
                    page=1,
                    text="Invoice INV-1 total 10.00",
                    bbox=(0.0, 0.0, 100.0, 20.0),
                )
            ],
        )
        session = DemoSession(
            extractor=_Liar(),
            adjudicator=recorder,
            producer=DoubleReadingProducer(),
            force_structure_ambiguous=True,
            today=TODAY,
        )

        view = session.ingest(
            doc_id="d1", filename="d.pdf", parser="test", raw=raw, elements=elements
        )

        assert recorder.calls == 0
        assert view.review_items[0].reason == "ungrounded"


class TestLabelsAreVisible:
    """Saying "recorded as labels" is a claim; the demo must let you verify it."""

    @pytest.fixture
    def client(self):
        from rag_ingest.demo import server

        server._session = _quarantine_session()
        return TestClient(server.app)

    def _submit(self, client, answer="1234.56"):
        from rag_ingest.demo import server

        raw, elements = _source("Invoice INV-3003 total l,234.56 USD")
        server._session.ingest(
            doc_id="d1", filename="d.pdf", parser="test", raw=raw, elements=elements
        )
        item = server._session.open_items()[0]
        return client.post(
            "/api/review",
            json={"item_id": item.item_id, "after": answer, "role": "owner"},
        )

    def test_labels_endpoint_is_empty_before_any_review(self, client):
        assert client.get("/api/labels").json()["corrections"] == []

    def test_submitted_correction_appears_in_the_label_list(self, client):
        self._submit(client, answer="1234.56 dated 2026-10-27")

        corrections = client.get("/api/labels").json()["corrections"]

        assert len(corrections) == 1
        assert corrections[0]["after"] == "1234.56 dated 2026-10-27"

    def test_label_records_who_and_when(self, client):
        self._submit(client)

        correction = client.get("/api/labels").json()["corrections"][0]

        assert correction["reviewer_role"] == "owner"
        assert correction["captured_on"] == TODAY.isoformat()
        assert correction["validator_error"]

    def test_labels_download_is_the_same_data(self, client):
        import json

        self._submit(client)

        response = client.get("/api/labels.json")

        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        payload = json.loads(response.content)
        assert len(payload["corrections"]) == 1
        assert "review_items" in payload

    def test_downloaded_labels_feed_the_scorer(self, client, tmp_path):
        # The loop the download exists for: save it, score it.
        import json

        from rag_ingest.cli import main

        self._submit(client)
        path = tmp_path / "labels.json"
        path.write_bytes(client.get("/api/labels.json").content)

        exit_code = main(["--score", str(path)])

        assert exit_code == 0
        assert len(json.loads(path.read_text())["corrections"]) == 1


class TestShapeDiscoveryFlow:
    """The owner sees quarantined chunks as shape candidates, approves one,
    and the pipeline verifies the next chunk against it -- no restart."""

    def _session(self):

        return _quarantine_session()

    def _ingest_tables(self, session, doc_id="tables"):
        # Three documents, each one oversized tabular chunk on its own page --
        # exactly the recurring-shape situation discovery exists for: same
        # shape, multiple chunks, multiple pages, no ontology entry for it.
        for i in range(3):
            filler = "Torque limits for standard hardware. " * 60
            text = f"TABLE 7-{i} torque values. | 1/4-{i} | 50-70 | Coarse | {filler}"
            raw, elements = RawDocument.from_elements(
                f"{doc_id}{i}",
                [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1, text=text)],
            )
            session.ingest(
                doc_id=f"{doc_id}{i}",
                filename=f"t{i}.pdf",
                parser="test",
                raw=raw,
                elements=elements,
            )

    def test_discovery_reports_tabular_cluster_after_ingest(self):
        session = self._session()
        self._ingest_tables(session)

        report = session.discovery_report()

        assert report is not None
        assert report.unclassified_total > 0

    def test_no_documents_means_no_report(self):
        session = self._session()

        assert session.discovery_report() is None

    def test_empty_quarantine_reports_nothing(self):
        from rag_ingest.demo.session import DemoSession

        session = DemoSession(today=TODAY)
        raw, elements = RawDocument.from_elements(
            "clean",
            [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1,
                     text="INVOICE\nInvoice INV-2041\nTotal 1,234.56 KRW")],
        )
        session.ingest(doc_id="clean", filename="c.pdf", parser="test",
                       raw=raw, elements=elements)

        assert session.discovery_report() is None

    def test_proposal_from_cluster_is_deterministic_and_evidenced(self):
        session = self._session()
        self._ingest_tables(session)

        report = session.discovery_report()
        candidate = report.candidates[0]

        proposal = session.propose_for(candidate.shape)

        assert proposal is not None
        # Deterministic: same inputs, same proposal (run twice).
        again = session.propose_for(candidate.shape)
        assert again.fields == proposal.fields

    def test_adopt_registers_the_contract_without_a_restart(self):
        from rag_ingest.gate_tier import GateTier
        from rag_ingest.shapes import FieldSpec, ShapeContract

        session = self._session()
        self._ingest_tables(session)

        contract = ShapeContract(
            name="torque_spec",
            tier=GateTier.STRICT,
            fields=(
                FieldSpec(name="thread_size", kind="text", required=True),
                FieldSpec(name="torque_min", kind="number", required=True),
                FieldSpec(name="torque_max", kind="number", required=True,
                          min_ref="torque_min"),
            ),
        )

        session.adopt(contract)

        assert session.registry().lookup("torque_spec") is contract

    def test_adopted_contract_verifies_the_next_chunk(self):
        from rag_ingest.gate_tier import GateTier
        from rag_ingest.shapes import FieldSpec, ShapeContract

        session = self._session()
        self._ingest_tables(session)
        contract = ShapeContract(
            name="torque_spec",
            tier=GateTier.STRICT,
            fields=(
                FieldSpec(name="thread_size", kind="text", required=True),
                FieldSpec(name="torque_min", kind="number", required=True),
                FieldSpec(name="torque_max", kind="number", required=True,
                          min_ref="torque_min"),
            ),
        )
        session.adopt(contract)

        # A new chunk arrives after adoption; its values claim the new shape.
        result = session.check_against_shape(
            "torque_spec",
            "TABLE 7-2: 7/16-20 torque 450-500",
            {
                "thread_size": "7/16-20",
                "torque_min": "450",
                "torque_max": "500",
            },
        )

        assert result.verified is True

    def test_adopted_contract_refuses_an_inverted_range(self):
        from rag_ingest.gate_tier import GateTier
        from rag_ingest.shapes import FieldSpec, ShapeContract

        session = self._session()
        self._ingest_tables(session)
        contract = ShapeContract(
            name="torque_spec",
            tier=GateTier.STRICT,
            fields=(
                FieldSpec(name="thread_size", kind="text", required=True),
                FieldSpec(name="torque_min", kind="number", required=True),
                FieldSpec(name="torque_max", kind="number", required=True,
                          min_ref="torque_min"),
            ),
        )
        session.adopt(contract)

        result = session.check_against_shape(
            "torque_spec",
            "TABLE 7-2: 7/16-20 torque 500-450",  # inverted, verbatim
            {
                "thread_size": "7/16-20",
                "torque_min": "500",
                "torque_max": "450",
            },
        )

        assert result.verified is False


class TestShapeDiscoveryAPI:
    """The UI surface: discovery visible, proposal evidenced, adoption live."""

    def _client_with_tables(self):
        from fastapi.testclient import TestClient

        from rag_ingest.demo.server import _session, app
        from rag_ingest.escalation import ReviewQueue

        _session.__dict__.update(
            {
                "outcomes": {},
                "_shapes": None,
                "queue": ReviewQueue(),
                "views": {},
                "documents": {},
                "chunk_index": {},
            }
        )
        session = _session
        filler = "Torque limits for standard hardware. " * 60
        for i in range(3):
            text = f"TABLE 7-{i} torque values. | 1/4-{i} | 50-70 | Coarse | {filler}"
            raw, elements = RawDocument.from_elements(
                f"t{i}",
                [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1, text=text)],
            )
            session.ingest(doc_id=f"t{i}", filename=f"t{i}.pdf", parser="test",
                           raw=raw, elements=elements)
        return TestClient(app), session

    def test_discovery_endpoint_lists_clusters(self):
        client, _ = self._client_with_tables()

        response = client.get("/api/discovery")

        assert response.status_code == 200
        payload = response.json()
        assert payload["unclassified_total"] == 3
        assert payload["clusters"][0]["shape"] == "tabular"
        assert payload["clusters"][0]["is_candidate"] is True

    def test_discovery_endpoint_is_empty_without_quarantine(self):
        from fastapi.testclient import TestClient

        from rag_ingest.demo.server import _session, app

        client = TestClient(app)
        _session.__dict__.update({"outcomes": {}, "_shapes": None})

        response = client.get("/api/discovery")

        assert response.status_code == 200
        assert response.json() is None

    def test_propose_endpoint_returns_schema_with_evidence(self):
        client, _ = self._client_with_tables()

        response = client.post("/api/propose", json={"shape": "tabular"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["parseable"] is True
        assert any(f["name"] == "torque_min" for f in payload["fields"])

    def test_propose_endpoint_refuses_unknown_shape(self):
        client, _ = self._client_with_tables()

        response = client.post("/api/propose", json={"shape": "nonexistent"})

        assert response.status_code == 404

    def test_adopt_endpoint_registers_and_reports_gold_status(self):
        client, session = self._client_with_tables()

        contract = {
            "name": "torque_spec",
            "tier": "strict",
            "fields": [
                {"name": "thread_size", "kind": "text", "required": True},
                {"name": "torque_min", "kind": "number", "required": True},
                {"name": "torque_max", "kind": "number", "required": True,
                 "min_ref": "torque_min"},
            ],
        }
        cases = [
            {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70"},
            {"thread_size": "1/4-28", "torque_min": "70", "torque_max": "50",
             "expected": "mismatch"},
        ]
        response = client.post("/api/adopt", json={"contract": contract, "cases": cases})

        assert response.status_code == 200
        assert response.json()["registered"] is True
        assert session.registry().lookup("torque_spec") is not None

    def test_adopt_endpoint_blocks_a_falsified_contract_with_the_failures(self):
        client, session = self._client_with_tables()

        contract = {
            "name": "loose",
            "tier": "strict",
            "fields": [
                {"name": "lo", "kind": "number", "required": True},
                {"name": "hi", "kind": "number", "required": True},
            ],
        }
        cases = [
            # The owner expects the gate to refuse an inverted range, but the
            # contract declares no min_ref constraint -- the gate passes it,
            # which falsifies the contract. Blocked.
            {"lo": "70", "hi": "50", "expected": "mismatch"},
        ]
        response = client.post("/api/adopt", json={"contract": contract, "cases": cases})

        assert response.status_code == 409
        body = response.json()
        assert body["detail"]["failed_cases"][0]["expected"] == "mismatch"
        assert session.registry().lookup("loose") is None

    def test_adopt_rejects_an_invalid_contract(self):
        client, _ = self._client_with_tables()

        contract = {
            "name": "bad",
            "tier": "strict",
            "fields": [
                {"name": "x", "kind": "number", "required": True,
                 "min_ref": "ghost_field"},
            ],
        }
        cases = [
            {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70"},
        ]
        response = client.post("/api/adopt", json={"contract": contract, "cases": cases})

        assert response.status_code == 422

    def test_adopt_without_cases_is_refused(self):
        client, _ = self._client_with_tables()

        contract = {
            "name": "any",
            "tier": "strict",
            "fields": [{"name": "x", "kind": "text", "required": True}],
        }
        response = client.post("/api/adopt", json={"contract": contract})

        assert response.status_code == 422

    def test_verify_endpoint_checks_values_against_adopted_shape(self):
        client, _ = self._client_with_tables()
        contract = {
            "name": "torque_spec",
            "tier": "strict",
            "fields": [
                {"name": "thread_size", "kind": "text", "required": True},
                {"name": "torque_min", "kind": "number", "required": True},
                {"name": "torque_max", "kind": "number", "required": True,
                 "min_ref": "torque_min"},
            ],
        }
        client.post("/api/adopt", json={"contract": contract, "cases": [
            {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70"},
            {"thread_size": "1/4-28", "torque_min": "70", "torque_max": "50",
             "expected": "mismatch"},
        ]})

        response = client.post(
            "/api/verify-shape",
            json={
                "shape": "torque_spec",
                "text": "TABLE 7-2: 7/16-20 torque 450-500",
                "values": {"thread_size": "7/16-20", "torque_min": "450",
                           "torque_max": "500"},
            },
        )

        assert response.status_code == 200
        assert response.json()["verified"] is True

    def test_verify_endpoint_reports_inverted_range_as_not_verified(self):
        client, _ = self._client_with_tables()
        contract = {
            "name": "torque_spec",
            "tier": "strict",
            "fields": [
                {"name": "thread_size", "kind": "text", "required": True},
                {"name": "torque_min", "kind": "number", "required": True},
                {"name": "torque_max", "kind": "number", "required": True,
                 "min_ref": "torque_min"},
            ],
        }
        client.post("/api/adopt", json={"contract": contract, "cases": [
            {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70"},
            {"thread_size": "1/4-28", "torque_min": "70", "torque_max": "50",
             "expected": "mismatch"},
        ]})

        response = client.post(
            "/api/verify-shape",
            json={
                "shape": "torque_spec",
                "text": "TABLE 7-2: 7/16-20 torque 500-450",
                "values": {"thread_size": "7/16-20", "torque_min": "500",
                           "torque_max": "450"},
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["verified"] is False
        assert body["status"] == "mismatch"


class TestAdoptionGateInSession:
    """Adoption runs the gate first: cases pass -> register; any failure ->
    refuse with the falsifications, registry untouched."""

    def _session_with_tables(self):
        from rag_ingest.demo.session import DemoSession

        session = DemoSession(today=TODAY)
        # Reuse the three-document tabular cluster helper from the discovery
        # flow so the scenario matches what the UI shows.
        flow = TestShapeDiscoveryFlow()
        flow._ingest_tables(session)
        return session

    def _contract(self):
        from rag_ingest.gate_tier import GateTier
        from rag_ingest.shapes import FieldSpec, ShapeContract

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

    def _cases(self):
        from rag_ingest.shapes import GoldCaseSpec

        return GoldCaseSpec.from_claims("torque_spec", [
            {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70"},
            {"thread_size": "1/4-28", "torque_min": "70", "torque_max": "50",
             "expected": "mismatch"},  # gate MUST refuse: constraint proven
        ])

    def test_adopt_runs_cases_and_registers_on_pass(self):
        session = self._session_with_tables()

        result = session.adopt_with_gate(self._contract(), self._cases())

        assert result.registered is True
        assert session.registry().lookup("torque_spec") is not None

    def test_adopt_blocks_when_a_case_falsifies_the_contract(self):
        from rag_ingest.shapes import GoldCaseSpec

        session = self._session_with_tables()
        # A case expects a refusal the contract cannot produce: no enum, and
        # the value is verbatim in the span, so the gate passes it.
        cases = GoldCaseSpec.from_claims("torque_spec", [
            {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70",
             "expected": "verified"},
            {"thread_size": "1/4-28", "torque_min": "70", "torque_max": "50",
             "expected": "verified"},  # gate will REFUSE (min_ref) -> falsifies
        ])

        result = session.adopt_with_gate(self._contract(), cases)

        assert result.registered is False
        assert session.registry().lookup("torque_spec") is None
        assert result.failed_cases
        assert result.failed_cases[0].expected == "verified"
        assert result.failed_cases[0].got == "mismatch"

    def test_failed_adoption_keeps_the_shape_unavailable_for_verification(self):
        from rag_ingest.shapes import GoldCaseSpec

        session = self._session_with_tables()
        cases = GoldCaseSpec.from_claims("torque_spec", [
            {"thread_size": "1/4-28", "torque_min": "70", "torque_max": "50",
             "expected": "verified"},  # will be refused -> falsification
        ])

        result = session.adopt_with_gate(self._contract(), cases)
        assert result.registered is False

        import pytest

        with pytest.raises(KeyError):
            session.check_against_shape(
                "torque_spec", "1/4-28 50-70",
                {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70"},
            )


class TestContractPersistence:
    """Adopted contracts survive a restart: written on adopt, loaded on boot."""

    def _session_with_tables(self):
        from rag_ingest.demo.session import DemoSession

        session = DemoSession(today=TODAY)
        flow = TestShapeDiscoveryFlow()
        flow._ingest_tables(session)
        return session

    def _contract(self):
        from rag_ingest.gate_tier import GateTier
        from rag_ingest.shapes import FieldSpec, ShapeContract

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

    def test_adopt_writes_the_contract_to_disk(self, tmp_path):
        session = self._session_with_tables()
        session.shapes_path = tmp_path / "shapes.json"
        from rag_ingest.shapes import GoldCaseSpec

        cases = GoldCaseSpec.from_claims("torque_spec", [
            {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70"},
            {"thread_size": "1/4-28", "torque_min": "70", "torque_max": "50",
             "expected": "mismatch"},
        ])
        session.adopt_with_gate(self._contract(), cases)

        assert (tmp_path / "shapes.json").exists()

    def test_adopted_contract_survives_a_recreated_session(self, tmp_path):
        from rag_ingest.demo.session import DemoSession
        from rag_ingest.shapes import GoldCaseSpec

        cases = GoldCaseSpec.from_claims("torque_spec", [
            {"thread_size": "1/4-28", "torque_min": "50", "torque_max": "70"},
            {"thread_size": "1/4-28", "torque_min": "70", "torque_max": "50",
             "expected": "mismatch"},
        ])

        first = DemoSession(today=TODAY, shapes_path=tmp_path / "shapes.json")
        TestShapeDiscoveryFlow()._ingest_tables(first)
        first.adopt_with_gate(self._contract(), cases)

        # A "restart": a fresh session pointed at the same file.
        second = DemoSession(today=TODAY, shapes_path=tmp_path / "shapes.json")
        assert second.registry().lookup("torque_spec") is not None
        # And it is the same contract, not just a name.
        assert second.registry().lookup("torque_spec").to_definition() == \
            self._contract().to_definition()

    def test_builtin_shapes_are_not_written_to_disk(self, tmp_path):
        # Persistence records ADOPTED contracts only: builtins come from code.
        from rag_ingest.demo.session import DemoSession

        DemoSession(today=TODAY, shapes_path=tmp_path / "shapes.json")
        path = tmp_path / "shapes.json"
        content = path.read_text() if path.exists() else ""
        assert "torque_spec" not in content

    def test_corrupt_shapes_file_does_not_crash_the_session(self, tmp_path):
        from rag_ingest.demo.session import DemoSession

        (tmp_path / "shapes.json").write_text("{not json")
        session = DemoSession(today=TODAY, shapes_path=tmp_path / "shapes.json")
        # Fails closed to builtins, never a partial registry.
        assert session.registry().lookup("invoice") is not None


class TestRowDiscoveryInSession:
    """Session discovery reports row-level clusters: rows are the unit of
    recurrence, so a quarantined TABLE chunk contributes one observation
    per row -- a recurring row shape across tables is what the owner is
    asked to define."""

    def _ingest_real_tables(self, session):
        # Real TABLE elements: the row-routing path, not paragraph mimicry.
        rows = (
            ("Thread Size", "Tension", "Shear"),
            ("8-36", "12-15", "7-9"),
            ("10-32", "20-25", "12-15"),
            ("1/4-28", "50-70", "30-40"),
        )
        raw, located = RawDocument.from_elements(
            "tbl",
            [
                Element(element_id="p1", kind=ElementKind.PARAGRAPH, page=1,
                        text="Torque values. " * 30),
                Element(element_id="t1", kind=ElementKind.TABLE, page=1,
                        text="", table_rows=rows),
            ],
        )
        session.ingest(doc_id="tbl", filename="t.pdf", parser="test",
                       raw=raw, elements=located)

    def test_discovery_report_includes_row_clusters(self):

        session = _quarantine_session()
        self._ingest_real_tables(session)

        report = session.discovery_report()

        assert report is not None
        row_clusters = [c for c in report.clusters if c.shape.startswith("row:")]
        assert row_clusters, "row observations never reached discovery"

    def test_row_cluster_is_proposable_from_row_observations(self):

        session = _quarantine_session()
        self._ingest_real_tables(session)

        report = session.discovery_report()
        row_cluster = next(
            c for c in report.clusters if c.shape.startswith("row:")
        )

        proposal = session.propose_for(row_cluster.shape)

        assert proposal is not None
        assert proposal.parseable
        names = [f.name for f in proposal.fields]
        assert "thread_size" in names or any("min" in n for n in names)


class TestLlmProposerFallback:
    """When the grammar proposer refuses a cluster, the session may fall
    back to the LLM proposer -- and the UI must see who proposed what."""

    def _session_with_llm(self, llm_proposal):
        import json
        from types import SimpleNamespace

        from rag_ingest.demo.session import DemoSession

        class _FakeCompletions:
            def __init__(self, payload):
                self._payload = payload
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=self._payload)
                    )]
                )

        class _FakeClient:
            def __init__(self, payload):
                self.chat = SimpleNamespace(completions=_FakeCompletions(payload))

        session = DemoSession(
            today=TODAY,
            extractor=TemplateExtractor(),
            llm_proposer=_FakeClient(json.dumps(llm_proposal)),
        )
        return session

    def _ingest_all_text_table(self, session):
        # A table the grammar cannot parse: every data cell is free text,
        # so the grammar proposer refuses it as prose.
        rows = (
            ("부품코드", "상태", "메모"),
            ("AN310-4", "PASS", "정상"),
            ("MS20365-3", "FAIL", "토크부족"),
            ("NAS1223", "HOLD", "재검사"),
        )
        raw, located = RawDocument.from_elements(
            "kr",
            [
                Element(element_id="p1", kind=ElementKind.PARAGRAPH, page=1,
                        text="검사 기록. " * 30),
                Element(element_id="t1", kind=ElementKind.TABLE, page=1,
                        text="", table_rows=rows),
            ],
        )
        session.ingest(doc_id="kr", filename="kr.pdf", parser="test",
                       raw=raw, elements=located)

    def test_grammar_refusal_falls_back_to_llm_and_labels_the_source(self):
        llm_proposal = {
            "propose": True,
            "note": "status column is a closed set",
            "fields": [
                {"name": "part_code", "kind": "text", "column": 0,
                 "sample_value": "AN310-4", "row_index": 1, "required": True},
                {"name": "status", "kind": "enum", "column": 1,
                 "values": ["PASS", "FAIL", "HOLD"], "sample_value": "PASS",
                 "row_index": 1, "required": True},
                {"name": "memo", "kind": "text", "column": 2,
                 "sample_value": "정상", "row_index": 1, "required": False},
            ],
        }
        session = self._session_with_llm(llm_proposal)
        self._ingest_all_text_table(session)

        report = session.discovery_report()
        row_cluster = next(
            c for c in report.clusters if c.shape.startswith("row:")
        )

        proposal = session.propose_for(row_cluster.shape)

        assert proposal is not None
        assert proposal.source == "llm"
        assert proposal.parseable
        assert next(f for f in proposal.fields if f.name == "status").kind == "enum"

    def test_llm_refusal_stays_an_honest_refusal(self):
        llm_proposal = {"propose": False, "note": "cannot distinguish roles", "fields": []}
        session = self._session_with_llm(llm_proposal)
        self._ingest_all_text_table(session)

        report = session.discovery_report()
        row_cluster = next(
            c for c in report.clusters if c.shape.startswith("row:")
        )

        proposal = session.propose_for(row_cluster.shape)

        # Still None to the caller: a refusal is a refusal, whichever layer
        # produced it. The UI never gets a guessed schema to show.
        assert proposal is None


class TestRowVerificationInView:
    """The ingest view carries row verification results so the UI can show
    what an adopted contract actually did to the newest document."""

    def test_verified_rows_appear_in_the_view(self):
        from rag_ingest.shapes import FieldSpec, GoldCaseSpec, ShapeContract

        session = _quarantine_session()
        rows = (
            ("부품코드", "상태", "판정", "메모"),
            ("AN310-4", "PASS", "합격", "정상"),
            ("MS20365-3", "FAIL", "불합격", "손상"),
        )
        raw, located = RawDocument.from_elements(
            "d1",
            [
                Element(element_id="p1", kind=ElementKind.PARAGRAPH, page=1,
                        text="검사 기록. " * 30),
                Element(element_id="t1", kind=ElementKind.TABLE, page=1,
                        text="", table_rows=rows),
            ],
        )
        session.ingest(doc_id="d1", filename="d1.pdf", parser="test",
                       raw=raw, elements=located)

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
        cases = (
            GoldCaseSpec(values={"field_0": "AN310-4", "field_1": "PASS",
                                  "field_2": "합격", "field_3": "정상"},
                          expected="verified"),
            GoldCaseSpec(values={"field_0": "MS20365-3", "field_1": "폐기",
                                  "field_2": "불합격", "field_3": "손상"},
                          expected="mismatch"),
        )
        assert session.adopt_with_gate(contract, cases).registered

        raw2, located2 = RawDocument.from_elements(
            "d2",
            [
                Element(element_id="p1", kind=ElementKind.PARAGRAPH, page=1,
                        text="검사 기록. " * 30),
                Element(element_id="t1", kind=ElementKind.TABLE, page=1,
                        text="", table_rows=rows),
            ],
        )
        view = session.ingest(doc_id="d2", filename="d2.pdf", parser="test",
                              raw=raw2, elements=located2)

        data_rows = [(s2, t) for s2, t in view.row_verifications if "부품코드" not in t]
        assert [s2 for s2, _t in data_rows] == ["grounded", "grounded"]
        # The header row is refused: read as data, its status cell is not in
        # the closed set. Named, not silent.
        assert any(s2 == "mismatch" and "부품코드" in t for s2, t in view.row_verifications)


class TestProseChunksVerifyByDefault:
    """A prose chunk is quotable, not unknowable: ProseExtractor quotes it
    verbatim and the LENIENT gate checks quotation fidelity. The demo session
    shipped with only TemplateExtractor, so every prose chunk stayed
    unclassified -- withheld forever, a question nobody needed to answer.
    The layered extractor is the default the demo should always have had."""

    def test_korean_prose_chunk_verifies_without_any_adoption(self):
        session = DemoSession(today=TODAY)
        raw, located = RawDocument.from_elements(
            "d",
            [Element(element_id="p1", kind=ElementKind.PARAGRAPH, page=1,
                     text="본 대장은 자재창고의 월간 재고 현황을 기록한다. " * 8)],
        )
        view = session.ingest(doc_id="d", filename="d.pdf", parser="test",
                              raw=raw, elements=located)

        assert view.verified_count >= 1, [
            (c.verified, c.text[:40]) for c in view.chunks
        ]


class TestExportEndpoint:
    """The session's honest output as a file: verified chunks only, in a
    format the LightRAG pusher can read back. Quarantined chunks cannot
    appear here -- the endpoint is a projection of the gate, not a choice."""

    def test_export_returns_jsonl_of_verified_chunks_only(self):
        from fastapi.testclient import TestClient

        import rag_ingest.demo.server as server_mod

        raw, elements = _source("본 대장은 자재창고의 월간 재고 현황을 기록한다. " * 10)
        session = DemoSession(today=TODAY)
        session.ingest(doc_id="d1", filename="a.pdf", parser="test",
                       raw=raw, elements=elements)

        original, server_mod._session = server_mod._session, session
        try:
            response = TestClient(server_mod.app).get("/api/export")
        finally:
            server_mod._session = original

        assert response.status_code == 200
        lines = [json.loads(line) for line in response.text.splitlines() if line]
        assert lines
        assert all(line["export_format"] == "rag_export_v1" for line in lines)
        assert all(line["doc_id"] == "d1" for line in lines)


class TestOwnerApprovalPromotion:
    """Approving a review item promotes its chunk out of quarantine -- but
    only through the gate. The demo endpoint is the loop closer: the owner
    answers, the chunk enters retrieval, the next export carries it."""

    def test_approving_unclassified_chunk_promotes_it_as_quote(self):
        raw, elements = _source("본 대장은 자재창고의 월간 재고 현황을 기록한다. " * 8)
        session = DemoSession(today=TODAY, extractor=TemplateExtractor())
        view = session.ingest(doc_id="d1", filename="a.pdf", parser="test",
                              raw=raw, elements=elements)
        item = view.review_items[0]

        result = session.approve_chunk(item.item_id)

        assert result.refusal is None
        fresh = session.views["d1"]
        assert fresh.verified_count == view.verified_count + 1
        assert fresh.quarantined_count == view.quarantined_count - 1
        # The item is answered: the queue no longer holds it.
        assert all(i.item_id != item.item_id for i in session.open_items())

    def test_approved_chunks_carry_their_approval_into_export(self):
        raw, elements = _source("본 대장은 자재창고의 월간 재고 현황을 기록한다. " * 8)
        session = DemoSession(today=TODAY, extractor=TemplateExtractor())
        view = session.ingest(doc_id="d1", filename="a.pdf", parser="test",
                              raw=raw, elements=elements)
        session.approve_chunk(view.review_items[0].item_id)

        rows = session.export_rows()

        approved = [r for r in rows if r.approval is not None]
        assert approved
        assert approved[0].approval.reviewer == "demo-owner"


class TestApproveEndpoint:
    def test_approve_endpoint_promotes_and_reflects_in_the_view(self):
        from fastapi.testclient import TestClient

        import rag_ingest.demo.server as server_mod

        raw, elements = _source("본 대장은 자재창고의 월간 재고 현황을 기록한다. " * 8)
        session = DemoSession(today=TODAY, extractor=TemplateExtractor())
        view = session.ingest(doc_id="d1", filename="a.pdf", parser="test",
                              raw=raw, elements=elements)
        item = view.review_items[0]

        original, server_mod._session = server_mod._session, session
        try:
            response = TestClient(server_mod.app).post(
                f"/api/approve/{item.item_id}"
            )
            export = TestClient(server_mod.app).get("/api/export")
        finally:
            server_mod._session = original

        assert response.status_code == 200
        body = response.json()
        assert body["promoted"] is True
        assert export.status_code == 200
        lines = [json.loads(line) for line in export.text.splitlines() if line]
        assert any(r.get("approval") for r in lines)


class TestDiscoveryAfterPromotion:
    """A promoted chunk is no longer a question, so its rows must not keep
    posing as one: discovery clusters built from stale row observations
    offered Define buttons with no withheld chunk behind them."""

    def test_promoted_table_rows_leave_discovery(self):
        raw, located = RawDocument.from_elements(
            "d1",
            [Element(element_id="t1", kind=ElementKind.TABLE, page=1,
                     text="부서 | 점검항목 | 상태 | 비고\n"
                          "생산팀 | 라인 점검 | 완료 | 특이 없음\n"
                          "품질팀 | 샘플 검사 | 진행중 | 재검사 예정",
                     table_rows=[["부서", "점검항목", "상태", "비고"],
                                 ["생산팀", "라인 점검", "완료", "특이 없음"],
                                 ["품질팀", "샘플 검사", "진행중", "재검사 예정"]])],
            page_sizes={1: (612.0, 792.0)},
        )
        session = DemoSession(today=TODAY, extractor=TemplateExtractor())
        view = session.ingest(doc_id="d1", filename="a.pdf", parser="test",
                              raw=raw, elements=located)
        assert view.quarantined_count >= 1
        assert session.outcomes["d1"].row_observations
        report = session.discovery_report()
        assert report is not None and report.clusters

        session.approve_chunk(view.review_items[0].item_id)

        assert session.discovery_report() is None


class TestStaleItemsCloseWhenChunkVerifies:
    """An item whose chunk verified in the same ingest is not a question the
    owner needs to answer -- the adopted contract answered it. The item must
    close without a label, or the queue nags about answered questions."""

    def test_verified_chunk_item_does_not_stay_open(self, tmp_path):
        raw, located = RawDocument.from_elements(
            "d1",
            [Element(element_id="t1", kind=ElementKind.TABLE, page=1,
                     text="출고일자 | 출고번호 | 수량 | 납품처\\n"
                          "2026-10-12 | SH-1001 | 40 | 부산공장\\n"
                          "2026-10-14 | SH-1002 | 120 | 인천공장\\n"
                          "2026-10-18 | SH-1003 | 75 | 광주공장",
                     table_rows=[["출고일자", "출고번호", "수량", "납품처"],
                                 ["2026-10-12", "SH-1001", "40", "부산공장"],
                                 ["2026-10-14", "SH-1002", "120", "인천공장"],
                                 ["2026-10-18", "SH-1003", "75", "광주공장"]])],
            page_sizes={1: (612.0, 792.0)},
        )
        shapes = tmp_path / "shapes.json"
        import json as _json

        shapes.write_text(_json.dumps([{
            "name": "shipment_spec", "tier": "strict", "fields": [
                {"name": "field_0", "kind": "text", "required": False},
                {"name": "field_1", "kind": "text", "required": False},
                {"name": "field_2", "kind": "number", "required": True},
                {"name": "field_3", "kind": "text", "required": False},
            ],
        }]))
        session = DemoSession(today=TODAY, extractor=TemplateExtractor(),
                              shapes_path=shapes)
        view = session.ingest(doc_id="d1", filename="a.pdf", parser="test",
                              raw=raw, elements=located)

        # The contract verified the table: its "Unrecognised content?" item
        # is answered by the pipeline and must not stay open.
        assert view.verified_count >= 1
        assert session.open_items() == []


class TestLightRagEndpoints:
    """The export pipeline's buttons need server-side proxies: push sends the
    session's verified rows to the configured LightRAG server, clear empties
    it. Both act only on verified output -- the gate's discipline carries to
    the retrieval store, including its destructive edge."""

    def test_push_reports_accepted_and_duplicate_counts(self, monkeypatch):
        from fastapi.testclient import TestClient

        import rag_ingest.demo.server as server_mod

        class _Result:
            accepted, duplicates, failed = ("a1", "a2"), ("d1",), ()

        captured = {}

        def fake_push(rows, poster):
            captured["rows"] = list(rows)
            return _Result()

        monkeypatch.setattr(server_mod, "_push_rows", fake_push)
        original = server_mod._session
        server_mod._session = DemoSession(today=TODAY)
        try:
            response = TestClient(server_mod.app).post("/api/lightrag/push")
        finally:
            server_mod._session = original

        assert response.status_code == 200
        assert response.json() == {"accepted": 2, "duplicates": 1, "failed": 0}

    def test_clear_posts_delete_to_the_configured_base(self, monkeypatch):
        from fastapi.testclient import TestClient

        import rag_ingest.demo.server as server_mod

        captured = {}

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"status": "success"}'

        def fake_urlopen(request, timeout):
            captured["method"] = request.method
            captured["url"] = request.full_url
            return _Response()

        monkeypatch.setattr(server_mod, "_urlopen", fake_urlopen)
        response = TestClient(server_mod.app).post("/api/lightrag/clear")

        assert response.status_code == 200
        assert captured["method"] == "DELETE"
        assert captured["url"].endswith("/documents")


class TestQueueOverflowIsVisible:
    @pytest.fixture
    def client(self):
        from rag_ingest.demo import server

        server._session = DemoSession(today=TODAY, extractor=TemplateExtractor())
        return TestClient(server.app)

    """The WIP cap exists so the queue cannot become a graveyard -- but a cap
    that silently drops questions makes the graveyard invisible instead. A
    slice workflow that overflows must show what was lost, or the owner
    believes everything was asked."""

    def test_dropped_items_are_counted_and_surfaced(self, client, tmp_path, monkeypatch):
        from datetime import date as _date

        import rag_ingest.demo.server as server_mod
        from rag_ingest.escalation import EscalationReason, ReviewItem, ReviewQueue

        session = DemoSession(today=TODAY, extractor=TemplateExtractor())
        session.queue = ReviewQueue(wip_limit=1)
        session.queue.enqueue(ReviewItem(
            item_id="prefill:chunk:triage", doc_id="prefill", chunk_id="chunk",
            reason=EscalationReason.UNCLASSIFIED, question="?", owner_id="o",
            created_on=_date(2026, 10, 1),
        ))
        assert session.queue.open_items()

        original = server_mod._session
        server_mod._session = session
        try:
            path = tmp_path / "a.pdf"
            make_pdf(path)
            with path.open("rb") as handle:
                response = client.post(
                    "/api/ingest?parser=pdfplumber",
                    files={"file": ("a.pdf", handle, "application/pdf")},
                )
        finally:
            server_mod._session = original

        assert response.status_code == 200, response.text
        assert response.json()["review_queue_dropped"] >= 1
