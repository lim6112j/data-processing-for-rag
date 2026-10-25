"""Demo layer: the HTTP contract and the state behind it.

The UI itself is verified by hand in a browser; these tests pin the parts that
break silently -- routing, id handling, and the review loop's effect on the
scorecard.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient
from helpers.pdf_fixture import make_pdf

from rag_ingest.demo.session import DemoSession
from rag_ingest.docmodel import Element, ElementKind, RawDocument
from rag_ingest.grounding import GroundingStatus

TODAY = date(2026, 10, 25)


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
        session = DemoSession(today=TODAY)

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
        session = DemoSession(today=TODAY)
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
        server._session = DemoSession(today=TODAY)
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
        assert "could not read total amount" in question

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

        server._session = DemoSession(today=TODAY)
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
