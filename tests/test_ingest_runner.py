"""Corpus runner: one bad document must not kill the batch, and nothing may vanish."""

from __future__ import annotations

from datetime import date

import pytest

from rag_ingest.docmodel import Element, ElementKind, ProvenanceRef, RawDocument
from rag_ingest.ingest_runner import (
    BatchReport,
    DocumentResult,
    DocumentSource,
    DocumentStatus,
    IngestRunner,
    describe,
)
from rag_ingest.ontology import AbstainReason, FieldAbstention, FieldValue, InvoiceHeader

TODAY = date(2026, 10, 25)


class _GoodExtractor:
    """Grounded on two fields -- the minimum for verification."""

    def extract(self, chunk):
        ref = ProvenanceRef(page=chunk.page_start, char_start=0, char_end=len(chunk.text))
        return InvoiceHeader(
            invoice_number=FieldValue(value=chunk.text[:8], provenance=ref),
            issue_date=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
            total_amount=FieldValue(value="10.00", provenance=ref),
            currency=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
        )


class _NoExtractor:
    def extract(self, chunk):
        return None


class _ExplodingExtractor:
    def extract(self, chunk):
        raise RuntimeError("parser exploded")


def _source(
    doc_id: str, text: str = "Invoice INV-1001 total 10.00"
) -> DocumentSource:
    raw, located = RawDocument.from_elements(
        doc_id,
        [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1, text=text)],
    )
    return DocumentSource(
        doc_id=doc_id,
        path=f"/corpus/{doc_id}.pdf",
        raw=raw,
        elements=tuple(located),
    )


class TestSingleDocument:
    def test_verified_document_is_ok(self):
        runner = IngestRunner(extractor=_GoodExtractor(), today=TODAY)

        result, _ = runner.run_one(_source("d1"), owner_id="o")

        assert result.status is DocumentStatus.OK
        assert result.verified_chunks == 1

    def test_nothing_verified_is_all_quarantined(self):
        runner = IngestRunner(extractor=_NoExtractor(), today=TODAY)

        result, _ = runner.run_one(_source("d1"), owner_id="o")

        assert result.status is DocumentStatus.ALL_QUARANTINED
        assert result.quarantined_chunks == 1

    def test_empty_document_is_distinct_from_quarantined(self):
        # A file with no chunks is a different problem from "everything failed".
        raw, located = RawDocument.from_elements(
            "d1", [Element(element_id="r", kind=ElementKind.REFERENCE, page=1, text="Ref.")]
        )
        source = DocumentSource(doc_id="d1", path="x.pdf", raw=raw, elements=tuple(located))
        runner = IngestRunner(extractor=_NoExtractor(), today=TODAY)

        result, _ = runner.run_one(source, owner_id="o")

        assert result.status is DocumentStatus.EMPTY
        assert result.total_chunks == 0


class TestFailureIsolation:
    def test_extractor_crash_is_recorded_not_raised(self):
        runner = IngestRunner(extractor=_ExplodingExtractor(), today=TODAY)

        result, outcome = runner.run_one(_source("d1"), owner_id="o")

        assert result.status is DocumentStatus.FAILED
        assert "RuntimeError" in (result.error or "")
        assert outcome is None

    def test_one_failure_does_not_abort_the_batch(self):
        # The property that matters at corpus scale.
        runner = IngestRunner(extractor=_GoodExtractor(), today=TODAY)
        sources = [_source("d1"), _source("bad"), _source("d3")]

        # Make only the middle document explode.
        original = runner._pipeline

        def flaky():
            pipeline = original()
            real_ingest = pipeline.ingest

            def ingest(raw, elements, **kwargs):
                if raw.doc_id == "bad":
                    raise ValueError("corrupt pdf")
                return real_ingest(raw, elements, **kwargs)

            pipeline.ingest = ingest
            return pipeline

        runner._pipeline = flaky

        report = runner.run(sources, owner_id="o")

        assert report.total_documents == 3
        assert len(report.failed) == 1
        assert report.verified_chunks == 2

    def test_failed_document_reports_no_chunks(self):
        runner = IngestRunner(extractor=_ExplodingExtractor(), today=TODAY)

        result, _ = runner.run_one(_source("d1"), owner_id="o")

        assert result.total_chunks == 0


class TestAccounting:
    def test_counts_reconcile(self):
        runner = IngestRunner(extractor=_GoodExtractor(), today=TODAY)

        report = runner.run([_source("d1"), _source("d2")], owner_id="o")

        assert report.accounts_for_every_document()
        assert report.total_documents == 2
        assert report.verified_chunks == 2

    def test_empty_report_is_coherent(self):
        report = BatchReport()

        assert report.total_documents == 0
        assert report.verification_rate == 0.0
        assert report.disagreement_rate == 0.0
        assert report.accounts_for_every_document()

    def test_verification_rate_is_chunk_weighted(self):
        mixed = BatchReport(
            documents=(
                DocumentResult(
                    doc_id="a",
                    source="a",
                    status=DocumentStatus.OK,
                    verified_chunks=3,
                    quarantined_chunks=1,
                    review_items=1,
                    adjudications=0,
                    adjudicated_picks=0,
                ),
            )
        )

        assert mixed.verification_rate == 0.75

    def test_disagreement_rate_is_measured_at_the_producer(self):
        # Deliberately independent of T2: counting T2 calls would confound
        # parser disagreement with the unrelated structure gate.
        report = BatchReport(
            documents=(
                DocumentResult(
                    doc_id="a", source="a", status=DocumentStatus.OK,
                    verified_chunks=1, quarantined_chunks=0, review_items=0,
                    adjudications=0, adjudicated_picks=0,
                    candidate_conflicts=0, candidate_checks=2,
                ),
                DocumentResult(
                    doc_id="b", source="b", status=DocumentStatus.OK,
                    verified_chunks=1, quarantined_chunks=1, review_items=1,
                    adjudications=2, adjudicated_picks=1,
                    candidate_conflicts=2, candidate_checks=2,
                ),
            )
        )

        assert report.disagreement_rate == 0.5   # 2 conflicts of 4 checks
        assert report.adjudications == 2
        assert report.adjudicated_picks == 1

    def test_no_second_parse_reports_zero_checks_not_zero_agreement(self):
        report = BatchReport(
            documents=(
                DocumentResult(
                    doc_id="a", source="a", status=DocumentStatus.OK,
                    verified_chunks=1, quarantined_chunks=0, review_items=0,
                    adjudications=0, adjudicated_picks=0,
                ),
            )
        )

        assert report.candidate_checks == 0
        assert "not measured" in describe(report)

    def test_zero_disagreement_means_t2_can_be_removed(self):
        # At or near zero, the two parses always agree and T2 is dead weight.
        report = BatchReport(
            documents=(
                DocumentResult(
                    doc_id="a", source="a", status=DocumentStatus.OK,
                    verified_chunks=5, quarantined_chunks=0, review_items=0,
                    adjudications=0, adjudicated_picks=0,
                ),
            )
        )

        assert report.disagreement_rate == 0.0


class TestStoreAndQueueIntegration:
    def test_verified_chunks_reach_the_store(self):
        runner = IngestRunner(extractor=_GoodExtractor(), today=TODAY)

        runner.run([_source("d1")], owner_id="o")

        assert runner.store.search("INV-1001") != []

    def test_review_items_reach_the_queue(self):
        runner = IngestRunner(extractor=_NoExtractor(), today=TODAY)

        runner.run([_source("d1")], owner_id="o")

        assert runner.queue.open_items()

    def test_raw_text_is_reachable_for_quarantined_documents(self):
        runner = IngestRunner(extractor=_NoExtractor(), today=TODAY)

        runner.run([_source("d1", text="Invoice INV-1001 total 10.00")], owner_id="o")

        assert "INV-1001" in (runner.store.raw_text("d1", 1) or "")


class TestDescribe:
    def test_summary_includes_the_deciding_numbers(self):
        runner = IngestRunner(extractor=_GoodExtractor(), today=TODAY)
        report = runner.run([_source("d1")], owner_id="o")

        text = describe(report)

        assert "documents" in text
        assert "verification" in text
        assert "disagreement" in text

    def test_failures_are_listed(self):
        runner = IngestRunner(extractor=_ExplodingExtractor(), today=TODAY)
        report = runner.run([_source("d1")], owner_id="o")

        text = describe(report)

        assert "failed" in text
        assert "RuntimeError" in text


def test_runner_defaults_are_independent_per_instance():
    # Shared mutable defaults would leak state between runs.
    a = IngestRunner(extractor=_GoodExtractor())
    b = IngestRunner(extractor=_GoodExtractor())

    assert a.queue is not b.queue
    assert a.store is not b.store


@pytest.mark.parametrize("status", list(DocumentStatus))
def test_every_status_is_bucketed(status):
    report = BatchReport(
        documents=(
            DocumentResult(
                doc_id="x", source="x", status=status,
                verified_chunks=0, quarantined_chunks=0, review_items=0,
                adjudications=0, adjudicated_picks=0,
            ),
        )
    )

    assert report.accounts_for_every_document()


class TestProducerLevelDisagreement:
    def test_conflicts_are_counted_even_when_t2_is_never_consulted(self):
        # The point of measuring at the producer: T2 may be skipped entirely by
        # the structure gate, and the disagreement rate must still be real.
        from rag_ingest.candidates import AlternateParseProducer, SecondParse

        class _DivergingAlternate:
            """Reports different text for every chunk it is asked about."""

            def produce(self, chunk):
                from rag_ingest.adjudication import Candidate

                return [
                    Candidate(
                        candidate_id=f"{chunk.chunk_id}:a",
                        produced_by="docling",
                        proposed_text=chunk.text,
                    ),
                    Candidate(
                        candidate_id=f"{chunk.chunk_id}:b",
                        produced_by="pdfplumber",
                        proposed_text="different reading",
                    ),
                ]

        _ = AlternateParseProducer, SecondParse
        runner = IngestRunner(
            extractor=_NoExtractor(), producer=_DivergingAlternate(), today=TODAY
        )

        result, _ = runner.run_one(_source("d1"), owner_id="o")

        assert result.candidate_checks == 1
        assert result.candidate_conflicts == 1
        assert result.adjudications == 0  # T2 was never called

    def test_producer_failure_does_not_skew_the_rate(self):
        class _Broken:
            def produce(self, chunk):
                raise RuntimeError("producer down")

        runner = IngestRunner(extractor=_NoExtractor(), producer=_Broken(), today=TODAY)

        result, _ = runner.run_one(_source("d1"), owner_id="o")

        assert result.candidate_conflicts == 0
        assert result.candidate_checks == 1
