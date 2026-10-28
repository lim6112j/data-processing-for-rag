"""Shape discovery over quarantined chunks.

The contract: only Unclassified observations are shape evidence -- a
grounding failure belongs to an existing shape's extractor, not to a missing
shape. Conflating the two would recommend new shapes to fix parser defects.
"""

from __future__ import annotations

from datetime import date

from rag_ingest.discovery import discover
from rag_ingest.docmodel import Element, ElementKind, RawDocument
from rag_ingest.escalation import ReviewQueue
from rag_ingest.ontology import Unclassified
from rag_ingest.pipeline import IngestPipeline


class _AllUnclassified:
    """Emits Unclassified with the same shape hint the real extractors use --
    the discovery input the pipeline actually produces."""

    def extract(self, chunk):
        from rag_ingest.extractors import _shape_hint, _whole_chunk_span

        return Unclassified(
            observed_shape=_shape_hint(chunk),
            provenance=_whole_chunk_span(chunk),
        )


def _run(texts: list[tuple[str, int, str]]) -> tuple:
    """texts: (text, page, element_id). Every chunk becomes unclassified."""
    elements = [
        Element(element_id=eid, kind=ElementKind.PARAGRAPH, page=page, text=text)
        for text, page, eid in texts
    ]
    raw, located = RawDocument.from_elements("d", elements)
    from rag_ingest.chunking import Chunker

    outcome = IngestPipeline(
        ReviewQueue(), _AllUnclassified(), chunker=Chunker(target_tokens=5), today=date(2026, 1, 1)
    ).ingest(raw, located, owner_id="o")
    return outcome


class TestDiscovery:
    def test_recurring_shape_across_pages_is_a_candidate(self):
        outcome = _run([
            ("| a | b |\n| c | d |", 1, "e1"),
            ("| e | f |\n| g | h |", 2, "e2"),
            ("| i | j |\n| k | l |", 3, "e3"),
        ])

        report = discover(outcome.quarantined_chunks)

        assert report.unclassified_total == 3
        assert report.candidates[0].shape == "tabular"
        assert report.candidates[0].chunks == 3
        assert report.candidates[0].pages == 3

    def test_single_page_shape_is_not_a_candidate(self):
        # Three chunks, all on one page: a document quirk, not a corpus shape.
        outcome = _run([
            ("| a | b |", 1, "e1"),
            ("| c | d |", 1, "e2"),
            ("| e | f |", 1, "e3"),
        ])

        report = discover(outcome.quarantined_chunks)

        assert report.clusters[0].shape == "tabular"
        assert report.candidates == ()

    def test_one_occurrence_is_noise(self):
        outcome = _run([("some ordinary prose", 1, "e1")])

        report = discover(outcome.quarantined_chunks)

        assert report.clusters[0].chunks == 1
        assert report.candidates == ()

    def test_grounding_failures_are_not_shape_evidence(self):
        # An invoice-shaped chunk that failed grounding has a shape -- the
        # problem is the value, not the ontology. It must not inflate a
        # tabular/prose cluster.
        from rag_ingest.docmodel import ProvenanceRef
        from rag_ingest.ontology import FieldValue, InvoiceHeader
        from rag_ingest.pipeline import _chunk_to_record

        chunk = _chunk_stub()
        from rag_ingest.ontology import AbstainReason, FieldAbstention

        lying = InvoiceHeader(
            invoice_number=FieldValue(
                value="WRONG", provenance=ProvenanceRef(page=1, char_start=0, char_end=5)
            ),
            issue_date=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
            total_amount=FieldValue(
                value="WRONG", provenance=ProvenanceRef(page=1, char_start=0, char_end=5)
            ),
            currency=FieldAbstention(reason=AbstainReason.NOT_PRESENT),
        )
        record = _chunk_to_record(chunk, verified=False, extraction=lying, tier=None)

        report = discover((record,))

        assert report.unclassified_total == 0

    def test_sample_text_carries_the_first_chunk(self):
        outcome = _run([
            ("| torque | values |\n| 50 | 70 |", 1, "e1"),
            ("| pressure | psi |\n| 30 | 40 |", 2, "e2"),
        ])

        report = discover(outcome.quarantined_chunks)

        assert "torque" in report.clusters[0].sample_text


def _chunk_stub():
    from rag_ingest.chunking import Chunk, ChunkKind, HeadingPath

    return Chunk(
        chunk_id="d:text:e1",
        doc_id="d",
        kind=ChunkKind.TEXT,
        heading_path=HeadingPath(titles=()),
        text="Some text",
        page_start=1,
        page_end=1,
        element_ids=("e1",),
        segments=(),
        estimated_tokens=1,
    )
