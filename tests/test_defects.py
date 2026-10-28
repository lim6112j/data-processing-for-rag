"""Defect marking: admitted text carries visible warnings instead of repairs.

The policy being locked in here is the one the project decided on and the
literature supports (survey.md §1: LLM repair max match 0.028; §6.1: human edit
hit rate 20-30%): never repair, never quarantine damaged prose, **mark it**.
A consumer that sees the mark cannot silently correct the value; a consumer
that does not see it will.

The mark must be mechanical. Same rule as everywhere else in T0/T1: no model.
"""

from __future__ import annotations

from rag_ingest.defects import scan_defects
from rag_ingest.docmodel import Element, ElementKind, RawDocument
from rag_ingest.escalation import ReviewQueue
from rag_ingest.extractors import LayeredExtractor
from rag_ingest.ontology import DefectCode
from rag_ingest.pipeline import IngestPipeline

BBOX = (10.0, 20.0, 400.0, 60.0)


def _paragraph(text: str, *, element_id: str = "e1") -> Element:
    return Element(
        element_id=element_id,
        kind=ElementKind.PARAGRAPH,
        page=1,
        text=text,
        bbox=BBOX,
    )


def _ingest(*texts: str):
    elements = [_paragraph(t, element_id=f"e{i}") for i, t in enumerate(texts)]
    raw, elements = RawDocument.from_elements("doc", elements)
    return IngestPipeline(ReviewQueue(), LayeredExtractor()).ingest(
        raw, elements, owner_id="o"
    )


class TestScanDefects:
    def test_damaged_amount_is_flagged(self):
        # The canonical damage: OCR read lowercase l for 1.
        assert scan_defects("recorded as l,234.56 in the ledger") == (
            DefectCode.SUSPECTED_OCR_DAMAGE,
        )

    def test_pipe_for_one_is_flagged(self):
        assert scan_defects("total |,234.56 due") == (DefectCode.SUSPECTED_OCR_DAMAGE,)

    def test_clean_amount_is_not_flagged(self):
        assert scan_defects("recorded as 1,234.56 in the ledger") == ()

    def test_no_defects_for_plain_prose(self):
        assert scan_defects("Castle nuts should be inspected for cracks.") == ()

    def test_damaged_prose_is_quarantined_only_if_the_gate_says_so(self):
        # The scanner reports what it saw; admission stays the gate's decision.
        assert scan_defects("") == ()


class TestMarkingNotRepairing:
    def test_damaged_prose_is_admitted_and_marked(self):
        outcome = _ingest("The ledger recorded the total as l,234.56 last quarter.")

        assert len(outcome.verified_chunks) == 1
        record = outcome.verified_chunks[0]
        assert record.verified is True
        assert record.defects == (DefectCode.SUSPECTED_OCR_DAMAGE,)

    def test_clean_prose_is_admitted_without_a_mark(self):
        outcome = _ingest("The ledger recorded the total as 1,234.56 last quarter.")

        record = outcome.verified_chunks[0]
        assert record.verified is True
        assert record.defects == ()

    def test_the_text_is_never_repaired(self):
        # The whole point: the stored text is what the source says, damage
        # included. Anything else would be an unrecorded correction.
        damaged = "The ledger recorded the total as l,234.56 last quarter."
        outcome = _ingest(damaged)

        assert outcome.verified_chunks[0].text == damaged

    def test_damaged_prose_stays_retrievable(self):
        # Marking must not degrade into filtering: the chunk still answers
        # queries, which is why the owner chose admission over quarantine.
        from rag_ingest.store import ChunkStore, VerificationFilter

        text = "The ledger recorded the total as l,234.56 last quarter."
        raw, elements = RawDocument.from_elements("doc", [_paragraph(text)])
        outcome = IngestPipeline(ReviewQueue(), LayeredExtractor()).ingest(
            raw, elements, owner_id="o"
        )
        store = ChunkStore()
        store.add_raw(raw)
        for record in outcome.verified_chunks + outcome.quarantined_chunks:
            store.add(record)

        hits = store.search("l,234.56", mode=VerificationFilter.VERIFIED_ONLY)

        assert [h.record.chunk_id for h in hits] == [
            r.chunk_id for r in outcome.verified_chunks
        ]
        assert hits[0].record.defects == (DefectCode.SUSPECTED_OCR_DAMAGE,)

    def test_quarantined_records_are_not_marked_admitted(self):
        # A mark is a statement about text that entered retrieval. Stamping it on
        # quarantined chunks would blur the one distinction the store keeps.
        outcome = _ingest("INVOICE")

        assert outcome.verified_chunks == ()
        assert all(r.defects == () for r in outcome.quarantined_chunks)

    def test_mark_survives_into_the_store(self):
        from rag_ingest.store import ChunkStore

        text = "The ledger recorded the total as l,234.56 last quarter."
        raw, elements = RawDocument.from_elements("doc", [_paragraph(text)])
        outcome = IngestPipeline(ReviewQueue(), LayeredExtractor()).ingest(
            raw, elements, owner_id="o"
        )
        store = ChunkStore()
        store.add_raw(raw)
        for record in outcome.verified_chunks + outcome.quarantined_chunks:
            store.add(record)

        assert store.record(outcome.verified_chunks[0].chunk_id).defects == (
            DefectCode.SUSPECTED_OCR_DAMAGE,
        )
