"""Multi-page quotation: a chunk spanning pages can still be cited honestly.

The old refusal: ``ProvenanceRef`` cites one page, so a multi-page chunk could
not be quoted through a single resolvable span, and quoting only its first page
would silently understate what was admitted. Measured cost on the FAA manual:
32 of 754 chunks (4.2%) quarantined for exactly this reason.

The fix keeps the single-page invariant of ``ProvenanceRef`` -- which the
grounding gate, the chunker and the demo's bbox overlay all rely on -- and
adds a composition: a multi-page quote cites a *tuple* of per-page spans, one
per page it touches. ``RawDocument.span_text`` resolves both shapes, and the
LENIENT check verifies that the quoted text is the join of the per-page slices,
in order. Same guarantee as before, just across a boundary: every character of
the quote is provably in the source at the claimed place.
"""

from __future__ import annotations

import unicodedata

from rag_ingest.chunking import Chunker
from rag_ingest.docmodel import Element, ElementKind, RawDocument
from rag_ingest.extractors import ProseExtractor
from rag_ingest.grounding import GroundingGate, GroundingStatus
from rag_ingest.ontology import ProseKnowledge


def _multi_page_doc():
    elements = [
        Element(element_id="e0", kind=ElementKind.PARAGRAPH, page=1,
                text="First page paragraph about torque values."),
        Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1,
                text="Continues with more detail here."),
        Element(element_id="e5", kind=ElementKind.PARAGRAPH, page=2,
                text="Second page begins a new paragraph."),
        Element(element_id="e6", kind=ElementKind.PARAGRAPH, page=2,
                text="And finishes the procedure."),
    ]
    raw, located = RawDocument.from_elements("d", elements)
    chunk = Chunker().chunk("d", located)[0]
    return raw, chunk


class TestMultiPageQuotation:
    def test_multi_page_chunk_is_quoted_not_refused(self):
        _raw, chunk = _multi_page_doc()

        extraction = ProseExtractor().extract(chunk)

        assert isinstance(extraction, ProseKnowledge)
        assert "torque values" in extraction.text.value

    def test_the_quote_carries_one_span_per_page(self):
        _raw, chunk = _multi_page_doc()

        extraction = ProseExtractor().extract(chunk)

        refs = extraction.text.multi_page_spans
        assert refs is not None
        assert [ref.page for ref in refs] == [1, 2]

    def test_gate_verifies_a_multi_page_quote(self):
        raw, chunk = _multi_page_doc()

        extraction = ProseExtractor().extract(chunk)
        result = GroundingGate(raw).check(extraction)

        assert result.status is GroundingStatus.GROUNDED
        assert result.verified

    def test_tampered_multi_page_quote_is_mismatch(self):
        raw, chunk = _multi_page_doc()

        extraction = ProseExtractor().extract(chunk)
        # Edit the middle of the quote: grounding must fail it, exactly as it
        # would for a single-page quote. The composition must not weaken the
        # fidelity guarantee.
        edited = extraction.text.value.replace("torque values", "torque figures")
        tampered = extraction.model_copy(
            update={"text": extraction.text.model_copy(update={"value": edited})}
        )

        result = GroundingGate(raw).check(tampered)

        assert result.status is GroundingStatus.MISMATCH

    def test_tampered_page_two_slice_is_mismatch_even_if_page_one_matches(self):
        # The adversarial case for a per-page tuple: quote page 1 verbatim,
        # invent page 2. A check that only verified "some span matches" would
        # pass this. The join check must not.
        raw, chunk = _multi_page_doc()

        extraction = ProseExtractor().extract(chunk)
        edited = extraction.text.value.replace("finishes the procedure", "invents the ending")
        tampered = extraction.model_copy(
            update={"text": extraction.text.model_copy(update={"value": edited})}
        )

        result = GroundingGate(raw).check(tampered)

        assert result.status is GroundingStatus.MISMATCH

    def test_single_page_quotes_are_unchanged(self):
        # The composition must not disturb the 96%-verified working path.
        elements = [Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=1,
                            text="A single page paragraph about castle nuts.")]
        raw, located = RawDocument.from_elements("d", elements)
        chunk = Chunker().chunk("d", located)[0]

        extraction = ProseExtractor().extract(chunk)

        assert isinstance(extraction, ProseKnowledge)
        assert extraction.text.multi_page_spans is None
        assert GroundingGate(raw).check(extraction).verified

    def test_unicode_normalization_applies_to_multi_page_quotes(self):
        # The buffer is NFKC; a quote assembled from per-page slices must match
        # it after the same normalization, including when NFD input straddles
        # the page boundary.
        elements = [
            Element(element_id="e0", kind=ElementKind.PARAGRAPH, page=1,
                    text=unicodedata.normalize("NFD", "첫 번째 페이지 절차")),
            Element(element_id="e1", kind=ElementKind.PARAGRAPH, page=2,
                    text=unicodedata.normalize("NFD", "두 번째 페이지로 이어진다")),
        ]
        raw, located = RawDocument.from_elements("d", elements)
        chunk = Chunker().chunk("d", located)[0]

        extraction = ProseExtractor().extract(chunk)
        assert isinstance(extraction, ProseKnowledge)
        assert GroundingGate(raw).check(extraction).verified
