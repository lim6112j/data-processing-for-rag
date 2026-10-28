"""Unicode normalization: one canonical form at the buffer boundary.

Korean business documents arrive in mixed normalization forms -- macOS PDF
tooling emits NFD (jamo decomposed), databases and copy-pasted text carry NFC.
Python compares strings by code point, so NFC "한국" != NFD "한국" as strings
even though they are the same characters, and a decomposed form is LONGER in
code points (15 -> 19 for one short line), which shifts every char offset in
every ProvenanceRef if the page buffer and the chunk text disagree.

The rule: NFKC at the entry point, before offsets exist. NFKC (not NFC)
because full-width digits and symbols (U+FF11..U+FF34, full-width won) are
a parser's glyph transfer choice, not content; leaving them breaks both the
amount regex and currency detection while naming no real difference in the
document.

This is normalization of representation, not repair: the abstract characters
are unchanged, so the mark-never-repair policy is not engaged. It happens once
at the boundary rather than at every comparison site, because per-site
normalization cannot keep char offsets coherent -- only a single canonical
form before offsets are computed can.

RUF001 is ignored file-wide: the full-width characters below ARE the test
data -- a lint that folded them away would unplug the fixtures.
"""

from __future__ import annotations

from rag_ingest.docmodel import Element, ElementKind, RawDocument

BBOX = (10.0, 20.0, 400.0, 60.0)


def _element(text: str, *, element_id: str = "e1") -> Element:
    return Element(
        element_id=element_id,
        kind=ElementKind.PARAGRAPH,
        page=1,
        text=text,
        bbox=BBOX,
    )


class TestPageBufferNormalization:
    def test_decomposed_hangul_is_composed_in_the_buffer(self):
        decomposed = "총액 1,234.56 KRW"  # NFC input...
        import unicodedata as ud

        decomposed = ud.normalize("NFD", decomposed)  # ...delivered as NFD
        raw, elements = RawDocument.from_elements("d1", [_element(decomposed)])

        assert raw.pages[0].text == ud.normalize("NFKC", decomposed)
        # And the element's own text is canonicalized in step, so chunk text
        # and page buffer never disagree.
        assert elements[0].text == raw.pages[0].text

    def test_offsets_are_coherent_with_the_normalized_buffer(self):
        # The failure D exists to prevent: if the buffer were normalized after
        # offsets were computed, char_end would point past the real text. The
        # span must resolve and round-trip to the element text.
        import unicodedata as ud

        text = ud.normalize("NFD", "청구서 합계 1,234.56 원")
        raw, elements = RawDocument.from_elements("d1", [_element(text)])

        el = elements[0]
        assert raw.pages[0].text[el.char_start : el.char_end] == el.text

    def test_fullwidth_digits_and_symbols_are_folded(self):
        raw, _ = RawDocument.from_elements(
            "d1", [_element("Invoice １２３４ total ￦１２３４．５６")]
        )

        assert "１２３４" not in raw.pages[0].text
        assert "1234" in raw.pages[0].text
        assert "￦" not in raw.pages[0].text
        assert "₩" in raw.pages[0].text

    def test_verbatim_quotation_survives_the_round_trip(self):
        # The LENIENT tier quotes the chunk text and grounds it against the
        # page buffer. Both sides are canonicalized at the same boundary, so
        # quoting still matches -- this is the regression a normalization bug
        # would cause most visibly.
        import unicodedata as ud

        text = ud.normalize("NFD", "캐스터너트는 균열을 검사해야 한다.")
        raw, elements = RawDocument.from_elements("d1", [_element(text)])
        from rag_ingest.chunking import Chunker

        chunk = Chunker().chunk("d1", elements)[0]
        ref = chunk.locate(0)
        assert ref is not None
        assert raw.span_text(ref) is not None
        assert chunk.text in raw.pages[0].text

    def test_already_normalized_text_is_untouched(self):
        text = "Invoice INV-2041 total 1234.56 KRW"
        raw, elements = RawDocument.from_elements("d1", [_element(text)])

        assert raw.pages[0].text == text
        assert elements[0].text == text
