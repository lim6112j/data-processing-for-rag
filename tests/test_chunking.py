"""Each test pins one structure-preserving rule from the plan (section 7-1)."""

from __future__ import annotations

from rag_ingest.chunking import Chunker, ChunkKind, HeadingPath
from tests.conftest import caption, heading, paragraph, reference, table


def test_table_is_never_split_across_chunks():
    # Arrange: a table far larger than the chunk target.
    rows = tuple(("item", str(i), "100") for i in range(50))
    chunker = Chunker(target_tokens=10)

    # Act
    chunks = chunker.chunk("d1", [table(rows)])

    # Assert
    assert len(chunks) == 1
    assert chunks[0].kind is ChunkKind.TABLE
    assert chunks[0].text.count("\n") == len(rows) - 1


def test_every_chunk_carries_its_heading_path():
    chunker = Chunker(target_tokens=5)

    chunks = chunker.chunk(
        "d1",
        [
            heading("Financials", level=1, element_id="h1"),
            heading("Q3 Results", level=2, element_id="h2"),
            paragraph("Revenue grew substantially this quarter.", element_id="p1"),
            paragraph("Margins also improved across every segment.", element_id="p2"),
        ],
    )

    text_chunks = [c for c in chunks if c.kind is ChunkKind.TEXT]
    assert text_chunks
    for chunk in text_chunks:
        assert chunk.heading_path.titles == ("Financials", "Q3 Results")


def test_heading_context_is_prepended_to_embed_text():
    chunker = Chunker(target_tokens=1000)

    chunks = chunker.chunk(
        "d1",
        [heading("Appendix", level=1, element_id="h1"), paragraph("Details.", element_id="p1")],
    )

    assert chunks[0].embed_text.startswith("Appendix")
    assert "Details." in chunks[0].embed_text


def test_sibling_headings_do_not_nest_under_each_other():
    chunker = Chunker(target_tokens=1000)

    chunks = chunker.chunk(
        "d1",
        [
            heading("One", level=1, element_id="h1"),
            paragraph("First body.", element_id="p1"),
            heading("Two", level=1, element_id="h2"),
            paragraph("Second body.", element_id="p2"),
        ],
    )

    paths = [chunk.heading_path.titles for chunk in chunks if chunk.kind is ChunkKind.TEXT]
    assert paths == [("One",), ("Two",)]


def test_nested_headings_are_restored_after_popping():
    chunker = Chunker(target_tokens=1000)

    chunks = chunker.chunk(
        "d1",
        [
            heading("Top", level=1, element_id="h1"),
            heading("Sub", level=2, element_id="h2"),
            paragraph("Under sub.", element_id="p1"),
            heading("Sibling", level=2, element_id="h3"),
            paragraph("Under sibling.", element_id="p2"),
        ],
    )

    paths = [chunk.heading_path.titles for chunk in chunks if chunk.kind is ChunkKind.TEXT]
    assert paths == [("Top", "Sub"), ("Top", "Sibling")]
    assert "Top" in paths[1]


def test_references_are_excluded_from_chunks():
    chunker = Chunker(target_tokens=1000)

    chunks = chunker.chunk(
        "d1",
        [
            paragraph("Body text that should be retrievable.", element_id="p1"),
            reference("Smith, J. (2020). A Title Nobody Wants To Retrieve.", element_id="r1"),
        ],
    )

    assert len(chunks) == 1
    assert "Smith" not in chunks[0].text


def test_caption_stays_attached_to_its_table():
    chunker = Chunker(target_tokens=1000)

    chunks = chunker.chunk(
        "d1",
        [
            paragraph("Intro prose.", element_id="p1"),
            caption("Table 1: Quarterly revenue by segment.", element_id="c1"),
            table((("segment", "revenue"), ("retail", "120")), element_id="t1"),
        ],
    )

    table_chunks = [c for c in chunks if c.kind is ChunkKind.TABLE]
    assert len(table_chunks) == 1
    assert "Table 1" in table_chunks[0].text
    assert "Quarterly revenue" in table_chunks[0].text


def test_chunk_size_is_bounded_by_target():
    chunker = Chunker(target_tokens=10)

    chunks = chunker.chunk(
        "d1",
        [
            paragraph(f"Paragraph number {i} with a reasonable amount of text.", element_id=f"p{i}")
            for i in range(20)
        ],
    )

    assert len(chunks) > 1
    # Every chunk except possibly an oversized single element stays near target.
    for chunk in chunks:
        assert chunk.estimated_tokens <= 15


def test_heading_path_pushed_replaces_deeper_run():
    path = HeadingPath(titles=("A", "B", "C"))

    assert path.pushed("D", level=2).titles == ("A", "D")
    assert path.pushed("E", level=1).titles == ("E",)
