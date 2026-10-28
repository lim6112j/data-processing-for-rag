"""Table → row routing: a table chunk is N candidate rows, not one blob.

A quarantined table chunk currently appears to discovery as a single
``tabular`` observation, so a 14-row torque table is one question ("what is
this table?") instead of 14 rows sharing a shape. Row routing decomposes the
chunk into row-level observations *before* discovery, so the recurring row
shape -- thread size, tension range, shear range, two limits -- is what the
owner is asked to define.

The routing is mechanical: rows come from the table's pipe-joined rendering
(that the gate already re-derives against), and a row's provenance spans its
own slice of the chunk, resolvable through the chunk's existing segments.
No model, no layout heuristics: a row is a line of the rendered table.
"""

from __future__ import annotations

from rag_ingest.chunking import Chunker
from rag_ingest.docmodel import Element, ElementKind, RawDocument
from rag_ingest.table_rows import row_observations


def _rows() -> tuple[tuple[str, ...], ...]:
    return (
        ("Thread Size", "Tension", "Shear", "MS20365", "MS20364"),
        ("8-36", "12-15", "7-9", "20", "12"),
        ("10-32", "20-25", "12-15", "40", "25"),
        ("1/4-28", "50-70", "30-40", "100", "60"),
    )


def _build(table_rows: tuple[tuple[str, ...], ...]):
    elements = [
        Element(
            element_id="p1",
            kind=ElementKind.PARAGRAPH,
            page=1,
            text="Surrounding prose context for the table below.",
        ),
        Element(
            element_id="t1",
            kind=ElementKind.TABLE,
            page=1,
            text="",
            table_rows=table_rows,
        ),
    ]
    raw, located = RawDocument.from_elements("doc1", elements)
    # The chunker must receive the located elements -- the ones with resolved
    # char offsets -- exactly as the pipeline does, or row spans cannot ground.
    chunks = Chunker(target_tokens=5).chunk("doc1", located)
    table_chunks = [c for c in chunks if c.kind.value == "table"]
    return raw, located, table_chunks[0]


class TestRowObservations:
    def test_each_data_row_becomes_one_observation(self):
        _raw, elements, chunk = _build(_rows())

        observations = row_observations(chunk, elements)

        # Header + 3 data rows; every rendered line of the table is a row.
        assert len(observations) == 4

    def test_row_cells_preserve_the_table_columns(self):
        _raw, elements, chunk = _build(_rows())

        observations = row_observations(chunk, elements)
        data = [o for o in observations if o.cells[0] == "10-32"]

        assert len(data) == 1
        assert data[0].cells == ("10-32", "20-25", "12-15", "40", "25")

    def test_row_provenance_resolves_through_chunk_segments(self):
        """A row's span must be groundable: it locates in the page buffer."""
        raw, elements, chunk = _build(_rows())

        observations = row_observations(chunk, elements)
        data = next(o for o in observations if o.cells[0] == "1/4-28")

        ref = chunk.span_for(data.chunk_start, data.chunk_end)
        assert ref is not None
        # The span text in the buffer contains the row's first cell verbatim.
        span = raw.span_text(ref)
        assert "1/4-28" in span

    def test_a_non_table_chunk_yields_nothing(self):
        _raw, elements, _chunk = _build(_rows())
        text_chunks = [
            c for c in Chunker(target_tokens=5).chunk("doc1", elements)
            if c.kind.value != "table"
        ]
        assert text_chunks  # the page text must be there for this test to mean anything

        for chunk in text_chunks:
            assert row_observations(chunk, elements) == ()

    def test_blank_cells_are_preserved_not_dropped(self):
        """A dropped cell shifts every later column -- routing must not fix up
        the data, only report what the parser produced."""
        _raw, elements, chunk = _build((("A", "", "C"), ("1", "", "3")))

        observations = row_observations(chunk, elements)

        assert observations[1].cells == ("1", "", "3")


class TestPipelineRowRouting:
    """The pipeline records row observations for quarantined table chunks,
    so discovery clusters rows, not blobs."""

    def _ingest(self):
        from rag_ingest.escalation import ReviewQueue
        from rag_ingest.extractors import NoExtractor
        from rag_ingest.pipeline import IngestPipeline

        rows = (
            ("Thread Size", "Tension", "Shear", "MS20365", "MS20364"),
            ("8-36", "12-15", "7-9", "20", "12"),
            ("10-32", "20-25", "12-15", "40", "25"),
            ("1/4-28", "50-70", "30-40", "100", "60"),
        )
        elements = [
            Element(
                element_id="p1",
                kind=ElementKind.PARAGRAPH,
                page=1,
                text="Torque limits for installation. " * 20,
            ),
            Element(
                element_id="t1",
                kind=ElementKind.TABLE,
                page=1,
                text="",
                table_rows=rows,
            ),
        ]
        raw, located = RawDocument.from_elements("doc1", elements)
        pipeline = IngestPipeline(
            ReviewQueue(), NoExtractor(), chunker=Chunker(target_tokens=5)
        )
        outcome = pipeline.ingest(raw, located, owner_id="owner")
        return outcome, elements

    def test_outcome_carries_row_observations_for_a_quarantined_table(self):
        outcome, _elements = self._ingest()

        assert outcome.row_observations  # the table was quarantined and routed
        assert len(outcome.row_observations) == 4

    def test_row_observations_carry_their_cells(self):
        outcome, _elements = self._ingest()

        data = [o for o in outcome.row_observations if o.cells[0] == "1/4-28"]
        assert len(data) == 1
        assert data[0].cells == ("1/4-28", "50-70", "30-40", "100", "60")


class TestRowDiscovery:
    """Row observations cluster by cell-kind signature: the recurring row
    shape is what discovery surfaces, at row granularity."""

    def _rows(self, n_docs: int = 1):
        from rag_ingest.table_rows import RowObservation

        rows = []
        for d in range(n_docs):
            for thread, tension, shear, ms1, ms2 in (
                ("8-36", "12-15", "7-9", "20", "12"),
                ("10-32", "20-25", "12-15", "40", "25"),
                ("1/4-28", "50-70", "30-40", "100", "60"),
            ):
                rows.append(
                    RowObservation(
                        doc_id=f"doc{d}",
                        page=1,
                        cells=(thread, tension, shear, ms1, ms2),
                        row_text=f"{thread} | {tension} | {shear} | {ms1} | {ms2}",
                        chunk_start=0,
                        chunk_end=40,
                    )
                )
        return rows

    def test_rows_cluster_into_a_row_shape(self):
        from rag_ingest.discovery import discover_from_rows

        report = discover_from_rows(self._rows(1))

        assert len(report.clusters) >= 1
        top = report.clusters[0]
        assert top.chunks == 3

    def test_row_cluster_spreads_across_documents(self):
        from rag_ingest.discovery import discover_from_rows

        report = discover_from_rows(self._rows(2))

        top = report.clusters[0]
        assert top.documents == 2
        assert top.is_candidate
