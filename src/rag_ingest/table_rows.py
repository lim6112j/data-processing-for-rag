"""Table → row routing: decompose a table chunk into row-level observations.

A table is not one fact; it is N rows sharing a shape. When a table chunk is
quarantined, discovery sees a single ``tabular`` blob and the owner is asked
"what is this table?" -- the wrong question. The right unit is the row: a
14-row torque table is 14 observations of the same shape, and recurrence is
exactly what discovery counts on.

This module performs that decomposition mechanically. The rows come from the
table's pipe-joined rendering (``"a | b | c"`` per line) -- the same form the
gate re-derives provenance against -- and each row's provenance is its own
slice of the chunk, resolvable through the chunk's existing
``ChunkSegment``s. No model, no layout heuristics: a row is a line of the
rendered table, nothing more.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rag_ingest.chunking import Chunk
from rag_ingest.docmodel import Element, ElementKind


class RowObservation(BaseModel):
    """One table row as discovery can count it.

    ``cells`` preserve what the parser produced, blanks included -- routing
    reports data, it never fixes it up. ``chunk_start``/``chunk_end`` index
    ``Chunk.text`` so ``Chunk.span_for`` can turn them into a provenance
    reference that grounds.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    page: int
    cells: tuple[str, ...]
    row_text: str
    chunk_start: int
    chunk_end: int


def row_observations(
    chunk: Chunk,
    elements: list[Element],
) -> tuple[RowObservation, ...]:
    """Split a table chunk into one observation per rendered row line.

    A non-table chunk yields nothing: routing is a table-only concern, and
    inventing rows out of prose would be the hallucination this pipeline
    refuses. The element list is the chunk's own elements (with resolved char
    offsets), matched by id so multi-table pages stay unambiguous.
    """
    if chunk.kind.value != "table":
        return ()

    by_id = {e.element_id: e for e in elements}
    observations: list[RowObservation] = []
    for segment in chunk.segments:
        element = by_id.get(segment.element_id)
        if element is None or element.kind is not ElementKind.TABLE:
            continue
        _emit_rows(chunk, segment.chunk_start, element, observations)
    return tuple(observations)


def _emit_rows(
    chunk: Chunk,
    segment_start: int,
    element: Element,
    out: list[RowObservation],
) -> None:
    rows = element.table_rows or ()
    if not rows:
        return
    # The rendered form is authoritative: it is what the page buffer holds,
    # so spans computed against it re-derive correctly.
    rendered_lines = element.rendered().split("\n")
    for index, cells in enumerate(rows):
        if index >= len(rendered_lines):
            break
        row_text = rendered_lines[index]
        line_start = segment_start + sum(
            len(line) + 1 for line in rendered_lines[:index]
        )
        out.append(
            RowObservation(
                doc_id=chunk.doc_id,
                page=element.page,
                cells=tuple(cells),
                row_text=row_text,
                chunk_start=line_start,
                chunk_end=line_start + len(row_text),
            )
        )


__all__ = ["RowObservation", "row_observations"]
