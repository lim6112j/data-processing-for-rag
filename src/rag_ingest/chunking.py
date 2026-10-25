"""Structure-preserving, layout-aware chunking.

Plan §7-1: most real malformed-content pain in RAG is chunk-boundary failure,
not extraction failure -- tables split mid-row, headings divorced from their
sections, references ingested as content. This module fixes that, and it is
therefore the first thing built.

Rules enforced (each has a test):
  * a table chunk is never split across chunks;
  * every chunk carries its section heading path, so a chunk retrieved alone
    still says what it is about;
  * reference/bibliography elements are excluded from retrievable chunks;
  * figure captions stay attached to the block they describe;
  * chunk size is bounded by a target token estimate, degraded gracefully so an
    oversized table is emitted whole rather than truncated.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.docmodel import BBox, Element, ElementKind, ProvenanceRef

#: Rough character-per-token ratio for length estimation without a tokenizer.
CHARS_PER_TOKEN = 4
#: Default chunk target. Small enough for precise retrieval, large enough to
#: hold a full table row group.
DEFAULT_TARGET_TOKENS = 400
#: How far past the target a chunk may grow to absorb one more element.
OVERSHOOT_FACTOR = 1.5


class ChunkKind(StrEnum):
    TEXT = "text"
    TABLE = "table"


class HeadingPath(BaseModel):
    """The heading ancestry a chunk sits under, outermost first."""

    model_config = ConfigDict(frozen=True)

    titles: tuple[str, ...] = Field(default_factory=tuple)

    def pushed(self, title: str, level: int) -> HeadingPath:
        """Return a new path with ``title`` at ``level``.

        Replaces the trailing run of headings at the same or deeper level, so a
        sibling H2 does not accumulate under the previous H2.
        """
        return HeadingPath(titles=(*self.titles[: max(level - 1, 0)], title))

    def __str__(self) -> str:
        return " > ".join(self.titles)


class ChunkSegment(BaseModel):
    """One element's slice of a chunk, with both coordinate systems.

    ``chunk_start``/``chunk_end`` index ``Chunk.text``; ``raw_start``/
    ``raw_end`` index the page text buffer in ``RawDocument``.
    """

    model_config = ConfigDict(frozen=True)

    element_id: str
    page: int = Field(ge=1)
    chunk_start: int = Field(ge=0)
    chunk_end: int = Field(ge=0)
    raw_start: int = Field(ge=0)
    raw_end: int = Field(ge=0)
    bbox: BBox | None = None


class Chunk(BaseModel):
    """A retrievable unit. Immutable once built."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    kind: ChunkKind
    text: str
    heading_path: HeadingPath
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    element_ids: tuple[str, ...]
    #: Union of the element boxes on ``page_start``. T2 crops its evidence from
    #: this; without it T2 has coordinates for nothing and must abstain.
    bbox: BBox | None = None
    #: Where each element sits inside ``text``, and where that element sits in
    #: the raw page. An extractor uses this to turn a match found in the chunk
    #: into a ProvenanceRef -- without it, an extracted value could not be
    #: grounded, which would defeat the whole gate.
    segments: tuple[ChunkSegment, ...] = Field(default_factory=tuple)
    #: Filled by the grounding gate; chunks default to unverified.
    verified: bool = False
    estimated_tokens: int = Field(ge=0)

    def locate(self, offset: int) -> ProvenanceRef | None:
        """Map a chunk-local character offset to a raw page span.

        Returns None when the offset falls outside every segment -- which the
        caller must treat as ungrounded rather than guessing a nearby region.
        """
        for segment in self.segments:
            if segment.chunk_start <= offset < segment.chunk_end:
                delta = offset - segment.chunk_start
                return ProvenanceRef(
                    page=segment.page,
                    char_start=segment.raw_start + delta,
                    char_end=segment.raw_end,
                    element_id=segment.element_id,
                    bbox=segment.bbox,
                )
        return None

    def span_for(self, start: int, end: int) -> ProvenanceRef | None:
        """Provenance for a chunk-local [start, end) match.

        The span is widened to the whole element's raw range. That is deliberate:
        the gate re-derives from the raw text, so pointing at the element that
        contains the match keeps the comparison meaningful even when the chunk
        re-joined lines with different whitespace than the page buffer.
        """
        if end <= start:
            return None
        resolved = self.locate(start)
        if resolved is None:
            return None
        if not self.segments:
            return resolved
        # Prefer the span of the first segment that covers the match start.
        for segment in self.segments:
            if segment.chunk_start <= start < segment.chunk_end:
                return ProvenanceRef(
                    page=segment.page,
                    char_start=segment.raw_start,
                    char_end=segment.raw_end,
                    element_id=segment.element_id,
                    bbox=segment.bbox,
                )
        return resolved

    @property
    def embed_text(self) -> str:
        """Text to embed: heading context is prepended, not thrown away.

        A chunk whose heading lives only in a neighbouring chunk is retrievable
        only by luck.
        """
        if not self.heading_path.titles:
            return self.text
        return f"{self.heading_path}\n\n{self.text}"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


class _Accumulator(BaseModel):
    """Mutable builder for one text chunk (internal; not part of the API)."""

    text_parts: list[str] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    chars: int = 0
    #: Union of element boxes. Kept per page, since bbox coordinates are
    #: page-relative and merging across pages is meaningless.
    bboxes: dict[int, BBox] = Field(default_factory=dict)
    segments: list[ChunkSegment] = Field(default_factory=list)

    def add(self, element: Element) -> None:
        rendered = element.rendered()
        if self.text_parts:
            self.chars += 2  # "\n\n" between blocks
        start = self.chars
        self.text_parts.append(rendered)
        self.chars += len(rendered)
        self.element_ids.append(element.element_id)
        self.segments.append(
            ChunkSegment(
                element_id=element.element_id,
                page=element.page,
                chunk_start=start,
                chunk_end=self.chars,
                raw_start=element.char_start,
                raw_end=element.char_end,
                bbox=element.bbox,
            )
        )
        if self.page_start is None:
            self.page_start = element.page
        self.page_end = element.page
        if element.bbox is not None:
            self.bboxes = {
                **self.bboxes,
                element.page: _union(self.bboxes.get(element.page), element.bbox),
            }

    def bbox_for_first_page(self) -> BBox | None:
        return self.bboxes.get(self.page_start or 1)

    @property
    def tokens(self) -> int:
        return self.chars // CHARS_PER_TOKEN

    def is_empty(self) -> bool:
        return not self.text_parts


class Chunker:
    """Turns located elements into structure-preserving chunks."""

    def __init__(self, target_tokens: int = DEFAULT_TARGET_TOKENS) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        self._target = target_tokens
        self._max = int(target_tokens * OVERSHOOT_FACTOR)

    def chunk(self, doc_id: str, elements: list[Element]) -> list[Chunk]:
        heading_path = HeadingPath()
        chunks: list[Chunk] = []
        buffer = _Accumulator()
        # Heading titles seen at each depth, so a popped section can restore the
        # path of its parent rather than losing it.
        stack: list[tuple[int, str]] = []
        pending_captions: list[Element] = []

        def flush() -> None:
            nonlocal buffer
            if buffer.is_empty():
                return
            chunks.append(self._make_chunk(doc_id, buffer, heading_path, ChunkKind.TEXT))
            buffer = _Accumulator()

        for element in elements:
            if element.kind is ElementKind.REFERENCE:
                # References are navigational noise in a retrieval corpus: they
                # match queries about titles and authors that the user does not
                # want a bibliography entry for.
                flush()
                continue

            if element.is_heading():
                flush()
                level = element.level or 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, element.text.strip()))
                heading_path = HeadingPath(titles=tuple(title for _, title in stack))
                continue

            if element.kind is ElementKind.TABLE:
                # Never split a table and never glue it into surrounding prose:
                # a table row without its header row is worse than no chunk.
                flush()
                table_buffer = _Accumulator()
                for caption in pending_captions:
                    table_buffer.add(caption)
                pending_captions.clear()
                table_buffer.add(element)
                chunks.append(
                    self._make_chunk(doc_id, table_buffer, heading_path, ChunkKind.TABLE)
                )
                continue

            if element.kind is ElementKind.FIGURE_CAPTION:
                # Hold the caption until the next block arrives, then attach it
                # there. A caption alone in a chunk describes nothing.
                flush()
                pending_captions.append(element)
                continue

            if pending_captions:
                for pending in pending_captions:
                    buffer.add(pending)
                pending_captions.clear()

            projected = buffer.tokens + estimate_tokens(element.rendered())
            if not buffer.is_empty() and projected > self._max:
                flush()
            buffer.add(element)
            if buffer.tokens >= self._target:
                flush()

        flush()

        if pending_captions:
            chunks.append(
                self._make_chunk(
                    doc_id,
                    _accumulate(pending_captions),
                    heading_path,
                    ChunkKind.TEXT,
                )
            )
        return chunks

    def _make_chunk(
        self,
        doc_id: str,
        buffer: _Accumulator,
        heading_path: HeadingPath,
        kind: ChunkKind,
    ) -> Chunk:
        text = "\n\n".join(buffer.text_parts)
        return Chunk(
            chunk_id=f"{doc_id}:{kind.value}:{buffer.element_ids[0]}",
            doc_id=doc_id,
            kind=kind,
            text=text,
            heading_path=heading_path,
            page_start=buffer.page_start or 1,
            page_end=buffer.page_end or buffer.page_start or 1,
            element_ids=tuple(buffer.element_ids),
            segments=tuple(buffer.segments),
            bbox=buffer.bbox_for_first_page(),
            verified=False,
            estimated_tokens=estimate_tokens(text),
        )


def _union(existing: BBox | None, new: BBox) -> BBox:
    """Smallest box containing both. Bounds a multi-element chunk's evidence."""
    if existing is None:
        return new
    left, top, right, bottom = existing
    n_left, n_top, n_right, n_bottom = new
    return (min(left, n_left), min(top, n_top), max(right, n_right), max(bottom, n_bottom))


def _accumulate(elements: list[Element]) -> _Accumulator:
    buffer = _Accumulator()
    for element in elements:
        buffer.add(element)
    return buffer
