"""Raw document model and the adapter boundary to a layout parser.

Deliberately dependency-free: the pipeline consumes a small, stable element
tree that a Docling/Unstructured adapter can produce. Keeping this boundary
narrow means the chunker and the grounding gate can be tested without a heavy
PDF stack installed.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ElementScores(BaseModel):
    """Per-element confidence reported by the layout/OCR pipeline.

    Mirrors Docling's confidence report: ``layout_score``, ``ocr_score`` and
    ``parse_score``. ``table_score`` is deliberately *absent* -- upstream
    documents it as not yet implemented, so table quality must be derived from
    cell coverage and grounding re-derivation instead of trusting a score that
    does not exist.
    """

    model_config = ConfigDict(frozen=True)

    layout_score: float | None = Field(default=None, ge=0.0, le=1.0)
    ocr_score: float | None = Field(default=None, ge=0.0, le=1.0)
    parse_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def is_empty(self) -> bool:
        return (
            self.layout_score is None
            and self.ocr_score is None
            and self.parse_score is None
        )


class ElementKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    FIGURE_CAPTION = "figure_caption"
    REFERENCE = "reference"


#: (left, top, right, bottom) in page coordinate space. Carried through from the
#: parser so downstream consumers can crop the evidence region.
BBox = tuple[float, float, float, float]


class ProvenanceRef(BaseModel):
    """Points at an exact region of the raw source.

    Two coordinate systems, both optional-but-meaningful:

    * ``char_start``/``char_end`` index the page text buffer. The grounding gate
      uses these to re-derive a value.
    * ``bbox`` locates the same region on the rendered page. T2 uses this to
      crop the image it needs in order to adjudicate what the text alone cannot
      settle (is that a lowercase L or a one?).

    A value without a resolvable span cannot be grounded, and therefore cannot
    be verified. ``page`` is 1-based.
    """

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    element_id: str | None = None
    bbox: BBox | None = None

    def is_well_formed(self) -> bool:
        return self.char_end > self.char_start

    @property
    def has_visual_evidence(self) -> bool:
        """True when this ref can produce an image crop for T2.

        Without this, T2 can only see text -- and the questions T2 exists to
        answer are exactly the ones text cannot settle.
        """
        return self.bbox is not None


class Element(BaseModel):
    """One structural unit as the parser saw it, with its confidence scores."""

    model_config = ConfigDict(frozen=True)

    element_id: str
    kind: ElementKind
    page: int = Field(ge=1)
    text: str
    level: int | None = Field(default=None, ge=1, le=6)
    table_rows: tuple[tuple[str, ...], ...] | None = None
    scores: ElementScores = Field(default_factory=ElementScores)

    #: Location on the rendered page. A Docling adapter fills this from
    #: ``item.prov[0].bbox``. Losing it silently disables T2's evidence channel,
    #: which is why ``RawDocument.from_elements`` carries it forward into every
    #: ProvenanceRef it hands out.
    bbox: BBox | None = None

    # Offsets are assigned by RawDocument.from_elements; they are absolute
    # offsets into that page's raw text buffer.
    char_start: int = Field(default=0, ge=0)
    char_end: int = Field(default=0, ge=0)

    def provenance(self) -> ProvenanceRef:
        """This element's region as a provenance reference.

        The single place that should turn an element into a ref -- so bbox
        cannot be forgotten at a call site.
        """
        return ProvenanceRef(
            page=self.page,
            char_start=self.char_start,
            char_end=self.char_end,
            element_id=self.element_id,
            bbox=self.bbox,
        )

    def is_heading(self) -> bool:
        return self.kind is ElementKind.HEADING

    def rendered(self) -> str:
        """Text as it appears in the raw buffer.

        Tables keep their cells joined with a pipe so that a re-derivation pass
        can locate a numeric value inside the span it was claimed from.
        """
        if self.kind is ElementKind.TABLE and self.table_rows:
            return "\n".join(" | ".join(row) for row in self.table_rows)
        return self.text


class RawPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_no: int = Field(ge=1)
    text: str
    #: Rendered page image, set by an adapter that can produce one. T2 crops
    #: ``ProvenanceRef.bbox`` out of this. Absent for text-only sources, which
    #: correctly disables T2's visual evidence channel rather than silently
    #: falling back to a text-only judgement.
    image_png: bytes | None = None
    #: Page extent in the same coordinate space as ``Element.bbox``. Required to
    #: map a bbox onto the rendered image: without it the overlay (and any
    #: evidence crop) would be a guess about scale.
    width: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)

    @property
    def char_len(self) -> int:
        return len(self.text)

    @property
    def has_size(self) -> bool:
        return self.width is not None and self.height is not None

    @property
    def has_image(self) -> bool:
        return self.image_png is not None


class RawDocument(BaseModel):
    """The raw source, preserved verbatim alongside any normalized view.

    Retrieval can fall back to this: page-level raw text is always retrievable,
    so a document whose normalized chunks were quarantined is never invisible.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    pages: tuple[RawPage, ...]

    def page(self, page_no: int) -> RawPage | None:
        for candidate in self.pages:
            if candidate.page_no == page_no:
                return candidate
        return None

    def span_text(self, ref: ProvenanceRef) -> str | None:
        """Return the raw text a provenance reference designates.

        Returns None when the reference is malformed or outside the buffer --
        which the gate treats as unverifiable, never as a pass.
        """
        page = self.page(ref.page)
        if page is None:
            return None
        if ref.char_start < 0 or ref.char_end > page.char_len:
            return None
        if ref.char_end <= ref.char_start:
            return None
        return page.text[ref.char_start : ref.char_end]

    def visual_evidence(self, ref: ProvenanceRef) -> bytes | None:
        """Return an image crop for a provenance reference, if available.

        This is what T2 sees. Returns None when the page has no rendered image
        or the ref carries no bbox -- and None must be treated as "T2 cannot
        judge this", never as "judge it from text". Text alone cannot answer
        the questions T2 exists for.

        Cropping is done by the caller's image library rather than here so this
        module stays dependency-free; ``VisualEvidence`` states the contract.
        """
        page = self.page(ref.page)
        if page is None or page.image_png is None or ref.bbox is None:
            return None
        return page.image_png

    @classmethod
    def from_elements(
        cls,
        doc_id: str,
        elements: list[Element],
        *,
        page_images: dict[int, bytes] | None = None,
        page_sizes: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[RawDocument, list[Element]]:
        """Build a raw buffer from elements and rewrite their char offsets.

        Page text is the joined rendering of its elements, which is what a
        text buffer extracted by a parser looks like. Elements are returned
        with absolute page offsets filled in so callers never hand-compute
        offsets (a rich source of silent provenance bugs).

        ``page_images`` carries rendered pages (1-based page number -> PNG
        bytes) from an adapter that can produce them. Without them T2 has no
        visual evidence and must decline to adjudicate rather than guess.
        ``Element.bbox`` is preserved untouched, since it is in page coordinate
        space and independent of char offsets.
        """
        images = page_images or {}
        sizes = page_sizes or {}
        by_page: dict[int, list[Element]] = {}
        for element in elements:
            by_page.setdefault(element.page, []).append(element)

        pages: list[RawPage] = []
        located: list[Element] = []
        for page_no in sorted(by_page):
            cursor = 0
            parts: list[str] = []
            for element in by_page[page_no]:
                rendered = element.rendered()
                if parts:
                    cursor += 1  # the "\n" that joins elements
                start = cursor
                parts.append(rendered)
                cursor += len(rendered)
                located.append(
                    element.model_copy(update={"char_start": start, "char_end": cursor})
                )
            width, height = sizes.get(page_no, (None, None))
            pages.append(
                RawPage(
                    page_no=page_no,
                    text="\n".join(parts),
                    image_png=images.get(page_no),
                    width=width,
                    height=height,
                )
            )

        return cls(doc_id=doc_id, pages=tuple(pages)), located
