"""Chunks must carry a bbox, or T2 has coordinates for nothing."""

from __future__ import annotations

from rag_ingest.chunking import Chunker
from tests.conftest import heading, paragraph, table

BBOX_A = (10.0, 20.0, 100.0, 40.0)
BBOX_B = (10.0, 50.0, 120.0, 80.0)


def _para(text: str, element_id: str, bbox):
    return paragraph(text, element_id=element_id).model_copy(update={"bbox": bbox})


class TestChunkBbox:
    def test_single_element_chunk_inherits_its_bbox(self):
        chunks = Chunker(target_tokens=1000).chunk(
            "d1", [_para("Some text.", "p1", BBOX_A)]
        )

        assert chunks[0].bbox == BBOX_A

    def test_multi_element_chunk_unions_its_boxes(self):
        chunks = Chunker(target_tokens=1000).chunk(
            "d1", [_para("One.", "p1", BBOX_A), _para("Two.", "p2", BBOX_B)]
        )

        # (left, top) is the min; (right, bottom) is the max.
        assert chunks[0].bbox == (10.0, 20.0, 120.0, 80.0)

    def test_elements_without_bbox_yield_no_chunk_bbox(self):
        chunks = Chunker(target_tokens=1000).chunk(
            "d1", [paragraph("No box.", element_id="p1")]
        )

        assert chunks[0].bbox is None

    def test_mixed_elements_union_only_the_boxed_ones(self):
        chunks = Chunker(target_tokens=1000).chunk(
            "d1", [_para("Boxed.", "p1", BBOX_A), paragraph("Unboxed.", element_id="p2")]
        )

        assert chunks[0].bbox == BBOX_A

    def test_boxes_are_not_merged_across_pages(self):
        # Coordinates are page-relative; merging across pages is meaningless.
        first = _para("Page one.", "p1", BBOX_A)
        second = _para("Page two.", "p2", BBOX_B).model_copy(update={"page": 2})

        chunks = Chunker(target_tokens=1000).chunk("d1", [first, second])

        assert chunks[0].bbox == BBOX_A  # first page's box only

    def test_table_chunk_keeps_its_bbox(self):
        boxed_table = table((("a", "b"),), element_id="t1").model_copy(
            update={"bbox": BBOX_A}
        )
        chunks = Chunker(target_tokens=1000).chunk("d1", [boxed_table])

        assert chunks[0].bbox == BBOX_A

    def test_heading_path_does_not_disturb_bbox(self):
        chunks = Chunker(target_tokens=1000).chunk(
            "d1",
            [
                heading("Section", level=1, element_id="h1").model_copy(
                    update={"bbox": (0.0, 0.0, 500.0, 10.0)}
                ),
                _para("Body.", "p1", BBOX_A),
            ],
        )

        # The heading is structural, not part of the body chunk's evidence.
        assert chunks[0].bbox == BBOX_A
