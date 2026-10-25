"""Alignment is the part that decides whether T2 is asked a meaningful question.

A misaligned pair produces a confident verdict about two different regions,
which is worse than not adjudicating. These tests pin the alignment rules.
"""

from __future__ import annotations

import pytest

from rag_ingest.candidates import (
    AlternateParseProducer,
    NullProducer,
    SecondParse,
    boxes_agree,
    containment,
    describe_disagreement,
    iou,
)
from rag_ingest.chunking import Chunk, ChunkKind, HeadingPath
from rag_ingest.docmodel import Element, ElementKind

CHUNK_BOX = (100.0, 100.0, 300.0, 200.0)


def _chunk(text: str = "a | b\nc | d", bbox=CHUNK_BOX, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id="d1:table:e1",
        doc_id="d1",
        kind=ChunkKind.TABLE,
        text=text,
        heading_path=HeadingPath(),
        page_start=page,
        page_end=page,
        element_ids=("e1",),
        bbox=bbox,
        estimated_tokens=5,
    )


def _element(text: str, bbox, page: int = 1, element_id: str = "alt") -> Element:
    return Element(
        element_id=element_id,
        kind=ElementKind.PARAGRAPH,
        page=page,
        text=text,
        bbox=bbox,
    )


class TestGeometry:
    def test_identical_boxes_have_iou_one(self):
        assert iou(CHUNK_BOX, CHUNK_BOX) == pytest.approx(1.0)

    def test_disjoint_boxes_have_iou_zero(self):
        assert iou(CHUNK_BOX, (500.0, 500.0, 600.0, 600.0)) == 0.0

    def test_partial_overlap_is_between_zero_and_one(self):
        shifted = (150.0, 100.0, 350.0, 200.0)
        assert 0.0 < iou(CHUNK_BOX, shifted) < 1.0

    def test_degenerate_box_does_not_divide_by_zero(self):
        assert iou((10.0, 10.0, 10.0, 10.0), CHUNK_BOX) == 0.0

    def test_containment_of_inner_box_is_one(self):
        inner = (120.0, 120.0, 180.0, 160.0)
        assert containment(inner, CHUNK_BOX) == pytest.approx(1.0)

    def test_boxes_agree_on_good_overlap(self):
        assert boxes_agree(CHUNK_BOX, (110.0, 105.0, 295.0, 195.0))

    def test_boxes_disagree_when_far_apart(self):
        assert not boxes_agree(CHUNK_BOX, (400.0, 400.0, 500.0, 500.0))

    def test_nested_element_still_agrees(self):
        # A second parser may return one table row where the first returned the
        # whole table. That is a granularity difference, not a different region.
        row = (100.0, 100.0, 300.0, 130.0)
        assert boxes_agree(CHUNK_BOX, row)


class TestNullProducer:
    def test_single_reading_means_no_disagreement(self):
        # One parser, therefore nothing for T2 to decide.
        assert len(NullProducer().produce(_chunk())) == 1

    def test_empty_chunk_yields_no_candidates(self):
        assert NullProducer().produce(_chunk(text="   ")) == []


class TestAlternateParseProducer:
    def test_genuine_difference_produces_two_candidates(self):
        alternate = SecondParse(
            name="pdfplumber",
            elements=(_element("a | b | c\nc | d", (110.0, 105.0, 295.0, 195.0)),),
        )
        producer = AlternateParseProducer(primary_name="docling", alternate=alternate)

        candidates = producer.produce(_chunk())

        assert len(candidates) == 2
        assert {c.produced_by for c in candidates} == {"docling", "pdfplumber"}

    def test_identical_readings_produce_no_disagreement(self):
        # Two parsers agreeing is not a disagreement; spending a model call to
        # learn that is waste.
        alternate = SecondParse(
            name="pdfplumber",
            elements=(_element("a | b\nc | d", (110.0, 105.0, 295.0, 195.0)),),
        )
        producer = AlternateParseProducer(primary_name="docling", alternate=alternate)

        assert len(producer.produce(_chunk())) == 1

    def test_whitespace_only_difference_is_not_a_disagreement(self):
        alternate = SecondParse(
            name="pdfplumber",
            elements=(_element("a | b  \n\n  c | d", (110.0, 105.0, 295.0, 195.0)),),
        )
        producer = AlternateParseProducer(primary_name="docling", alternate=alternate)

        assert len(producer.produce(_chunk())) == 1

    def test_missing_counterpart_is_silent_by_default(self):
        # Cross-origin parsers segment differently, so a missing counterpart is
        # usually a segmentation difference, not a contradiction. Treating it as
        # disagreement manufactured noise on nearly every paragraph.
        producer = AlternateParseProducer(
            primary_name="docling",
            alternate=SecondParse(name="pdfplumber", elements=()),
        )

        assert len(producer.produce(_chunk())) == 1

    def test_missing_counterpart_can_be_enabled_explicitly(self):
        # A true absence is a real signal -- but opt-in, per document type.
        producer = AlternateParseProducer(
            primary_name="docling",
            alternate=SecondParse(name="pdfplumber", elements=()),
            allow_missing=True,
        )

        candidates = producer.produce(_chunk())

        assert len(candidates) == 2
        assert candidates[1].proposed_text == ""
        assert "no content" in (candidates[1].note or "")

    def test_missing_counterpart_can_be_suppressed(self):
        producer = AlternateParseProducer(
            primary_name="docling",
            alternate=SecondParse(name="pdfplumber", elements=()),
            allow_missing=False,
        )

        assert len(producer.produce(_chunk())) == 1

    def test_unrelated_element_is_not_aligned(self):
        # A far-away element must never become the "second opinion".
        alternate = SecondParse(
            name="pdfplumber",
            elements=(_element("completely different", (800.0, 800.0, 900.0, 900.0)),),
        )
        producer = AlternateParseProducer(
            primary_name="docling", alternate=alternate, allow_missing=False
        )

        assert len(producer.produce(_chunk())) == 1

    def test_elements_from_another_page_are_not_aligned(self):
        alternate = SecondParse(
            name="pdfplumber",
            elements=(_element("other page text", CHUNK_BOX, page=2),),
        )
        producer = AlternateParseProducer(
            primary_name="docling", alternate=alternate, allow_missing=False
        )

        assert len(producer.produce(_chunk(page=1))) == 1

    def test_chunk_without_bbox_yields_no_candidates(self):
        # Without a box the two readings cannot be shown to describe the same
        # region, and aligning on text would beg the question.
        alternate = SecondParse(
            name="pdfplumber", elements=(_element("x", CHUNK_BOX),)
        )
        producer = AlternateParseProducer(primary_name="docling", alternate=alternate)

        assert producer.produce(_chunk(bbox=None)) == []

    def test_multiple_matching_elements_are_joined(self):
        alternate = SecondParse(
            name="pdfplumber",
            elements=(
                _element("row one", (100.0, 100.0, 300.0, 150.0), element_id="a"),
                _element("row two", (100.0, 150.0, 300.0, 200.0), element_id="b"),
            ),
        )
        producer = AlternateParseProducer(primary_name="docling", alternate=alternate)

        candidates = producer.produce(_chunk())

        assert len(candidates) == 2
        assert "row one" in candidates[1].proposed_text
        assert "row two" in candidates[1].proposed_text

    def test_element_without_bbox_is_skipped(self):
        alternate = SecondParse(
            name="pdfplumber", elements=(_element("no box", None),)
        )
        producer = AlternateParseProducer(
            primary_name="docling", alternate=alternate, allow_missing=False
        )

        assert len(producer.produce(_chunk())) == 1


class TestDescribeDisagreement:
    def test_no_disagreement_yields_empty_string(self):
        assert describe_disagreement([]) == ""

    def test_disagreement_is_summarised_by_producer(self):
        alternate = SecondParse(
            name="pdfplumber",
            elements=(_element("a | b | c", (110.0, 105.0, 295.0, 195.0)),),
        )
        candidates = AlternateParseProducer(
            primary_name="docling", alternate=alternate
        ).produce(_chunk())

        summary = describe_disagreement(candidates)

        assert "docling" in summary
        assert "pdfplumber" in summary


class TestPageScaleGuard:
    def test_page_sized_element_does_not_match_every_chunk(self):
        # A parser returning one element for the whole page must not become the
        # "second opinion" for every chunk on it.
        page_box = (0.0, 0.0, 612.0, 792.0)
        small_chunk = (500.0, 400.0, 540.0, 430.0)

        assert not boxes_agree(page_box, small_chunk)

    def test_moderate_nesting_still_matches(self):
        # A whole table vs one of its rows is within the ratio.
        assert boxes_agree(CHUNK_BOX, (100.0, 100.0, 300.0, 130.0))

    def test_containment_is_symmetric(self):
        inner = (120.0, 120.0, 180.0, 160.0)
        assert boxes_agree(inner, CHUNK_BOX)
        assert boxes_agree(CHUNK_BOX, inner)


class TestCrossParserSegmentation:
    def test_coarse_alternate_block_does_not_produce_a_false_disagreement(self):
        # The real failure observed with Docling + pdfplumber: pdfplumber emits
        # one text block covering most of the page, so a Docling paragraph
        # overlaps it via containment. That is a segmentation difference, not a
        # contradiction, and must not become a T2 question.
        page_scale = (72.0, 62.0, 220.0, 140.0)
        paragraph = _chunk(text="Quarterly report", bbox=(72.0, 63.0, 156.0, 74.0))
        alternate = SecondParse(
            name="pdfplumber",
            elements=(_element("Quarterly report Widget 1200", page_scale),),
        )
        producer = AlternateParseProducer(primary_name="docling", alternate=alternate)

        # Either no alignment (different granularity) or a same-text match --
        # never a "pdfplumber read nothing" contradiction.
        candidates = producer.produce(paragraph)
        assert len(candidates) <= 2
        assert all(c.proposed_text != "" for c in candidates)
