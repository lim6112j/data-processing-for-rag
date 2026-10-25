"""Plan section 5a: fail closed per chunk, never per document."""

from __future__ import annotations

import pytest

from rag_ingest.docmodel import Element, ElementKind, RawDocument
from rag_ingest.ontology import ChunkRecord
from rag_ingest.store import ChunkStore, VerificationFilter, content_hash


def _raw(doc_id: str = "d1", text: str = "raw page text") -> RawDocument:
    raw, _ = RawDocument.from_elements(
        doc_id, [Element(element_id="p1", kind=ElementKind.PARAGRAPH, page=1, text=text)]
    )
    return raw


def _record(chunk_id: str, text: str, *, verified: bool, doc_id: str = "d1") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        embed_text=text,
        verified=verified,
    )


class TestFailClosedPerChunk:
    def test_unverified_chunks_are_excluded_by_default(self):
        store = ChunkStore()
        store.add_raw(_raw())
        store.add(_record("c1", "revenue growth", verified=True))
        store.add(_record("c2", "revenue guess", verified=False))

        hits = store.search("revenue")

        assert [hit.record.chunk_id for hit in hits] == ["c1"]

    def test_verified_chunk_stays_retrievable_when_sibling_is_quarantined(self):
        # The 200-page document case: one bad chunk must not hide the rest.
        store = ChunkStore()
        store.add_raw(_raw())
        store.add(_record("good", "margin improved", verified=True))
        store.add(_record("bad", "margin unreadable", verified=False))

        assert [hit.record.chunk_id for hit in store.search("margin")] == ["good"]

    def test_opt_in_mode_includes_unverified_but_ranks_them_below(self):
        store = ChunkStore()
        store.add_raw(_raw())
        store.add(_record("v", "margin", verified=True))
        store.add(_record("u", "margin", verified=False))

        hits = store.search("margin", mode=VerificationFilter.INCLUDE_UNVERIFIED)

        assert hits[0].record.chunk_id == "v"
        assert hits[1].record.chunk_id == "u"
        assert hits[1].rank_penalized is True

    def test_raw_page_text_remains_reachable(self):
        # Plan 7-3: quarantined content is never invisible.
        store = ChunkStore()
        store.add_raw(_raw(text="original scan text"))

        assert store.raw_text("d1", 1) == "original scan text"
        assert store.raw_text("d1", 2) is None
        assert store.raw_text("nope", 1) is None


class TestIngestGuards:
    def test_chunk_requires_its_raw_document_first(self):
        store = ChunkStore()

        with pytest.raises(ValueError):
            store.add(_record("c1", "text", verified=True))

    def test_reingest_replaces_rather_than_duplicates(self):
        # Same chunk_id, new pipeline version -> same slot, no duplicate.
        store = ChunkStore()
        store.add_raw(_raw())
        store.add(_record("c1", "old text", verified=True))
        store.add(_record("c1", "new text", verified=True))

        assert len(store.all_records()) == 1
        assert store.record("c1") is not None
        assert store.record("c1").text == "new text"  # type: ignore[union-attr]

    def test_empty_query_does_not_crash(self):
        store = ChunkStore()
        store.add_raw(_raw())
        store.add(_record("c1", "text", verified=True))

        assert len(store.search("   ")) == 1


class TestContentHash:
    def test_identity_is_stable_across_calls(self):
        assert content_hash(b"pdf bytes") == content_hash(b"pdf bytes")

    def test_different_content_different_identity(self):
        assert content_hash(b"a") != content_hash(b"b")
