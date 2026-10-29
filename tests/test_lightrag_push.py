"""The push script posts one export row per LightRAG insert so the pipeline's
chunk boundaries survive: LightRAG must not re-chunk what the gate already
verified. file_source carries our doc_id so References point back at source
documents; id carries our chunk_id so re-pushes are upserts, not duplicates."""


from rag_ingest.export import ExportRow
from rag_ingest.ontology import GateTier
from rag_ingest.scripts.lightrag_push import push_rows


def _row(chunk_id: str, doc_id: str = "d1") -> ExportRow:
    return ExportRow(chunk_id=chunk_id, doc_id=doc_id, text=f"text-{chunk_id}",
                     gate_tier=GateTier.STRICT)


class _Recorder:
    def __init__(self, failures: set[str] | None = None):
        self.calls: list[dict] = []
        self._failures = failures or set()

    def __call__(self, path: str, payload: dict) -> dict:
        self.calls.append({"path": path, **payload})
        chunk_id = payload["ids"]
        if chunk_id in self._failures:
            return {"status": "error", "message": "boom"}
        return {"status": "success", "track_id": f"t-{chunk_id}"}


class TestPushRows:
    def test_one_insert_per_row_with_document_and_chunk_identity(self):
        poster = _Recorder()
        push_rows([_row("c1"), _row("c2", doc_id="d2")], poster)

        assert len(poster.calls) == 2
        call = poster.calls[0]
        assert call["path"] == "/documents/text"
        assert call["text"] == "text-c1"
        assert call["file_source"] == "d1"
        assert call["ids"] == "c1"

    def test_failures_are_reported_not_swallowed(self):
        poster = _Recorder(failures={"c2"})
        result = push_rows([_row("c1"), _row("c2")], poster)

        assert result.accepted == ("c1",)
        assert result.failed == ("c2",)

    def test_empty_export_pushes_nothing(self):
        poster = _Recorder()
        assert push_rows([], poster).accepted == ()
        assert poster.calls == []


class TestDuplicateContent:
    """Identical prose repeated across monthly documents is one fact, not
    three: LightRAG 409s duplicate content, and the pusher must report that
    as a skip -- visible, distinct from a transport failure, never silent."""

    def test_409_conflict_reports_a_duplicate_skip(self):
        class _Conflictor:
            def __call__(self, path, payload):
                import urllib.error

                raise urllib.error.HTTPError(
                    path, 409, "Conflict", None, None
                )

        result = push_rows([_row("c1")], _Conflictor())
        assert result.duplicates == ("c1",)
        assert result.failed == ()


class TestOneInsertPerChunk:
    """LightRAG keys its document store by file_source, so a second chunk of
    the same document 409s unless the source name is derived per chunk. The
    first chunk keeps the clean document name (that is what References show);
    later chunks carry a deterministic sequential suffix so chunk boundaries
    -- the thing the gate verified -- survive instead of being merged for
    convenience. The suffix keeps the document name leading: LightRAG renders
    the tail of the source as the reference label, so the label must stay
    readable ("d1-2", not a hash fragment)."""

    def test_second_chunk_of_same_document_gets_a_deterministic_source(self):
        poster = _Recorder()
        result = push_rows([_row("c1"), _row("c2")], poster)

        assert result.accepted == ("c1", "c2")
        assert poster.calls[0]["file_source"] == "d1"
        assert poster.calls[1]["file_source"] == "d1-2"
