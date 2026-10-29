"""Push exported verified chunks into a LightRAG server.

One insert per export row: the chunk boundaries the gate verified are the
boundaries retrieval should reason over, so LightRAG must not re-chunk them.
``file_source`` carries the pipeline's ``doc_id`` so LightRAG's References
point back at source documents, and ``ids`` carries ``chunk_id`` so a re-push
upserts instead of duplicating. The poster is injected so tests never touch
the network; the CLI wires it to ``urllib`` against the running server."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from rag_ingest.export import EXPORT_FORMAT, ExportRow

#: Insert endpoint of the LightRAG API (version 0311 line).
INSERT_PATH = "/documents/text"
INSERT_TIMEOUT = 30


def push_rows(
    rows: list[ExportRow] | tuple[ExportRow, ...],
    poster: Callable[[str, dict], dict],
) -> PushResult:
    """Post each row as one insert, collecting accepted and failed chunk ids.

    A transport failure is a failed chunk, not a crashed run: a partial push
    is a reportable state, and the caller decides whether to retry. The
    result is ordered like the input so a diff against a previous run reads
    top to bottom."""
    accepted: list[str] = []
    failed: list[str] = []
    duplicates: list[str] = []
    seen_sources: dict[str, int] = {}
    for row in rows:
        file_source = row.doc_id
        ordinal = seen_sources.get(file_source, 0) + 1
        seen_sources[file_source] = ordinal
        if ordinal > 1:
            # LightRAG keys its document store by file_source and refuses a
            # repeat (409). Rather than merge chunks -- which surrenders the
            # boundaries the gate verified -- later chunks of the same
            # document carry a sequential suffix. It keeps the document name
            # leading because LightRAG renders the source's tail as the
            # reference label, and the label must stay readable.
            file_source = f"{row.doc_id}-{ordinal}"
        payload = {
            "text": row.text,
            "file_source": file_source,
            "ids": row.chunk_id,
        }
        try:
            response = poster(INSERT_PATH, payload)
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                # Duplicate content: the server already holds this text under
                # a different id. One fact repeated across documents is one
                # fact -- a visible skip, not a failure to hide.
                duplicates.append(row.chunk_id)
                continue
            failed.append(row.chunk_id)
            print(f"  FAIL {row.chunk_id}: HTTP {exc.code}", file=sys.stderr)
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failed.append(row.chunk_id)
            print(f"  FAIL {row.chunk_id}: transport error: {exc}", file=sys.stderr)
            continue
        if isinstance(response, dict) and response.get("status") == "success":
            accepted.append(row.chunk_id)
        else:
            failed.append(row.chunk_id)
            message = (
                response.get("message", "unknown error")
                if isinstance(response, dict)
                else response
            )
            print(f"  FAIL {row.chunk_id}: {message}", file=sys.stderr)
    return PushResult(
        accepted=tuple(accepted), failed=tuple(failed), duplicates=tuple(duplicates)
    )


class PushResult:
    """Which chunks the server accepted, and which it refused or dropped."""

    __module__ = "rag_ingest.scripts.lightrag_push"

    def __init__(
        self,
        accepted: tuple[str, ...],
        failed: tuple[str, ...],
        duplicates: tuple[str, ...] = (),
    ) -> None:
        self.accepted = accepted
        self.failed = failed
        self.duplicates = duplicates


def http_poster(base_url: str) -> Callable[[str, dict], dict]:
    """Build a poster against a running LightRAG server (e.g. via SSH tunnel)."""

    def post(path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            base_url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=INSERT_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))

    return post


def load_rows(path: str) -> tuple[ExportRow, ...]:
    """Read an export JSONL file back into rows, rejecting foreign formats."""
    rows: list[ExportRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("export_format") != EXPORT_FORMAT:
            raise ValueError(
                f"unexpected export_format {record.get('export_format')!r}; "
                f"expected {EXPORT_FORMAT!r}"
            )
        rows.append(
            ExportRow(
                chunk_id=record["chunk_id"],
                doc_id=record["doc_id"],
                text=record["text"],
                gate_tier=record["gate_tier"],
            )
        )
    return tuple(rows)


def main() -> None:  # pragma: no cover - process entry point
    if len(sys.argv) != 3:
        print("usage: lightrag-push <export.jsonl> <base_url>", file=sys.stderr)
        print("  e.g. lightrag-push verified.jsonl http://127.0.0.1:9621",
              file=sys.stderr)
        raise SystemExit(2)
    rows = load_rows(sys.argv[1])
    result = push_rows(rows, http_poster(sys.argv[2]))
    print(
        f"pushed: {len(result.accepted)} accepted, "
        f"{len(result.duplicates)} duplicates, {len(result.failed)} failed"
    )
    raise SystemExit(1 if result.failed else 0)


if __name__ == "__main__":
    main()

__all__ = ["INSERT_PATH", "PushResult", "http_poster", "load_rows", "push_rows"]
