"""Shape discovery: quarantined chunks form patterns, not just noise.

The gap this closes: when a corpus arrives full of content the ontology does
not know, every chunk is an individual `unclassified` and the owner sees 300
separate questions. But the chunks carry a signal they were not being read
for -- ``observed_shape`` -- and the same shape recurring across many chunks
and pages is a *candidate shape*, not noise. That cluster, not the individual
chunk, is what the owner should be asked about: "your corpus keeps producing
tabular content we cannot verify -- define what a row means".

Deliberately model-free. Clustering is counting with a grouping key; an LLM
classifying shapes would inject the exact non-determinism this pipeline keeps
out of T0/T1, and a mislabelled cluster steers the owner toward a wrong shape
declaration. The model may later *propose* a schema for a cluster (the
propose step, not built yet) -- but which clusters exist is a fact about the
quarantined chunks, counted mechanically.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from rag_ingest.ontology import ChunkRecord


@dataclass(frozen=True)
class ShapeCluster:
    """One recurring shape among quarantined chunks.

    ``pages`` and ``documents`` are the distinct page/document counts the
    cluster appears on. A shape recurring across documents is corpus
    structure; three occurrences on one page of one document is probably one
    bad table. Both count, because either spread is real recurrence: a
    three-page run inside one document and the same table in three separate
    documents are the same signal at corpus scale.
    """

    shape: str
    chunks: int
    pages: int
    documents: int = 1
    sample_text: str = ""

    @property
    def is_candidate(self) -> bool:
        """Recurring enough to be worth the owner's attention.

        Several chunks (one stray chunk is noise) AND more than one origin
        (page or document -- a single page's layout quirk is a document
        property, not a corpus shape). The numbers are deliberately low: this
        flag opens a conversation with the owner, it does not change the
        pipeline by itself -- the adoption gate does that.
        """
        spread = max(self.pages, self.documents)
        return self.chunks >= 3 and spread >= 2


@dataclass(frozen=True)
class DiscoveryReport:
    """The shape clusters in one quarantine outcome."""

    clusters: tuple[ShapeCluster, ...]
    unclassified_total: int

    @property
    def candidates(self) -> tuple[ShapeCluster, ...]:
        return tuple(c for c in self.clusters if c.is_candidate)

    @property
    def known_shapes(self) -> tuple[ShapeCluster, ...]:
        """Clusters whose shape the ontology already handles (e.g. prose)."""
        return tuple(c for c in self.clusters if not c.is_candidate)


def discover(quarantined: tuple[ChunkRecord, ...]) -> DiscoveryReport:
    """Cluster quarantined chunks by their observed shape.

    Only ``Unclassified`` observations count -- a grounding failure or an
    incomplete extraction is an extraction problem for an existing shape,
    not evidence of a missing one. Conflating them would recommend shape
    definitions to fix what is really a parser defect.
    """
    by_shape: Counter[str] = Counter()
    pages: dict[str, set[int]] = {}
    documents: dict[str, set[str]] = {}
    samples: dict[str, str] = {}

    for record in quarantined:
        extraction = record.extraction
        if extraction is None or getattr(extraction, "kind", None) != "unclassified":
            continue
        shape = extraction.observed_shape
        by_shape[shape] += 1
        pages.setdefault(shape, set()).add(_first_page(record))
        documents.setdefault(shape, set()).add(record.doc_id)
        samples.setdefault(shape, record.text.strip()[:120])

    unclassified_total = sum(by_shape.values())
    clusters = tuple(
        ShapeCluster(
            shape=shape,
            chunks=count,
            pages=len(pages[shape]),
            documents=len(documents[shape]),
            sample_text=samples.get(shape, ""),
        )
        for shape, count in by_shape.most_common()
    )
    return DiscoveryReport(clusters=clusters, unclassified_total=unclassified_total)


def _first_page(record: ChunkRecord) -> int:
    extraction = record.extraction
    if extraction is not None and getattr(extraction, "provenance", None) is not None:
        return extraction.provenance.page
    return 1


__all__ = ["DiscoveryReport", "ShapeCluster", "discover", "discover_from_rows", "row_signatures"]


def row_signatures(rows: tuple) -> dict[int, str]:
    """Cluster keys for every row: the column-major role sequence.

    Column roles are computed over the whole row set, not per row -- '8-36'
    and '12-15' have the same cell shape, and only the column's dominant
    kind tells thread from range. Exposed so the session's proposer keys
    rows with the exact grammar discovery clustered them by.
    """
    from rag_ingest.propose import classify_cell

    column_kinds: dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        for col, cell in enumerate(row.cells):
            column_kinds[col][classify_cell(cell)] += 1
    roles = {
        col: (kinds.most_common(1)[0][0] if kinds else "empty")
        for col, kinds in column_kinds.items()
    }
    return {
        id(row): "row:" + ",".join(roles.get(col, "empty") for col in range(len(row.cells)))
        for row in rows
    }


def discover_from_rows(rows: tuple) -> DiscoveryReport:
    """Cluster row observations by their cell-kind signature.

    A row's signature is the per-cell kind sequence (thread / range / number /
    text), classified by the same grammar the proposer uses -- so what
    discovery counts as recurring is exactly what the proposer can read.
    This keeps rows and blob-chunks in separate cluster spaces: a row shape
    and a prose shape are different conversations with the owner.
    """
    signatures = row_signatures(rows)
    by_signature: Counter[str] = Counter()
    pages: dict[str, set[int]] = {}
    documents: dict[str, set[str]] = {}
    samples: dict[str, str] = {}

    # Column-major keys: see row_signatures. '8-36' and '12-15' have the
    # same cell shape; only the column's dominant kind tells thread from
    # range, so per-cell classification would split one column's siblings
    # into two clusters.
    for row in rows:
        signature = signatures[id(row)]
        by_signature[signature] += 1
        pages.setdefault(signature, set()).add(row.page)
        documents.setdefault(signature, set()).add(row.doc_id)
        samples.setdefault(signature, row.row_text.strip()[:120])

    clusters = tuple(
        ShapeCluster(
            shape=signature,
            chunks=count,
            pages=len(pages[signature]),
            documents=len(documents[signature]),
            sample_text=samples.get(signature, ""),
        )
        for signature, count in by_signature.most_common()
    )
    return DiscoveryReport(clusters=clusters, unclassified_total=sum(by_signature.values()))
