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

from collections import Counter
from dataclasses import dataclass

from rag_ingest.ontology import ChunkRecord


@dataclass(frozen=True)
class ShapeCluster:
    """One recurring shape among quarantined chunks.

    ``pages`` is the number of distinct pages the cluster appears on -- a
    shape that recurs across pages is corpus structure; one that appears
    three times on one page is probably one bad table.
    """

    shape: str
    chunks: int
    pages: int
    sample_text: str

    @property
    def is_candidate(self) -> bool:
        """Recurring enough to be worth the owner's attention.

        Two thresholds, both must hold: several chunks (one stray chunk is
        noise, not a shape) and more than one page (a single page's layout
        quirk is a document property, not a corpus shape). The numbers are
        deliberately low: this flag opens a conversation with the owner, it
        does not change the pipeline by itself -- the adoption gate does that.
        """
        return self.chunks >= 3 and self.pages >= 2


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
    samples: dict[str, str] = {}

    for record in quarantined:
        extraction = record.extraction
        if extraction is None or getattr(extraction, "kind", None) != "unclassified":
            continue
        shape = extraction.observed_shape
        by_shape[shape] += 1
        pages.setdefault(shape, set()).add(_first_page(record))
        samples.setdefault(shape, record.text.strip()[:120])

    unclassified_total = sum(by_shape.values())
    clusters = tuple(
        ShapeCluster(
            shape=shape,
            chunks=count,
            pages=len(pages[shape]),
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


__all__ = ["DiscoveryReport", "ShapeCluster", "discover"]
