"""Candidate producer for the demo.

T2 adjudicates disagreements *between parsers*. With only one parser wired there
is no disagreement, so the branch never fires -- correct, and useless for a
walkthrough. This producer manufactures a second reading so the branch is
reachable.

The second reading is a whitespace-collapsed variant of the first. It is
labelled ``alternate`` and marked as scripted, so nobody mistakes it for a real
cross-origin parse result: a genuine disagreement comes from Docling vs
pdfplumber disagreeing about a *region*, not about line breaks.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rag_ingest.adjudication import Candidate


class DoubleReadingProducer(BaseModel):
    """Two readings per chunk: primary text, and a normalized variant."""

    model_config = ConfigDict(frozen=True)

    primary_name: str = "docling"
    alternate_name: str = "pdfplumber"

    def produce(self, chunk) -> list[Candidate]:
        if not chunk.text.strip():
            return []
        return [
            Candidate(
                candidate_id=f"{chunk.chunk_id}:primary",
                produced_by=self.primary_name,
                proposed_text=chunk.text,
                bbox=chunk.bbox,
            ),
            Candidate(
                candidate_id=f"{chunk.chunk_id}:alternate",
                produced_by=self.alternate_name,
                proposed_text=" ".join(chunk.text.split()),
                bbox=chunk.bbox,
                note="scripted second reading (whitespace-collapsed)",
            ),
        ]


__all__ = ["DoubleReadingProducer"]
