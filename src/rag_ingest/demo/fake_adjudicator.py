"""A scripted T2 for offline demos.

Real adjudication calls OpenRouter, costs money, and produces different answers
on different runs -- none of which is what a walkthrough needs. This backend
returns a fixed decision so the *shape* of the T2 path is demonstrable without a
key, and it is deliberately malleable so the eval loop can be shown.

It is a stand-in, not a model. It makes no judgement at all: it picks the
candidate whose text contains the token it was constructed with, and abstains
otherwise. That is enough to exercise the branch, and weak enough that nobody
mistakes it for evidence that T2 works.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rag_ingest.adjudication import AdjudicationDecision, AdjudicationQuestion
from rag_ingest.ontology import AbstainReason


class ScriptedAdjudicator(BaseModel):
    """Picks the candidate containing ``prefer``; abstains when none does."""

    model_config = ConfigDict(frozen=True)

    prefer: str = "pdfplumber"
    #: When True, answers abstain regardless -- demonstrates the safe path.
    always_abstain: bool = False

    def adjudicate(self, question: AdjudicationQuestion) -> AdjudicationDecision:
        if self.always_abstain:
            return AdjudicationDecision(
                kind="abstain",
                reason=AbstainReason.AMBIGUOUS,
                rationale="scripted abstention",
            )

        for index, candidate in enumerate(question.candidates):
            if self.prefer in candidate.produced_by:
                return AdjudicationDecision(
                    kind="pick",
                    choice_index=index,
                    rationale=f"scripted: preferring {candidate.produced_by}",
                )

        return AdjudicationDecision(
            kind="abstain",
            reason=AbstainReason.AMBIGUOUS,
            rationale=f"scripted: no candidate from {self.prefer!r}",
        )

    def adjudicate_with_evidence(
        self, question: AdjudicationQuestion, *, image_png: bytes | None
    ) -> AdjudicationDecision:
        # The scripted backend needs no image; it delegates so the pipeline's
        # evidence-aware call path is still exercised.
        return self.adjudicate(question)


__all__ = ["ScriptedAdjudicator"]
