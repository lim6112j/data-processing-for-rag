"""OpenRouter implementation of T2.

Wraps the OpenAI-compatible chat-completions endpoint that OpenRouter exposes.
The credential is resolved by the shared ``openrouter_client`` resolver -- the
OpenAI SDK does not read ``OPENROUTER_API_KEY`` on its own -- and is never
logged, stored, or written to the label set.

Three things this adapter does deliberately:

1. **No ``value`` field anywhere.** The output schema is the ``AdjudicationDecision``
   union, so the model can pick, abstain, or escalate -- never emit a field value
   that bypasses the provenance gate. This is the §1 contract enforced at the
   wire level, not merely by convention.
2. **It refuses to call the API without visual evidence.** If the question
   carries no bbox, or the page has no rendered image, it returns an abstention
   instead of asking a text-only question. Text cannot settle a
   character-identity dispute, and a confident text-only answer is a guess
   dressed as an adjudication.
3. **Transport failures become abstentions, not exceptions.** A 429, a timeout,
   or a malformed response must not take down ingestion; it must degrade to the
   human queue. That is the whole point of having one.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from rag_ingest.adapters.openrouter_client import resolve_client
from rag_ingest.adjudication import AdjudicationDecision, AdjudicationQuestion
from rag_ingest.ontology import AbstainReason

#: Default T2 model. Adjudication is the last automatic gate and runs on a small
#: fraction of traffic, so capability is worth more than price here. Override
#: with ``RAG_T2_MODEL`` to trade cost against quality.
DEFAULT_MODEL = "anthropic/claude-opus-5-5"

#: Ceiling on reasoning + answer tokens. Adjudications are short.
DEFAULT_MAX_TOKENS = 2000

#: Hard timeout. Ingestion is batch, but a hung call still blocks a worker.
DEFAULT_TIMEOUT_SECONDS = 90

SYSTEM_PROMPT = """\
You are a document-structure adjudicator. You receive two or more competing \
readings of the same region of a page, plus an image crop of that region.

Decide which reading is correct, judging ONLY from the image. Rules:

- If the image clearly supports one reading, pick it.
- If the image does not settle the question, abstain. Do NOT guess.
- If the image shows that no reading is correct, escalate to a human with a \
specific question.
- Never produce a field value, a number, or a corrected transcription. You are \
choosing between readings, not extracting.

An abstention is a correct answer. A confident wrong choice is not."""

#: JSON schema for the decision. Mirrors ``AdjudicationDecision``; kept as a
#: literal because it is sent over the wire and must be self-contained.
DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind"],
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["pick", "abstain", "escalate"],
            "description": "Which of the three legal outcomes applies.",
        },
        "choice_index": {
            "type": "integer",
            "minimum": 0,
            "description": "0-based index of the chosen candidate. Only for kind=pick.",
        },
        "reason": {
            "type": "string",
            "enum": [r.value for r in AbstainReason],
            "description": "Why no choice could be made. Only for kind=abstain.",
        },
        "owner_question": {
            "type": "string",
            "description": (
                "A specific, closed question for the document owner. Only for kind=escalate."
            ),
        },
        "rationale": {
            "type": "string",
            "description": "One sentence on what in the image supported the decision.",
        },
    },
}


class OpenRouterAdjudicator:
    """T2 backed by an OpenRouter model.

    Construct lazily: the SDK import is deferred so the core pipeline keeps
    working when ``openai`` is not installed.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        base_url: str = "https://openrouter.ai/api/v1",
        client: Any | None = None,
    ) -> None:
        self._model = model or os.environ.get("RAG_T2_MODEL", DEFAULT_MODEL)
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client = client
        # ``api_key`` is accepted for tests and injected clients; in normal use
        # the SDK resolves OPENROUTER_API_KEY from the environment.
        self._api_key = api_key
        self._base_url = base_url

    # -- client -----------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = resolve_client(
                None,
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                feature="OpenRouter adjudication",
            )
        return self._client

    # -- prompt -----------------------------------------------------------

    def _build_messages(
        self, question: AdjudicationQuestion, image_png: bytes
    ) -> list[dict[str, Any]]:
        readings = "\n".join(
            f"[{index}] {candidate.produced_by}: {candidate.proposed_text!r}"
            for index, candidate in enumerate(question.candidates)
        )
        text = (
            f"Question: {question.prompt}\n"
            f"Disagreement kind: {question.kind.value}\n"
            f"Competing readings:\n{readings}\n"
        )
        if question.context_text:
            text += f"\nSurrounding text (context only):\n{question.context_text[:1500]}\n"

        encoded = base64.b64encode(image_png).decode("ascii")
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    # Text before image: OpenRouter recommends this ordering.
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            },
        ]

    # -- decision ---------------------------------------------------------

    def adjudicate(self, question: AdjudicationQuestion) -> AdjudicationDecision:
        return self.adjudicate_with_evidence(question, image_png=None)

    def adjudicate_with_evidence(
        self, question: AdjudicationQuestion, *, image_png: bytes | None
    ) -> AdjudicationDecision:
        """Adjudicate, given the cropped evidence image.

        No image (or no bbox) yields an abstention: asking a text-only question
        would produce a plausible answer with no evidential basis.
        """
        if not question.has_visual_evidence:
            return AdjudicationDecision(
                kind="abstain",
                reason=AbstainReason.AMBIGUOUS,
                rationale="No bbox on the provenance ref; T2 cannot see the region.",
            )
        if image_png is None:
            return AdjudicationDecision(
                kind="abstain",
                reason=AbstainReason.AMBIGUOUS,
                rationale="Page image unavailable; T2 cannot judge from text alone.",
            )

        try:
            response = self._get_client().chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=self._build_messages(question, image_png),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "adjudication_decision",
                        "strict": True,
                        "schema": DECISION_SCHEMA,
                    },
                },
            )
        except Exception as exc:
            return AdjudicationDecision(
                kind="abstain",
                reason=AbstainReason.AMBIGUOUS,
                rationale=f"T2 transport failure: {type(exc).__name__}",
            )

        return self._parse(response, question)


    def _parse(
        self, response: Any, question: AdjudicationQuestion
    ) -> AdjudicationDecision:
        try:
            content = response.choices[0].message.content
            payload = json.loads(content)
            decision = AdjudicationDecision.model_validate(payload)
        except Exception as exc:
            return AdjudicationDecision(
                kind="abstain",
                reason=AbstainReason.AMBIGUOUS,
                rationale=f"T2 returned unreadable output: {type(exc).__name__}",
            )

        if not decision.is_coherent(len(question.candidates)):
            return AdjudicationDecision(
                kind="abstain",
                reason=AbstainReason.AMBIGUOUS,
                rationale=f"T2 returned an incoherent decision: {decision.kind}",
            )
        return decision


__all__ = ["DECISION_SCHEMA", "DEFAULT_MODEL", "OpenRouterAdjudicator"]
