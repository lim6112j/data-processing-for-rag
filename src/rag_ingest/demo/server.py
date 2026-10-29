"""Demo server: upload a PDF, watch it go through the cascade.

Route shape follows the thing being demonstrated, not REST convention:

  POST /api/ingest            upload -> a verdict per chunk, a tier per element
  GET  /api/page/{doc}/{n}    the page image the overlay is drawn on
  GET  /api/overlay/{doc}     chunk boxes with verdicts
  GET  /api/review            open review items, highest priority first
  POST /api/review/{id}       record a correction, return the new T2 scorecard
  GET  /api/scorecard         T2 precision / coverage / harm + breaker state

Run:  uv run rag-demo      then open http://127.0.0.1:8000
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from rag_ingest.demo.session import DemoSession

app = FastAPI(title="rag-ingest demo", version="0.1.0")

#: One session for the whole process. A demo, not a service: concurrent users
#: would share a review queue, which is either fine for a walkthrough or the
#: exact thing you would change first in a real deployment.
# Adoptions persist across restarts: a contract the owner proved through the
# gate is corpus knowledge, not session state. Losing it on a server bounce
# would silently re-quarantine every table the contract had already answered.
# Corrupt or missing files fail closed to builtins (DemoSession's contract).
_SESSION_DIR = Path(tempfile.gettempdir()) / "rag-demo"
_session = DemoSession(shapes_path=_SESSION_DIR / "shapes.json")


class ReviewRequest(BaseModel):
    """A correction. The item id travels here, not in the URL.

    Chunk ids contain ``:`` and ``/`` (e.g. ``doc:table:#/tables/0``); a path
    parameter cannot carry those reliably, which made every Submit return 404.
    """

    item_id: str = Field(min_length=1)
    after: str = Field(min_length=1)
    choice: str | None = None
    role: str = "owner"


def _configure_t2(enabled: bool) -> None:
    """Attach or detach a scripted T2.

    Off by default, because with no second parse configured there is genuine
    disagreement to adjudicate and a T2 that fires anyway would be theatre.
    When on, the producer supplies two readings so the branch is reachable.
    """
    from rag_ingest.demo.fake_adjudicator import ScriptedAdjudicator
    from rag_ingest.demo.producers import DoubleReadingProducer

    if not enabled:
        # Clear both: an adjudicator with no producer is inert, and a producer
        # with no adjudicator would raise into an abstention. They are a pair.
        _session.adjudicator = None
        _session.producer = None
        return

    _session.adjudicator = ScriptedAdjudicator(prefer="pdfplumber")
    _session.producer = DoubleReadingProducer()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def config() -> dict:
    return {
        "t2_enabled": _session.adjudicator is not None,
        "force_ambiguous": _session.force_structure_ambiguous,
    }


@app.post("/api/config")
def set_config(t2: bool = False, force_ambiguous: bool = False) -> dict:
    _configure_t2(t2)
    _session.force_structure_ambiguous = force_ambiguous
    return {"t2_enabled": t2, "force_ambiguous": force_ambiguous}


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...), parser: str = "auto") -> dict:
    """Parse an uploaded PDF and run it through the cascade."""
    payload = await file.read()
    if not payload:
        raise HTTPException(400, "empty upload")

    doc_id = Path(file.filename or "document").stem
    suffix = Path(file.filename or "x.pdf").suffix or ".pdf"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)

    try:
        from rag_ingest.demo.parsing import parse_pdf

        raw, elements, resolved_parser = parse_pdf(temp_path, doc_id=doc_id, parser=parser)
    except Exception as exc:  # a bad upload must not 500 the demo
        raise HTTPException(422, f"{type(exc).__name__}: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)

    view = _session.ingest(
        doc_id=doc_id,
        filename=file.filename or doc_id,
        parser=resolved_parser,
        raw=raw,
        elements=elements,
    )
    return view.model_dump(mode="json")


@app.get("/api/page/{doc_id}/{page_no}")
def page_image(doc_id: str, page_no: int) -> Response:
    image = _session.page_image(doc_id, page_no)
    if image is None:
        raise HTTPException(404, "no rendered image for that page")
    return Response(content=image, media_type="image/png")


@app.get("/api/view/{doc_id}")
def view(doc_id: str) -> dict:
    """The stored ingest view, with review items re-read from the live queue.

    Re-reading matters: the stored view is a snapshot, so an item resolved since
    ingestion would still show as pending and the counters would drift.
    """
    stored = _session.views.get(doc_id)
    if stored is None:
        raise HTTPException(404, "unknown document")
    payload = stored.model_dump(mode="json")
    closed = {item.chunk_id for item in _session.queue.items_by_chunk().values() if item.resolved}
    payload["review_items"] = [
        {**item, "resolved": item["chunk_id"] in closed} for item in payload["review_items"]
    ]
    return payload


@app.get("/api/overlay/{doc_id}")
def overlay(doc_id: str) -> dict:
    return _session.overlay(doc_id)


@app.get("/api/review")
def review_items() -> dict:
    return {"items": [item.model_dump(mode="json") for item in _session.fresh_review_items()]}


@app.post("/api/review")
def resolve(body: ReviewRequest) -> dict:
    # The parameter must not be named ``request``: FastAPI would inject the
    # Starlette Request object there instead of parsing the model, and every
    # call would fail with a 422 asking for a ``request`` query parameter.
    from rag_ingest.escalation import ReviewerRole

    try:
        role = ReviewerRole(body.role)
    except ValueError as exc:
        raise HTTPException(400, f"unknown role {body.role!r}") from exc

    try:
        card = _session.resolve(
            body.item_id,
            after=body.after,
            choice=body.choice,
            reviewer_role=role,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    return _scorecard_payload(card)


class AdoptRequest(BaseModel):
    """An owner-approved shape contract, arriving as data, with its cases."""

    contract: dict
    cases: list[dict] = Field(default_factory=list)


class VerifyShapeRequest(BaseModel):
    """Values for a new chunk, checked against an adopted shape."""

    shape: str = Field(min_length=1)
    text: str = Field(min_length=1)
    values: dict[str, str]


@app.post("/api/approve/{item_id:path}")
def approve_item(item_id: str) -> dict:
    """Owner approval of a review item: the loop closer.

    The gate still re-checks the chunk against its source -- approval is
    about meaning, citation is the machine's. A refused approval stays
    quarantined and is reported as such; pretending it succeeded would turn
    the owner's sign-off into a new fabrication channel."""
    try:
        result = _session.approve_chunk(item_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "promoted": result.refusal is None,
        "refusal": (
            {"reason": result.refusal.reason, "detail": result.refusal.detail}
            if result.refusal is not None
            else None
        ),
    }


@app.get("/api/export")
def export_verified_jsonl() -> Response:
    """The session's verified chunks as JSONL for a downstream retrieval store.

    A projection of the gate, not a choice: quarantined chunks cannot appear
    here no matter how the request is shaped, because export reads only
    ``verified_chunks`` from the pipeline's outcomes. The text/plain response
    is one JSON object per line -- the format the push script round-trips."""
    body = "\n".join(row.to_jsonl() for row in _session.export_rows())
    return Response(content=body + ("\n" if body else ""), media_type="text/plain")


@app.get("/api/discovery")
def discovery() -> dict | None:
    """Shape clusters among quarantined chunks, across ingested documents."""
    report = _session.discovery_report()
    if report is None:
        return None
    return {
        "unclassified_total": report.unclassified_total,
        "clusters": [
            {
                "shape": c.shape,
                "chunks": c.chunks,
                "pages": c.pages,
                "documents": c.documents,
                "sample_text": c.sample_text,
                "is_candidate": c.is_candidate,
            }
            for c in report.clusters
        ],
    }


@app.post("/api/propose")
def propose_schema(body: dict) -> dict:
    """A schema claim for one shape cluster, with per-field evidence.

    404 on an unknown or absent cluster: the UI must show "cannot propose"
    rather than a guessed schema.
    """
    shape = (body or {}).get("shape", "")
    report = _session.discovery_report()
    known = {c.shape for c in report.clusters} if report else set()
    if shape not in known:
        raise HTTPException(404, f"no quarantined cluster for shape {shape!r}")
    proposal = _session.propose_for(shape)
    if proposal is None:
        raise HTTPException(404, f"cannot propose for shape {shape!r}")
    return {
        "shape_name": proposal.shape_name,
        "tier_declared": proposal.tier_declared,
        "source": proposal.source,
        "parseable": proposal.parseable,
        "note": proposal.note,
        "fields": [f.model_dump(mode="json") for f in proposal.fields],
        "evidence": [
            {"sample_value": e.sample_value, "row_index": e.row_index}
            for e in proposal.evidence
        ],
    }


@app.post("/api/adopt")
def adopt(body: AdoptRequest) -> dict:
    """Run the adoption gate, then register. Effective immediately, no restart.

    422 on an invalid declaration, 409 on a duplicate, and -- the important
    one -- 409-with-failures when the gold cases falsify the contract: the
    response carries every failing case so the UI can show what broke and
    the registry stays untouched.
    """
    from rag_ingest.shapes import ShapeContract

    try:
        contract = ShapeContract.from_definition(body.contract)
    except Exception as exc:
        raise HTTPException(422, f"invalid contract: {exc}") from exc

    cases = body.cases or [
        # No cases supplied: derive the honest minimum -- one positive case
        # per required field pair is not derivable generically, so an
        # adoption without cases is refused rather than rubber-stamped.
    ]
    if not cases:
        raise HTTPException(
            422,
            "adoption requires gold cases: declare at least one case "
            "expecting verified and one expecting a refusal of a value the "
            "contract must catch",
        )

    try:
        from rag_ingest.shapes import GoldCaseSpec

        spec = GoldCaseSpec.from_claims(contract.name, cases)
    except Exception as exc:
        raise HTTPException(422, f"invalid cases: {exc}") from exc

    result = _session.adopt_with_gate(contract, spec)
    if not result.registered:
        raise HTTPException(
            409,
            detail={
                "message": "adoption blocked: the contract failed its cases",
                "failed_cases": [f.model_dump(mode="json") for f in result.failed_cases],
            },
        )
    return {"registered": True, "name": contract.name, "tier": contract.tier.value}


@app.post("/api/verify-shape")
def verify_shape(body: VerifyShapeRequest) -> dict:
    """Check values against an adopted shape through the standard gate."""
    try:
        result = _session.check_against_shape(body.shape, body.text, body.values)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {
        "status": result.status.value,
        "verified": result.verified,
        "issues": [
            {"field": i.field, "status": i.status.value, "message": i.message}
            for i in result.issues
        ],
    }


@app.get("/api/labels")
def labels() -> dict:
    """Every correction recorded, in the form the eval loop consumes.

    The scorecard says how many corrections exist; this says *what they are*.
    Without it "recorded as labels" is an unverifiable claim -- the reviewer has
    to take the server's word for it.
    """
    payload = _session.queue.export()
    return {
        "corrections": payload["corrections"],
        # Download target for the same data, so it can be fed to
        # `rag-goldset --score`.
        "download": "/api/labels.json",
    }


@app.get("/api/labels.json")
def labels_download() -> Response:
    import json

    payload = _session.queue.export()
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    return Response(
        content=body,
        media_type="application/json",
        headers={"content-disposition": 'attachment; filename="labels.json"'},
    )


@app.get("/api/scorecard")
def scorecard() -> dict:
    return _scorecard_payload(_session.scorecard())


def _scorecard_payload(card) -> dict:
    breaker = _session.breaker_status()
    return {
        "scored": card.total_scored,
        "excluded": card.excluded,
        # Why the numbers above may not apply. Empty when they do.
        "explain": card.explain(),
        "exclusions": {
            reason.value: count for reason, count in card.exclusions.items()
        },
        "picks": card.picks,
        "abstentions": card.abstentions,
        "precision": card.precision,
        "coverage": card.coverage,
        "harm_rate": card.harm_rate,
        "verdict": card.verdict(),
        "breaker": None
        if breaker is None
        else {
            "state": breaker.state.value,
            "trip_reason": breaker.trip_reason.value,
            "labels_seen": breaker.labels_seen,
        },
    }


def main() -> None:  # pragma: no cover - process entry point
    import os

    import uvicorn

    # 8000 is commonly taken by other dev servers; allow an override.
    port = int(os.environ.get("RAG_DEMO_PORT", "8765"))
    uvicorn.run(app, host="127.0.0.1", port=port)


__all__ = ["app", "main"]
