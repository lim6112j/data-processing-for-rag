# Project Instructions

## What This Is

Grounding-gated ingestion for RAG. Every extracted value is re-derived from its
provenance span mechanically (T1) — no model in T0/T1. A chunk is verified,
quarantined as a specific review question, or recorded as a failure. There is no
fourth path. T2 (OpenRouter LLM) only resolves *structural disagreements between
parsers*; it never sees grounding failures and never verifies anything.

## Tech Stack

- Python >= 3.13, managed with `uv` (uv_build backend)
- FastAPI + Uvicorn (demo server), Pydantic v2, Docling + pdfplumber (parsing)
- pytest (markers `live` and `docling` are excluded by default), ruff

## Commands

- Tests (no network): `uv run pytest`
- Live tests (needs OPENROUTER_API_KEY): `uv run pytest -m live`
- Lint: `uv run ruff check .`
- Demo: `uv run rag-demo` → http://127.0.0.1:8765
- Gold set: `uv run rag-goldset` (add `--corpus ./pdfs` for real PDFs,
  `--score labels.json` for the T2 scorecard)

## Code Style

- Line length 100, target py313; ruff rules E, F, I, UP, B, RUF
- snake_case files; one concept per module (~100–350 lines)
- Frozen dataclasses for records; immutable collections (tuples) in outcomes
- Express failure as explicit status values (`GroundingStatus`, `TierDecision`,
  `AbstainReason`), not exceptions, on the pipeline's happy path
- Docstrings explain invariants and *why*, not visible mechanics
- FastAPI dependency-injection-in-defaults is exempted via per-file ignore in
  `demo/server.py` (B008) — keep that ignore when touching that file

## Testing

- Run: `uv run pytest`
- Convention: `tests/test_<module>.py` mirrors `src/rag_ingest/<module>.py`
- Mark external-API tests `@pytest.mark.live`; Docling model tests
  `@pytest.mark.docling` (both excluded from the default run)
- Build synthetic documents with the helpers in `tests/conftest.py`
- After any change to thresholds, chunking, or the ontology, run
  `uv run rag-goldset` — decision-level report must stay green

## Project Structure

- `src/rag_ingest/pipeline.py` — orchestration (T0→T1→T2→T3 cascade)
- `src/rag_ingest/grounding.py`, `tiering.py`, `scoring.py`, `chunking.py` — gate core
- `src/rag_ingest/adapters/` — docling, pdfplumber, openrouter wrappers
- `src/rag_ingest/adjudication.py`, `circuit.py` — T2 contract + breaker
- `src/rag_ingest/escalation.py` — review queue (T3)
- `src/rag_ingest/demo/` — FastAPI demo (server, session, static UI)
- `goldset/cases.yaml` — decision-level gold set cases

## Supporting Modules (added by codebase onboarding)

- `docmodel.py` — dependency-free element tree + provenance model (adapter boundary)
- `extractors.py`, `candidates.py` — T0 regex extractors; dual-parse disagreement production for T2
- `checks.py` — deterministic semantic checks after the gate: date plausibility,
  arithmetic reconciliation. The corpus is **born-digital documents only** —
  no OCR machinery, no character-substitution matching
- `gate_tier.py`, `store.py`, `metrics.py` — STRICT/STANDARD/LENIENT bars, fail-closed
  per-chunk retrieval policy, per-class escalation rates
- `ingest_runner.py`, `goldset.py`, `ontology.py` — batch runner, gold-set evaluator, schema
- `demo/session.py` — demo review-loop state (largest demo file)

## Conventions

- Commits: conventional commits with scopes (`fix(demo):`, `docs(survey):`)
- Feature branches (`feat/...`) merged to `main` via PR
- Two invariants to never violate:
  1. T2 is never handed a grounding failure — structural questions only
  2. T2's pick only narrows candidates; the chosen reading is re-checked by T1
     before entering retrieval (enforced in `IngestPipeline`)

## Roadmap (remaining work)

- **Contract persistence**: DONE (`shapes_path` on DemoSession — adopt writes,
  startup loads; corrupt file fails closed to builtins)
- **Table → row routing**: DONE (`table_rows.py` — pipe-rendered rows,
  newline-packed cell expansion; FAA TABLE 7-1 routes to 25 rows, one
  signature, fully proposable)
- **LLM proposer layer**: DONE (`llm_propose.py` + session fallback — grammar
  refusal falls back to the model; claims labelled source="llm", evidence
  verified against cited rows, adoption gate unchanged)
- **T2 repurposing**: OpenRouter tier could resolve data-quality claims
  (structural disagreements between parsers) — invariant #1 still holds
- **Real Korean business corpus**: measured multi-doc
  (`corpus/korean/` — 3 monthly inventory PDFs + shipment log, born-digital
  via reportlab). Cross-document row recurrence works (12 rows/3 docs →
  candidate). Full loop closes end-to-end: adopt → next document's rows
  verified mechanically at ingest (`row_verify.py`, `pipeline` registry +
  `row_verifications` in `IngestOutcome`/`IngestView`). Header rows are
  honestly refused when read as data against an enum contract
