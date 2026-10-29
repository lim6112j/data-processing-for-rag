# rag-ingest

**Status:** working prototype. 493 tests, runs real PDFs end-to-end
through Docling + pdfplumber with OpenRouter as the adjudication backend, ships
an interactive demo server, **learns table shapes** (discovery → proposal →
adoption → row verification), and exports verified chunks into a downstream
LightRAG graph-RAG store.

## Summary

Ingestion for RAG that refuses to guess. The output of a document is one of three
things: **verified** chunks that enter retrieval, **quarantined** chunks that
become a specific question to the document owner, or a **recorded failure**. There
is no fourth path where a plausible-looking value slips through. And the loop
closes: when the owner answers a quarantine question, the answer goes back
through the gate before the chunk enters retrieval — approval is about meaning,
citation is the machine's.

The problem it exists for: most pipelines gate on "does the extracted output
validate against a schema?" When the extractor is a constrained-decoding LLM, that
question is answered by construction — the decoder masks invalid tokens, so it is
impossible to emit a schema violation. A schema-valid result therefore carries
**zero** evidence of correctness, and an invented total (`1334.56` where the
document says `1234.56`) passes straight into retrieval with a citation attached.

The fix is to gate on **grounding instead of shape**. Every extracted value cannot
be represented without a provenance span pointing at the raw source; the gate
re-derives the value from that span and compares. No model participates in that
comparison. Whatever fails is quarantined at *chunk* granularity, and the
question it becomes is specific ("column 3 is unlabelled — what does it mean?")
rather than "please re-upload".

A model appears in exactly two places, both narrow: **T2** adjudicates structural
disagreements *between parsers* (never extraction, and never a grounding failure),
and its picks are scored against owner corrections so it can be switched off when
it stops being right.

### What runs today

| Layer | What it is |
|---|---|
| T0 | Deterministic parsers (Docling, pdfplumber) + regex extraction, emitting confidence and provenance |
| T1 | Mechanical grounding validator — re-derivation, required-evidence threshold, no model |
| T2 | OpenRouter adjudicator, consulted only on parser disagreement, guarded by a circuit breaker |
| T3 | Chunk-scoped review queue with owner-correction labels |
| Shape learning | Discovery clusters recurring table shapes → grammar (then LLM, as labelled fallback) proposes a contract → the owner adopts it through adversarial gold cases → the contract verifies rows in later documents and promotes fully-grounded table chunks out of quarantine |
| Export | `GET /api/export` projects verified chunks (and owner-approval history) to JSONL; `lightrag_push` injects them chunk-by-chunk into a LightRAG server so provenance survives |

T1 applies one of three bars, chosen from the ontology type the extractor claimed
and never from a model:

| Gate tier | Bar | Applies to |
|---|---|---|
| `STRICT` | grounding + required evidence | `InvoiceHeader` |
| `STANDARD` | grounding + required evidence | `ReportSection` |
| `LENIENT` | grounding only — the text is verbatim at the span it cites | `ProseKnowledge` |

`STRICT` and `STANDARD` currently apply the same checks because the arithmetic and
range checks that are meant to separate them are not built yet. `LENIENT` verifies
**fidelity of quotation, not correctness**: a pass means "this document says this at
this place". Every record stamps its tier, so the two claims are never conflated.
A LENIENT bar on a typed record raises rather than quietly admitting it.

The corpus is born-digital documents only, so there is no OCR stage and no
damage-repair machinery: the text layer is the source, and a value the document
does not contain is a mismatch, not something to read through.

```bash
uv run pytest                          # 493 tests, no network
uv run rag-demo                        # interactive UI on :8765
uv run rag-goldset                     # decision-level gold set (6/6)
uv run rag-goldset --corpus ./pdfs     # real PDFs, per-document accounting
uv run rag-goldset --score labels.json # is T2 still trustworthy?
```

### The demo

`uv run rag-demo` serves a UI at `http://127.0.0.1:8765` (override with
`RAG_DEMO_PORT`). Drop a PDF and it shows the verdict rather than the result:
page image with **green boxes** on regions that entered retrieval and **red
boxes** on regions withheld, each chunk's tier route with its scores, and the
review queue on the right.

The withheld box lands exactly on the chunk whose total the document does not
support — you can watch the gate refuse the invented value.

```bash
uv run rag-demo
# then drag an invoice PDF onto the page
```

Two things in the demo deserve honest labelling:

- **The "scripted T2" switch is a demo control, not a simulation of a defect.**
  Real corpora rarely reach T2: pdfplumber reports no scores and Docling's
  layout scores on readable pages sit above the threshold, so
  `STRUCTURE_UNKNOWN` does not fire. The switch lowers the layout score so the
  branch is reachable, and the scripted adjudicator picks by name rather than
  judging — it demonstrates the *shape* of the path, not that T2 works.
- **The review dropdown has no preselected default.** Pre-selecting T2's own
  pick would turn "confirm" into a rubber stamp and inflate precision with
  agreement the reviewer never actually exercised.

The UI itself was verified in a browser, not just via `TestClient`; the
screenshots below are from that run.

| Case | Result |
|---|---|
| Readable invoice | 100% verified, green box, empty queue |
| Invented total | 0% verified, red box, queued as `incomplete_extraction` |
| T2 enabled | two candidates shown, pick recorded, **still withheld** until confirmed |
| After confirming | precision 100%, breaker `closed`, queue 2 → 1 |

`tests/test_demo.py` covers the parts that break silently: routing, item-id
handling, and the review loop's effect on the scorecard.

### Evidence base

The design is informed by a literature survey of LLM-based data cleaning:
[**survey.md**](survey.md). Three findings there changed decisions in this repo:

- **Deterministic profiling beat LLM detection** (F1 0.561 vs 0.421) and LLM
  *repair* was near-worthless (0.028) — so the model was kept out of T0/T1.
- **Human edit systems run at a 20–30% flag hit rate** (Statistics Canada) — 70–80%
  of flags being false alarms is the industry baseline, not a failure.
- **Span grounding is not solved** (best word-level F1 46.4%) — requiring a
  provenance span is necessary, not sufficient.

The survey also records where the evidence is thin: no paper runs the direct
head-to-head of "LLM proposes claim + deterministic verifier" against "LLM
rewrites directly", and no benchmark scores a finding together with its evidence
span. The design is supported element-by-element, not as a ratified pattern.

Its **appendix** collects the design conclusions drawn from it, including the
unbuilt ideas (strict vs lenient validation tiers for DB-bound vs knowledge
content) and the one measurement that has not been run: this pipeline has never
processed a real document corpus. The next step recorded there is
`rag-goldset --corpus`, and the number to read first is the `unclassified`
share.

### What it does not do yet

- **Ontology v1 covers two built-in shapes** (invoice header, report section), but
  table shapes are no longer fixed: discovery + adoption grows contracts at
  runtime, gated by the same adversarial cases as everything else. Non-table
  shapes still fall to `unclassified`, which is the intended signal.
- **`TemplateExtractor` is one invoice template**, not a general extractor.
- **T2 is rarely exercised** on clean corpora — no structural damage means no
  disagreement, which is the honest answer but means its precision is unmeasured
  outside the example label set.
- **No built-in vector store.** The verified-chunk policy is implemented and
  proven end-to-end against an external LightRAG server (`/api/export` +
  `lightrag_push`); embeddings and graph construction are deliberately left to
  the downstream store, which already does them well. The demo holds one
  in-memory session, though adopted shape contracts persist across restarts.

Details for each of these are in *Known limitations* and *Deliberate non-goals*
below.

---

The short version of the critique this implements:

> Not "ontology + a fast model gates ingestion," but: **deterministic extraction
> emits confidence and provenance → a mechanical grounding validator (not a
> schema check) decides → only disagreements and low-confidence cases reach a
> model, as adjudicator → ungrounded content is quarantined at chunk level and
> becomes a specific question to the doc owner, whose answer is captured as a
> label.**

## Why the obvious design does not work

A pipeline that routes documents through an ontology-aligned model and ingests
whatever "validates against the schema" is gated on nothing. Under constrained
decoding the decoder masks invalid tokens during generation, so schema
conformance is guaranteed by construction and carries **zero** evidence of
correctness.

Concretely: the extractor emits `1334.56`. The schema wants a decimal. The
decoder cannot emit invalid JSON, so a schema-valid but invented total sails
through. Validation passes, the chunk enters retrieval, and the system returns a
confidently wrong number with a citation. That is worse than quarantining it, and
it is the default outcome of a shape-only gate.

`tests/test_grounding.py::TestCentralClaim::test_schema_valid_but_ungrounded_value_is_rejected`
is that case, executable.

## The fix, and the centre of this repo

Every extracted value is wrapped so it **cannot be represented without
provenance** (`ontology.py`). The gate re-derives the value from the raw source
at that span and compares (`grounding.py`). No model participates in the
comparison. Mismatch is terminal -- it is never fed back for repair, because a
loop told "this value failed provenance" will produce a better-disguised
hallucination with an invented span.

Two consequences worth stating:

- **Abstention is a legal output.** `FieldAbstention` carries a closed reason
  enum. An `Optional[str] = None` field would *not* be equivalent: structured
  generation prefers filling optional fields over omitting them.
- **`Unclassified` is a first-class outcome**, not a validation failure. A
  closed ontology cannot detect its own blind spots, so the out-of-ontology
  path has to be explicit or documents get force-fit and silently dropped.

## Build order

Built in the order the critique ranked by retrieval-quality gain per unit
effort. Everything here outranks ontology work, which is why the ontology is
deliberately tiny.

| Step | Module | What it fixes |
|---|---|---|
| 1 | `chunking.py` | Chunk-boundary failure: tables split mid-row, headings divorced from sections, references ingested as content |
| 2 | `goldset/` + `goldset.py` | Thresholds that were previously vibes |
| 3 | `store.py` | Raw text stays reachable; verified-only retrieval by default |
| 4 | `store.content_hash` | Idempotent re-ingest via content-addressed identity |
| 5 | `tiering.py`, `ontology.py`, `grounding.py` | Score-gated tiers, ontology v1, the grounding gate |
| 6 | `adjudication.py`, `candidates.py` | T2 contract + bbox-based alignment of two parses |
| 7 | `adapters/docling.py`, `adapters/pdfplumber.py`, `adapters/openrouter.py` | Parser backends + T2 backend |
| 8 | `scoring.py` | Does T2 actually help? Measured, not assumed |
| 9 | `circuit.py` | ...and acted on: the breaker stops T2 when it stops being trustworthy |
| 10 | `ingest_runner.py` | Real corpus at scale: failure isolation and honest accounting |
| 11 | `adapters/docling.py` | Primary parse stream: layout model + TableFormer, page-level confidence |
| 12 | `extractors.py` | T0 extraction: deterministic, provenance-producing, model-free |

The literature survey behind these choices lives in [survey.md](survey.md).

Tiers 0 and 1 contain **no model**.

## T2: adjudication, not extraction

T2 exists to settle *structural* disagreements between parsers — column count,
cell boundaries, reading order, character identity. A disagreement has to be
*manufactured* by running two parses over the same region, which is what
`candidates.py` does.

### Where disagreements come from

`AlternateParseProducer` aligns the two parses by **page + bounding-box overlap**,
never by text similarity — text is precisely what the parsers disagree about, so
using it to align them would beg the question.

The parser pair should be **cross-origin**, not two settings of one parser:
Docling reconstructs structure from a layout model (handles scans, can invent a
grid) while pdfplumber reads the text layer and finds tables from ruled lines
(faithful to digital text, blind to anything unruled). Same-origin parsers share
failure modes and agree on the same wrong answer.

Alignment rules that matter, each tested:

| Rule | Why |
|---|---|
| Nesting counts in **both** directions | One parser may return a table where the other returned a row, or vice versa |
| A page-scale element matches nothing | Otherwise one whole-page element "agrees" with every chunk and manufactures disagreements everywhere |
| No bbox ⇒ no candidates | Without a box the two readings cannot be shown to describe the same region |
| Identical text ⇒ one candidate | Agreement is not a disagreement; spending a model call to learn that is waste |

It is wired to OpenRouter (`adapters/openrouter.py`), and three properties are
enforced structurally rather than by convention:

| Property | How it is enforced |
|---|---|
| T2 never emits a field value | The wire schema has no `value` property; `AdjudicationDecision` forbids extra fields |
| T2 cannot verify anything | A `pick` only narrows the candidate set; the chunk stays quarantined until T1 passes it |
| Grounding failures never reach T2 | `_review_item` checks `MISMATCH` first and returns `UNGROUNDED`, which skips the adjudication branch |

T2 also **refuses to answer without visual evidence.** If the chunk has no
`bbox`, or the page has no rendered image, it abstains instead of judging from
text. Text cannot settle "is that a lowercase L or a one?", and a confident
text-only answer is a guess dressed as an adjudication.

### Setup

```bash
export OPENROUTER_API_KEY=...      # read from the environment; never logged
export RAG_T2_MODEL=anthropic/claude-opus-5-5   # optional override
uv run pytest -m live              # one real call, opt-in only
```

Model choice: adjudication is the last automatic gate and runs on a small
fraction of traffic (escalation rate × volume), so capability beats price here.
Haiku is the wrong trade — saving tens of dollars a month to put the weakest
model at the highest-consequence decision point.

### Was T2 worth it?

Every T2 pick is recorded on the review item, and `ReviewQueue.resolve` copies
it onto the correction automatically — the reviewer never restates it. That
makes the loop measurable:

```bash
uv run rag-goldset --score goldset/t2_labels.example.json
```

```
T2 adjudication quality (plan section 5c):
  scored cases : 3
  picks        : 3
  abstentions  : 0
  precision    : 67%
  coverage     : 100%
  harm rate    : 33%
  verdict      : T2 precision 67% is below 85%: prefer abstention over picking until it improves
```

Three numbers, deliberately kept apart because they fail differently:

| Metric | Means | Why it is separate |
|---|---|---|
| **precision** | of the times T2 picked, how often the owner agreed | A single "accuracy" hides whether T2 was wrong or merely silent |
| **coverage** | how often T2 committed at all | Low coverage is cheap; it is not a failure |
| **harm rate** | fraction of cases where T2 pointed the wrong way | A wrong pick misleads a human; a wasted call does not |

That distinction is the point. A system that abstains often and is never wrong
is healthier than one that always picks and is right 80% of the time — the
human queue cannot tell the difference between a correct pick and a confident
wrong one. `harm_rate` is the number to watch.

Scoring only counts **owner** corrections: a verifier's guess is not ground
truth, and scoring against it measures agreement with a bystander. An owner
answer that does not name a candidate is marked `unscorable` rather than
pattern-matched to the nearest reading, which is how scoring systems start
flattering themselves.

### And it acts on that measurement

A report nobody reads at 3am is not a control. `T2CircuitBreaker` wraps any
adjudicator and stops it when it stops being trustworthy — a standard
three-state breaker (closed → open → half-open probe), with one design point
worth calling out.

**Correctness is invisible to the fast signal.** A confidently wrong pick is a
perfectly well-formed response: transport succeeds, the schema is satisfied,
the breaker sees health 1.0. Only owner labels can catch it. So the breaker
takes two signals with very different latency:

| Signal | Sees | Latency | Blind to |
|---|---|---|---|
| Call health (backend errors, unreadable output, incoherent decisions) | infrastructure breakage | same request | wrongness |
| Owner-labelled precision | wrongness | hours–days | a broken endpoint |

A breaker on labels alone keeps making the same wrong pick for the entire
review lag. A breaker on failures alone never notices wrongness. Both feed it,
and `status.trip_reason` names which one tripped.

When open, the breaker returns a **legitimate abstention** — not an exception —
so the human path handles a degraded T2 with no special casing.

```python
from rag_ingest.circuit import T2CircuitBreaker

breaker = T2CircuitBreaker(OpenRouterAdjudicator())
pipeline = IngestPipeline(queue=queue, extractor=extractor, adjudicator=breaker)
```

The `--score` report ends with the breaker's verdict, so "would T2 be stopped
right now?" is answered where operators already look:

```
Circuit breaker:
  state        : open
  tripped by   : low_precision
  labelled     : 12
```

Two floors keep it from flapping: nothing trips on fewer than 5 calls or 10
labelled cases. A single transport blip is not a pattern.

## The shape-learning loop

The built-in ontology covers two shapes; most real tables fit neither. Rather
than guess, the pipeline turns unrecognised tables into *learnable* shapes, with
the owner's judgment as the only authority and the gate as the only verifier:

```
quarantined table chunks → row signatures (per table: column-major, roles pooled
        only when tables agree on their own column meanings)
  → discovery clusters recurring signatures across documents
  → proposal: the grammar proposer emits a JSON schema + evidence from samples
      (falls back to an LLM proposer ONLY after the grammar refuses; its claims
       are labelled source="llm" and every one is mechanically verified: cited
       values must literally occur in the cited row, columns must fit the width)
  → adoption: the owner must prove the contract with adversarial gold cases
      (a "refusal" case that the gate cannot refuse is rejected — an
       unfalsifiable case is a rubber stamp in reverse)
  → verification: later documents' rows are matched by declared field kinds
      (not by contract name) and every cell re-grounded through the standard gate
  → promotion: a table chunk whose matched rows ALL ground enters retrieval;
      one refused row keeps the whole chunk withheld — a partially understood
      table is not understood. Header rows stay discovery observations.
```

Two invariants are structural, not conventional: the LLM proposer never sees a
grounding failure (it resolves structural questions only), and its pick is
re-checked by T1 before anything enters retrieval.

Measured on a Korean multi-document corpus (3 monthly inventory PDFs + a
shipment log): the inventory shape recurs across documents and its contract
verifies every row of a new month — including the honesty cases, like a header
row read against an enum contract being refused rather than passed.

## Getting the verified chunks out

Verification is the entry requirement for retrieval, and the handoff keeps it:

```bash
curl http://127.0.0.1:8765/api/export > verified.jsonl
uv run python -m rag_ingest.scripts.lightrag_push verified.jsonl http://127.0.0.1:9621
```

- **Verified only.** Export is a projection of the gate, not a choice —
  quarantined chunks cannot appear no matter how the request is shaped.
- **Chunk boundaries survive.** One insert per chunk, so the boundaries the gate
  verified are the boundaries the graph-RAG reasons over; `gate_tier` rides
  along so "contract arithmetic" stays distinguishable from "verbatim quote".
- **Owner approvals ride along too.** A chunk promoted by owner approval carries
  its approval record; downstream stores can separate the two knowledge grades.
- **Duplicate content is reported, not swallowed.** The same prose in twelve
  monthly documents is one fact — LightRAG's 409 lands in a `duplicates` bucket,
  visible and distinct from failures.

Measured end-to-end against a live LightRAG (v1.5.3, SSH tunnel): 6 verified
chunks pushed 6/6, and hybrid queries answer from the injected knowledge with
the source document cited — "재고가 안전재고 이하로 내려가면 발주를 개시한다",
referencing `inventory-2026-08`.

## T0 extraction: making verification non-zero

`TemplateExtractor` is a pattern matcher, not a model. Its job is to find values
*and* say exactly where each came from, because a value without provenance
cannot be grounded and therefore cannot be verified.

The load-bearing part is coordinate translation. A match is found in
`Chunk.text`, but the gate re-derives from the page buffer; the chunk re-joins
elements with `\n\n` and may render a table with pipes, so the two offset
systems drift. `Chunk.span_for` carries both and every extractor goes through
it rather than computing offsets itself.

Two rules that came from real failures against Docling output:

**Partial extraction is not verification.** An invoice where only the
identifier was found used to pass as GROUNDED and enter retrieval claiming to be
a total it did not carry. `MIN_EXTRACTED_FIELDS = 2` makes a thin record
`INCOMPLETE` — its own escalation reason, distinct from `MISMATCH` (terminal) and
`ABSTAINED` (nothing found). This is a required-evidence rule, not a knob to
lower when the queue gets busy.

**A missing field is named, not papered over.** An extraction that declines a
field reports `NOT_PRESENT` — a legitimate abstention — and the required-evidence
rule still blocks verification:

```bash
$ uv run rag-goldset --corpus ./invoices
documents      : 3
verified chunks: 2
quarantined    : 1
verification   : 67%          # the damaged invoice is withheld
```

## Running a real corpus

```bash
uv run rag-goldset --corpus ./pdfs                              # docling if installed
uv run rag-goldset --corpus ./pdfs --parser pdfplumber
uv run rag-goldset --corpus ./pdfs --second-parse                # + the other parser
```

```
primary parser : docling
alternate      : pdfplumber (2 aligned)
documents      : 3
verified chunks: 2
quarantined    : 1
verification   : 67%
T2 consulted   : 0 (0 picks)
disagreement   : 0% of chunks (0/7)
```

**Read `disagreement` before trusting T2.** It is measured at the *candidate
producer*, not at T2, and that distinction matters: T2 is only consulted when a
chunk also has unreliable structure, so counting T2 calls confounds parser
disagreement with an unrelated gate. On the clean synthetic PDF above both
parsers agree on all 4 chunks — the second parse earns nothing there, and T2
could be removed for that corpus. That is the honest answer the metric exists
to give, and it is reported as `not measured` rather than `0%` when no second
parse is configured.

Two properties are enforced rather than hoped for:

- **One document cannot abort the batch.** A corrupt PDF, an encrypted file, a
  parser crash — each becomes a recorded failure for that document. Corpus work
  fails document-by-document or not at all. The corrupt file above is reported
  and skipped; the other three still process.
- **Nothing is silently dropped.** Every document ends in exactly one status
  bucket, and `BatchReport.accounts_for_every_document()` asserts it. A document
  that vanishes between input and output is the bug you find six months later.

`--no-extract` disables T0, so every chunk becomes an unclassified review item.
Useful for a parser-only run; it is not the default.

## Wiring it up

```python
from rag_ingest.adapters.openrouter import OpenRouterAdjudicator
from rag_ingest.pipeline import IngestPipeline

from rag_ingest.candidates import AlternateParseProducer, SecondParse
from rag_ingest.adapters.pdfplumber import second_parse_from_pdf

pipeline = IngestPipeline(
    queue=queue,
    extractor=extractor,
    adjudicator=OpenRouterAdjudicator(),   # omit to escalate instead of guessing
    producer=AlternateParseProducer(
        primary_name="docling",
        alternate=second_parse_from_pdf("report.pdf"),   # the second parse
    ),
)
```

Without an adjudicator the pipeline uses the stub, which raises and is caught
into an abstention — so an unconfigured deployment degrades to the human queue
rather than failing.

## Layout

```
src/rag_ingest/
  docmodel.py     raw element tree + the adapter boundary to Docling/Unstructured
  checks.py       deterministic semantic checks after the gate: dates, arithmetic
  tiering.py      parse/layout score -> tier selector (before any semantics)
  chunking.py     structure-preserving, heading-aware chunks
  ontology.py     ontology v1 (small on purpose) + version stamps
  grounding.py    provenance re-derivation -- the check a schema cannot fake
  escalation.py   chunk-scoped review queue, demand x authority priority, labels
  metrics.py      per-class escalation rate, ontology-rework flag
  store.py        fail-closed-per-chunk retrieval policy, content hashing
  adjudication.py T2 contract: questions and the three-outcome decision
  candidates.py   bbox alignment of two parses -> competing readings
  extractors.py   T0: pattern extraction with provenance spans
  scoring.py      score T2 picks against owner corrections
  circuit.py      breaker wrapping T2: fast + slow signals, half-open probes
  ingest_runner.py corpus batch: failure isolation + per-document accounting
  table_rows.py   row extraction from table chunks: signatures + cell provenance
  discovery.py    shape discovery: row signatures, per-table column roles, clusters
  propose.py      grammar proposer: schema + evidence from cluster samples
  llm_propose.py  LLM proposer fallback: claims only, every claim re-verified
  shapes.py       shape contracts, adoption gate (gold cases decide, not vibes)
  row_verify.py   verify later documents' rows against adopted contracts
  promotion.py    owner approval closes quarantine — through the gate, never past it
  export.py       verified chunks → JSONL (tier + approval history ride along)
  scripts/
    lightrag_push.py  chunk-by-chunk injection into a LightRAG server
  adapters/
    openrouter_client.py  one OpenRouter client resolver for every LLM role
  demo/
    static/index.html    the UI
    (discovery, proposal cards, adoption cases, row verification, approval)
    docling.py    DoclingDocument -> Element/RawDocument, bbox + page images
    pdfplumber.py PDF -> Element/RawDocument, geometric tables + page images
    openrouter.py T2 backend: structured output, image input, degrades safely
    openrouter_client.py one client resolver shared by T2 and the LLM proposer
  pipeline.py     T0 -> T1 -> T2 -> quarantine orchestration (+ row verification)
  gate_tier.py    which bar a claimed record must clear, from its ontology type
  goldset.py      gold-set loader, replay extractor, decision checker
  cli.py          `rag-goldset`
  demo/
    server.py       FastAPI routes: ingest, overlay, page image, review
    session.py      per-session state and view models
    parsing.py      parser selection (same rules as the CLI)
    producers.py    scripted second reading, so T2 is reachable in a demo
    fake_adjudicator.py  scripted T2 -- shape of the path, not judgement
    static/index.html    the UI
tests/            493 tests
goldset/cases.yaml  adversarial cases, each asserting a decision
```

## Run it

```bash
uv run pytest                                 # 493 tests
uv run pytest -m live                         # one real OpenRouter call
uv run pytest --cov=rag_ingest                # coverage
uv run rag-goldset                            # gold-set report
uv run rag-goldset --score LABELS.json        # T2 precision / coverage / harm
uv run rag-goldset --corpus DIR               # batch a real PDF directory
uv run rag-goldset path/to/cases.yaml
uv run rag-demo                               # interactive demo on :8765
uv run ruff check src tests
```

## Deliberate non-goals

Stated so they are not mistaken for oversights:

- **No vector store.** `store.search` is naive lexical matching so the *policy*
  (what may appear at all, how unverified content ranks) is testable without an
  embedding model.
- **Docling is optional.** Both adapters work; `--parser auto` picks Docling
  when it is importable and pdfplumber otherwise. The fake-based adapter tests
  run everywhere; `tests/test_docling_live.py` runs only when Docling is
  installed (mark: `docling`).
- **Docling confidence is page-level, not per element.** The API exposes
  `confidence.pages[n]`; there is no per-element attribution. Every element on
  a page inherits that page's scores, so one bad table drags the whole page's
  parse score down and the tier selector acts page-wise. `table_score` comes back
  as `nan` — documented unimplemented — and is dropped rather than passed
  through, since a `nan` makes every threshold comparison false.
- **Born-digital only.** Docling's OCR pass is disabled (`do_ocr=False`); the
  text layer is the source of truth. A scanned page with no text layer has
  nothing to verify against and escalates rather than guesses.
- **No persisted queue.** The WIP limit and staleness policy are the substance;
  persisting them is a deployment detail.

## What the gold set does and does not tell you

The cases are **adversarial by design** -- most exist in order to fail. So the
escalation rates the CLI prints validate mechanisms, not production health. In
this repo `invoice` shows 100% escalation, which is correct and meaningless as a
health signal. Real rates come from live traffic, where a healthy corpus mostly
verifies and a class whose rate *trends up* release over release is the signal --
and per-class, because a 5% global rate can hide 100% on one high-volume class.

## Known limitations

- **T1 cannot catch two consistent-but-wrong numbers.** The gate re-derives
  values from the *text buffer*, so a total that contradicts its own line items
  passes if both are verbatim. The deterministic checks in `checks.py` close
  part of that gap (dates, arithmetic); T2's page-image path covers the rest.
- Ontology v1 deliberately covers two built-in shapes only. Table shapes grow
  through discovery + adoption; other shapes extend from the cluster report on
  a real seed corpus, not from imagination.
- **`verified` is a weaker claim than it sounds.** It means "the value appears in
  the parser's text buffer where it claims to", not "the value is correct". A
  self-consistent wrong number passes. See [survey.md](survey.md) §3 on why span
  validity does not imply claim support.
- `RawDocument.from_elements` synthesizes page text by joining element
  renderings. Both shipped adapters go through it, so provenance spans index into
  the *joined* buffer rather than the parser's native text. That is internally
  consistent (the gate re-derives from the same buffer) but it means T1 verifies
  "matches what the parser reported", not "matches the PDF bytes". T2's page-image
  path is what closes that gap; a direct text-layer buffer would close it without
  a model.
