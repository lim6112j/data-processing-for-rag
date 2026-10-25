# rag-ingest

**Status:** working prototype. 265 tests, 93% coverage, runs real PDFs end-to-end
through Docling + pdfplumber with OpenRouter as the adjudication backend.

## Summary

Ingestion for RAG that refuses to guess. The output of a document is one of three
things: **verified** chunks that enter retrieval, **quarantined** chunks that
become a specific question to the document owner, or a **recorded failure**. There
is no fourth path where a plausible-looking value slips through.

The problem it exists for: most pipelines gate on "does the extracted output
validate against a schema?" When the extractor is a constrained-decoding LLM, that
question is answered by construction — the decoder masks invalid tokens, so it is
impossible to emit a schema violation. A schema-valid result therefore carries
**zero** evidence of correctness, and the OCR-damaged number (`l,234.56` read as
`1334.56`) passes straight into retrieval with a citation attached.

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

```bash
uv run pytest                          # 265 tests, no network
uv run rag-goldset                     # decision-level gold set (6/6)
uv run rag-goldset --corpus ./pdfs     # real PDFs, per-document accounting
uv run rag-goldset --score labels.json # is T2 still trustworthy?
```

### What it does not do yet

- **Ontology v1 covers two shapes** (invoice header, report section). Real corpora
  will mostly fall to `unclassified`, which is the intended signal for extending it.
- **`TemplateExtractor` is one invoice template**, not a general extractor.
- **T2 is rarely exercised** on clean corpora — no structural damage means no
  disagreement, which is the honest answer but means its precision is unmeasured
  outside the example label set.
- **No vector store, no persistence.** Policies are implemented; storage backends
  are not.

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

Concretely: OCR reads `l,234.56`. The schema wants a decimal. The decoder cannot
emit invalid JSON, so it emits `1334.56`. Validation passes, the chunk enters
retrieval, and the system returns a confidently wrong number with a citation.
That is worse than quarantining it, and it is the default outcome of a
shape-only gate.

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

**A damaged value is not an absent value.** OCR reading `l,234.56` for `1,234.56`
used to be reported as `NOT_PRESENT`, which turned an extraction failure into a
clean-looking abstention and hid it from the metrics. It is now `ILLEGIBLE`, and
it correctly blocks verification:

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
  tiering.py      OCR/layout score -> tier selector (before any semantics)
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
  adapters/
    docling.py    DoclingDocument -> Element/RawDocument, bbox + page images
    pdfplumber.py PDF -> Element/RawDocument, geometric tables + page images
    openrouter.py T2 backend: structured output, image input, degrades safely
  pipeline.py     T0 -> T1 -> T2 -> quarantine orchestration
  goldset.py      gold-set loader, replay extractor, decision checker
  cli.py          `rag-goldset`
tests/            265 tests, 93% coverage
goldset/cases.yaml  adversarial cases, each asserting a decision
```

## Run it

```bash
uv run pytest                                 # 265 tests
uv run pytest -m live                         # one real OpenRouter call
uv run pytest --cov=rag_ingest                # coverage
uv run rag-goldset                            # gold-set report
uv run rag-goldset --score LABELS.json        # T2 precision / coverage / harm
uv run rag-goldset --corpus DIR               # batch a real PDF directory
uv run rag-goldset path/to/cases.yaml
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
  OCR score down and the tier selector acts page-wise. `table_score` comes back
  as `nan` — documented unimplemented — and is dropped rather than passed
  through, since a `nan` makes every threshold comparison false.
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

- **T1 cannot catch a self-consistent OCR error.** The gate re-derives values
  from the *text buffer*, so if OCR read `l,234.56` consistently, a claim of
  `1234.56` can pass. T2 sees the page image and closes that gap — which is a
  concrete reason T2 earns its keep beyond parser disagreements.
- `_value_matches_span` has four comparison tiers; tier 4 (separator-stripped)
  is permissive. It reports `needed_normalization: True` so permissive matches
  stay visible rather than silently accepted.
- Ontology v1 deliberately covers two shapes only. Extend it from the cluster
  report on a real seed corpus, not from imagination.
- `RawDocument.from_elements` synthesizes page text by joining element
  renderings. Both shipped adapters go through it, so provenance spans index into
  the *joined* buffer rather than the parser's native text. That is internally
  consistent (the gate re-derives from the same buffer) but it means T1 verifies
  "matches what the parser reported", not "matches the PDF bytes". T2's page-image
  path is what closes that gap; a direct text-layer buffer would close it without
  a model.
