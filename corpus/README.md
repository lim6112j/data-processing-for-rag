# Corpus: FAA AC 43.13-1B

**File:** `AC_43.13-1B.pdf` — *Acceptable Methods, Techniques, and Practices:
Aircraft Inspection and Repair*, FAA Advisory Circular 43.13-1B with Change 1.

**Source:** https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_43.13-1B_w-chg1.pdf
(retrieved 2026-10-28)

**License:** U.S. Government work (FAA advisory circular). Not subject to
domestic copyright; public domain. Safe to use as an open corpus — no PII, no
third-party licensing constraint.

**Size:** 646 pages, 21 MB. Text layer present (digital original, not a scan).

## Why this document

Chosen to exercise exactly the split that `survey.md` (Appendix, "should
knowledge and DB-bound content be cleaned to different standards?") says is
unmeasured:

| Property | Count | What it tests |
|---|---|---|
| Pages mentioning "table" | 172 | table extraction |
| Numeric-dense pages (>15% digits) | 93 | DB-bound data volume |
| Pages mentioning torque | 34 | numeric specs with tolerated ranges |
| Pages with part/spec refs (NAS/AN/MS) | 50 | identifier extraction |
| Near-empty pages (<200 chars) | 74 | figure-only pages — must not be read as failures |
| Two-column prose pages | many | reading-order ambiguity → real T2 candidates |

It contains both content classes inside one document, which is the point:

- **DB-bound**: `TABLE 7-1. Recommended torque values (inch-pounds)` (p.320) —
  thread size → tension/shear/limit torque ranges, coarse and fine series.
  Note the `2200-3300*` asterisk footnote: a range qualifier that a naive
  extractor drops.
- **Knowledge**: soldering/flame procedure prose (p.201) — verbatim instructions,
  no arithmetic, unverifiable for correctness but perfectly quotable.

## Profile measured (pdfplumber, whole file)

```
total pages                                    646
chars per page: median 2455, mean 2253
near-empty (<200 chars, figure-like)            74
pages mentioning torque                         34
pages mentioning table                         172
numeric-dense (>15% digits)                     93
pages with part/spec refs                       50
```

## Files

- `AC_43.13-1B.pdf` — full 646-page manual
- `sample-25p.pdf` — 25-page subset (pp. 316–331 torque section + pp. 197–205
  two-column prose), for fast iteration before a full run

Both are gitignored (size). Re-download with the source URL above.

## Running

```bash
# fast iteration on the subset
uv run rag-goldset --corpus corpus/sample-25p.pdf

# full document (slow: Docling loads model weights on first run)
uv run rag-goldset --corpus corpus/AC_43.13-1B.pdf
```

## What to read off the run

Per `survey.md`, in this order:

| Metric | What it decides |
|---|---|
| `unclassified` share | how much knowledge the gate discards → is the LENIENT tier urgent? |
| document-type distribution | this file is manual-like, not invoice-like |
| `extraction_failed` share | damaged vs merely unknown |
| `disagreement` | whether the second parse earns its cost |

Expected: `unclassified` will dominate. `TemplateExtractor` knows one invoice
shape and this document contains zero invoices, so nearly every chunk should
land in `unclassified`. That is not a bug in the corpus — it is the measurement
the appendix says has never been taken.

## Caveats

- **Not an invoice corpus.** If the goal is to validate the existing
  `InvoiceHeader` path, this file tests nothing on that path. Use it to measure
  the knowledge-side loss, which is the open question.
- **Two-column reading order.** pdfplumber interleaves columns (see p.201), so
  raw text is scrambled. This is a genuine structural disagreement source — good
  for T2, but expect high quarantine until a column-aware parser is in the path.
- **Figures.** 74 near-empty pages carry no extractable text. They should
  surface as figure/absent, not as extraction failures; watch that distinction.

## First measurements (2026-10-28)

Run with `TemplateExtractor` and no second parse. Reproduce with the commands
above. **These are the numbers `survey.md` says had never been taken.**

| Corpus / parser | chunks | verified | unclassified | incomplete |
|---|---|---|---|---|
| sample 25p / pdfplumber | 28 | 0 | 19 (68%) | 9 (32%) |
| sample 25p / docling | 49 | 0 | 39 (80%) | 10 (20%) |
| **full 646p / pdfplumber** | **754** | **0** | **613 (81%)** | **141 (19%)** |

Verification rate: **0% in every configuration**.

### Reading 1: the LENIENT tier is the next thing to build

`survey.md` set the threshold explicitly: *"If `unclassified` dominates (say
>60%), the tier design is the next thing to build."* Measured at 81% on the
full document, with 0% verification. The gate currently admits nothing from a
real manual — not because the content is bad, but because no tier exists that
is allowed to admit it.

### Reading 2 (unexpected): a grounded-but-wrong-field bug class

All 141 `incomplete` chunks are false positives, not partial successes. The
extractor found an "invoice number" in a bolt manual — every one is a part
number or standard reference:

```
AN504  AN310  AN960  AN385  MS20073  MS20364  MS20365  NAS1223
STD-410  HDBK-728  AMS-2647  ARP-1870  PRF-39016  DTL-27500  GTSIO-520
```

`extractors.py` `_IDENTIFIER` is `[A-Z]{2,5}[-/]?\d{3,10}`, which is exactly the
shape of aviation hardware designations. Consequences:

1. **Grounding does not catch it, and structurally cannot.** The value *is*
   verbatim in the source span, so T1 returns no `MISMATCH`. Grounding verifies
   presence, not semantic role. This is the sharpest available demonstration of
   why `survey.md` ranks deterministic semantic checks (arithmetic, schema,
   context) above more grounding.
2. **It masks the true classification.** These chunks should have been
   `unclassified` (honest: "not an invoice") and instead became `incomplete`
   (misleading: "an invoice we only partly read").
3. **It wastes review slots with unanswerable questions** — 141 of 754 chunks, or
   19% of the queue, ask the owner to supply an issue date and total amount for
   a page about castle nuts. `ReviewQueue.priority` weights those by retrieval
   demand, but demand for a bolt manual is not zero in a maintenance corpus.

Cheapest fix: require corroboration before assigning `InvoiceHeader` — an
amount plus a date plus a currency, or an explicit label match from
`_INVOICE_HINT`. A part-number-shaped token alone should abstain with
`OUT_OF_ONTOLOGY`, not open an invoice record.

### Reading 3: docling drops table cells silently

During the sample run Docling reported:

```
41 of 233 pdf cells matched neither a row nor a column band of the 28x6 grid
and were dropped from the table
```

That is TABLE 7-1 (the torque table). 18% of its cells vanished, with no
`MISMATCH` and no review item — the chunk is simply smaller than the source.
Grounding cannot see data that was never extracted. Any table-based DB path
needs a cell-count reconciliation against `pdfplumber`'s geometric detection,
which is why `--second-parse` exists.

### Reading 4: the review questions are the right *shape*, wrong *content*

The question template is specific and answerable in principle ("read invoice
number; did not find issue date, total amount and currency. For the missing
fields, what should they be?") — the design goal from `escalation.py` holds. It
is the *extraction* that mislabelled the page, so the question, while
well-formed, is unanswerable for these 141 chunks. Attribution matters when
triaging: this is an extractor bug, not a queue-design problem.

## Second measurements (2026-10-28, after the two fixes)

Two changes, in the order `survey.md` recommended:

1. `TemplateExtractor` now requires corroboration before opening an
   `InvoiceHeader` — a monetary amount or an explicit invoice label. A
   part-number-shaped token alone is not a record.
2. The gate strictness tiers are implemented (`gate_tier.py`), with
   `ProseExtractor` + `LayeredExtractor` making the LENIENT tier reachable.

| Corpus / parser / extractor | chunks | verified | unclassified | other |
|---|---|---|---|---|
| sample 25p / pdfplumber / template | 28 | 0 | 19 | 9 incomplete |
| sample 25p / pdfplumber / layered | 28 | **27 (96%)** | 0 | 1 multi-page |
| full 646p / pdfplumber / template | 754 | 0 | 754 | 0 |
| **full 646p / pdfplumber / layered** | **754** | **722 (96%)** | **0** | **32 multi-page** |

All 141 false-positive invoice records are gone. The 32 remaining quarantines are
all `multi_page_span`: a chunk crossing a page boundary cannot be cited through
the single-page `ProvenanceRef` a LENIENT check needs, so `ProseExtractor`
declines rather than quoting only the part it can cite.

### What 96% does not mean

- **Not a quality number.** LENIENT admits *fidelity of quotation*, which prose
  satisfies almost by construction. The check is not vacuous — an edited quote
  returns MISMATCH, a bad span returns UNVERIFIABLE, both tested — but it says
  "this document says this here", not "this is correct".
- **The DB/knowledge split is still absent.** All 172 table chunks are now
  admitted as `ProseKnowledge`, including `TABLE 7-1`. Verified directly: the
  torque table's rows survive pdfplumber intact (`1/4-28`, `50-70`, and the
  `2200-3300*` footnote are all present in the quoted text). But a quote is not a
  row — nothing can aggregate it, and the `*` qualifier is preserved without being
  resolved to its footnote. `TABLE 7-1` is exactly the DB-bound content the
  Appendix distinguishes, and nothing routes it to STRICT.
- **Docling still drops cells.** The 41-of-233 torque-table cell loss reported
  above is unaffected by either fix: grounding cannot see data that was never
  extracted, and a quote of a damaged table is admitted just as happily as a quote
  of an intact one.

### The next increment, in order

1. **Arithmetic and range reconciliation** (`survey.md` conclusion 1, and the only
   thing that would give STRICT a distinct meaning): line items vs total, torque
   ranges vs their stated psi derivation. This is what makes a table a row.
2. **Multi-page spans**, if 32/754 matters — needs a `ProvenanceRef` that can cite
   a page range, which is a change to the core extraction contract.
3. **A cell-count cross-check** between Docling and pdfplumber for table chunks, so
   dropped cells become a review item instead of an invisible loss.

### Defect marking (2026-10-28)

The cleaning policy question was settled as **mark, never repair**: damaged text
is admitted as what the source says, stamped with `suspected_ocr_damage`, and
stays retrievable. survey.md §1 (LLM repair max match 0.028) and §6.1 (human edit
hit rate 20–30%) both say repair is the weaker move; an unmarked defect does not
disappear, it moves into the answering model, which silently corrects it with no
record. The gold set now carries `damaged-prose-is-marked-not-quarantined`, which
locks in both halves of the policy: admitted *and* marked. A future change that
silently quarantines or repairs damaged prose will fail it.

Measured on this corpus: **0 of 782 chunks marked (sample 28 + full 754)**. That
is the correct result, not a miss — the FAA manual is a digital original with a
clean text layer, so there is no OCR damage in it to find. This corpus cannot
validate the marker against real damage; the gold-set case does that, and a
scanned corpus would be the real-world test.

