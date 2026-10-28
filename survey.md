# LLMs for data cleaning / data quality / document preparation for RAG

Literature survey, 2025–2026. Compiled to decide whether an LLM can author the
rules that a deterministic validator then executes.

> **Read this before citing anything below.** Results were gathered by automated
> web research. Several 2026 items are preprints whose stated venue could not be
> independently verified ("to appear SIGMOD'26", "ICDM 2026", "ACL 2026").
> A few figures came from search summaries of a paper's tables rather than the
> primary text; those are flagged in-line. Where the evidence is thin or absent,
> that is stated rather than papered over.
>
> **The three findings that changed our design:**
>
> 1. Deterministic profiling **outperformed** LLM-based error detection
>    (F1 0.561 vs 0.421), and LLM *repair* is near-worthless (0.028).
> 2. In mature human edit systems the flag hit rate is **20–30%** — 70–80% of
>    warnings are false alarms. Precision expectations must be set accordingly.
> 3. Span grounding is **not solved**: best word-level grounding F1 is 46.4%.
>    Requiring a span is necessary, not sufficient.

---

## 1. LLM as error detector vs error corrector

**Bottom line: detection and repair are separate problems with different
evidence requirements, and LLM repair is much weaker than LLM detection. But
there is NOT a documented consensus that LLMs must only propose claims. The
best recent survey explicitly declines to declare one mode superior.**

### 1.1 The cleanest statement of the distinction

- **Claim:** "Detection can be supported by profiling, constraints, and anomaly
  indicators, but repair requires stronger evidence because a value may be
  unusual and still valid." The paper frames these as "separate tasks that
  require different levels of evidence."
- **Measured:** Best detection F1 was the **deterministic profiling baseline at
  0.561**; the best LLM-based configuration reached **0.421** (≈0.14 F1 worse).
  Repair was near-absent: max repair-match **0.028**, and the most conservative
  config "performed no direct repairs." Adding capabilities produced trade-offs,
  not monotonic gains.
- **Source:** *Agentic Data Cleaning Without a Clean Reference: An Experimental
  Study of Capabilities and Trade-offs*, Hadi Fadlallah, arXiv:2608.14765,
  14 Aug 2026 (submitted to *New Generation Computing*), 126 runs across
  financial/clinical/environmental data. <https://arxiv.org/abs/2608.14765>
- **Strength:** Benchmark-style controlled study, single author, **no
  significance testing reported** (the paper itself says no p-values/CIs).
  Directionally strong, statistically weak.

### 1.2 Mechanism for why repair is harder

- **Claim:** For detection the verifier only needs a binary flip (precision
  after verification ≈ 1.0); for repair the answer space is open, so
  verification precision ≈ 0.5 ("usually one obvious correction"). The paper
  gives a formal condition for when a critic helps:
  `(1−p_g)·p_c·p_r > p_g·(1−p_c)`, with typical repair probability
  `p_r ≈ 0.5` vs `≈ 1.0` for detection.
- **Measured:** Debate raised **error-detection F1 by +27.4 points**
  (d = 1.0, p < 0.001) but **degraded generation by −1.6 to −15.5 pp** across
  four model families. Self-verification with the same tools failed
  (+0.3 to −1.3 pp); only a separate critic with sandboxed code execution plus
  evidence-gated generation won (+5.3 pp).
- **Source:** *When Helping Hurts and How to Fix It: Multi-Agent Debate for Data
  Cleaning*, Parmar et al., arXiv:2606.02866, 1 Jun 2026.
  <https://arxiv.org/abs/2606.02866>
- **Strength:** Large within-subjects study (3 benchmarks, 4 model families,
  >6,000 task-condition pairs), pre-registered-style prediction check across 19
  published comparisons. **Preprint.** Best evidence found for the
  detect/correct asymmetry.

### 1.3 Hybrid systems that already separate the roles

- **Cocoon** — *Data Cleaning Using Large Language Models*, Zhang, Huang, Wu,
  arXiv:2410.15547, 21 Oct 2024; published **ICDEW 2025**
  (DOI 10.1109/icdew67478.2025.00008). Not SIGMOD, contrary to some citations.
  The LLM judges semantics, output is compiled to SQL, statistics do detection.
  Reported F1: Cocoon **0.90/0.57/0.97/0.86/0.87**
  (Hospital/Flights/Beers/Rayyan/Movies) vs HoloClean 0.63/0.47/0.04/0.59/0.00,
  Raha+Baran 0.72/0.70/0.96/0.50/0.80. End-to-end LLM baselines **CleanAgent and
  RetClean scored ~0 on most benchmarks** — the paper's strongest evidence
  against LLM-direct-rewrite. No LLM-only ablation was run, so the source of the
  gain is untested.
- **IterClean** — Ni et al., ACM TUR-C 2024, DOI 10.1145/3674399.3674436.
  Explicit **Error Detector → Error Verifier → Error Repairer** roles in a loop.
  Claims "up to 3× F1" over SOTA with 5 labeled tuples, and that the sequential
  detect-repair paradigm "leaves ~48% of errors unresolved." **Strength:**
  peer-reviewed workshop, but the ACM PDF could not be fetched (403); numbers
  come from abstract/secondary summary.
- **ZeroED** — Ni et al., arXiv:2504.05345, 6 Apr 2025. LLM labels only
  cluster-representative rows, then a conventional classifier detects
  corpus-wide. **+30% F1 max, up to 90% token cost reduction.** This is the
  "LLM proposes, deterministic system scales the decision" pattern in its most
  literal form.

### 1.4 The claim→verify pattern outside cleaning

- **VeraDM** — *From Natural-Language Hypotheses to Verified Data-Mining
  Discoveries*, ICDM 2026 (venue unverified),
  <https://lexuanbach.github.io/publication/ICDM2026b.pdf>. Explicit "strict
  separation between proposal and acceptance": LLM proposes constrained
  hypotheses, a verifier compiles each into a typed query and accepts only
  through six evidence gates (support, effect threshold, permutation test,
  BH/BY FDR correction, matched contrast, held-out replication), each producing
  an auditable "evidence card." **The best formalization of the architecture
  this repo is considering — but it is data mining, not data cleaning.** No
  cleaning paper adopts it wholesale.
- **Counter-evidence on self-verification:** Stechly, Valmeekam, Kambhampati,
  arXiv:2402.08115 (Feb 2024, rev. Aug 2024): GPT-4 shows "significant
  performance collapse with self-critique" and gains only with *sound external*
  verification. **ReaLMistake** (Kamoi et al., **COLM 2024**, arXiv:2404.03602):
  GPT-4/Claude 3 detect LLM errors with "very low recall," all 12 LLM detectors
  far worse than humans, explanations unreliable, self-consistency/majority
  voting did not help. This bounds how much an LLM can be trusted as its own
  verifier — it argues for external/deterministic verification, not for
  LLM-as-judge.

### Where evidence is thin

No paper found runs the direct head-to-head: "LLM proposes claim + deterministic
verifier" vs "LLM rewrites value" on the same cleaning benchmark with the same
model. The case is assembled from adjacent evidence (near-zero LLM repair-match,
0-F1 end-to-end baselines, 20–30% edit hit rates in official statistics).

The survey *Can LLMs Clean Up Your Mess?* (arXiv:2601.17058, 22 Jan 2026,
19 authors incl. Guoliang Li, Yeye He, Yuyu Luo) explicitly reports a
**pluralistic** landscape — coupled detect+repair (IterClean, LLMErrorBench),
rules-first (LLMClean), detection-only (ZeroED, LLM-TabAD), repair-only (GIDCL),
hybrid (ForestED) — and **does not identify a consensus** on proposing vs direct
fixing. It flags "persistent hallucinations even in advanced agents." **So:
practitioner norm is converging on propose-then-verify, but the literature has
not ratified it.**

---

## 2. Hybrid rule + LLM architectures

**Bottom line: the "deterministic first, LLM on the residual" pattern is
well-attested academically and in data-*curation* tooling, and is the explicit
recommendation of the one enterprise field study. But the major warehouse
vendors (Databricks, Snowflake) do NOT document it. No vendor publishes a
rule-vs-LLM coverage ratio.**

### 2.1 The single best direct evidence (enterprise, production provenance)

- **Claim:** Rules and LLMs are systematically complementary. Rules are
  deterministic, accurate and cheap but blind to semantic errors; LLMs catch
  semantics but misfire where local rules are authoritative. Authors
  **recommend a hybrid architecture**: rules as primary validator for
  structured/format-sensitive fields (email, DOB, postal code, phone), LLM
  applied selectively where semantic judgment is needed (names, locality, free
  text).
- **Measured (Exp. 1, 10,000 synthetic customer records, Microsoft 365 Copilot
  vs Ataccama ONE):** Copilot F1 **1.00** on "salutation in surname,"
  "placeholder," "empty names"; **0.89** on name swaps (precision 0.81, 109
  false positives from genuinely ambiguous names); **0.97** any-error.
  Exp. 2 (10,000 real CRM records, 14 fields, 28 Ataccama rules): rules won on
  postal-code–locality (multiple valid codes per locality); LLM won on
  company-names-in-name-field, staircase-info-in-house-number, placeholders;
  **both failed** on street names in the house-number field.
- **Source:** Korom, Dominka-Kiss, Glock, Ehrlinger, *Data Quality Validation in
  Enterprise Master Data Management*, **DATA 2026**, SciTePress, pp. 961–968,
  DOI 10.5220/0015161700004091.
  <https://www.scitepress.org/PublishedPapers/2026/151617/>
- **Strength:** Peer-reviewed conference proceedings (low-prestige venue), real
  corporate MDM program (Austrian Post), but n = 2 experiments and a single LLM
  via Copilot. The most concrete rule-vs-LLM coverage evidence in the corpus.

### 2.2 Second direct study (weaker)

- **Claim:** Rules detected 100% of 5 structured error categories and **0%** of
  2 context-dependent categories; the LLM caught the context-dependent errors
  but **missed duplicates entirely and had precision 0.35**; hybrid had the best
  overall **recall 0.98** but inherited the LLM's false alarms and was far
  slower/costlier. Conclusion: rules should remain the core, LLM used
  selectively.
- **Source:** *LLM-Assisted Data-Quality Validation: Where It Helps and Where
  Rules Still Win*, Bhanu Avusherla (UNT), Zenodo 20921341, 26 Jun 2026.
  <https://zenodo.org/records/20921341>
- **Strength:** **Preprint, not peer reviewed.** Synthetic data with 420
  injected errors. Directionally consistent with 2.1, not independent
  confirmation.

### 2.3 Generate-then-filter — the closest published match to this repo's design

- **Claim/measured:** LeDQeR generates candidate rules in the target tool's
  syntax (Great Expectations), then applies four filters:
  **syntactic** (executable), **correctness** (re-run on the triggering tuple;
  must be a true positive), **coverage** (recall on same-error-type tuples —
  note: threshold was disabled in this implementation, so it removed nothing),
  **redundancy** (greedy covering on a held-out set).
- **Results:** 87.13–100% of generated rules executable; **mean F1 0.66**
  (best "embedded values" 0.87, worst "contradictions" 0.40); mean recall 0.88.
  At **precision = 1, rules covered ≥50% of dirty tuples on 4 of 5 test sets**;
  on the tax set, tolerating **6 false positives (0.007%)** took coverage to
  **99.58%**. Final sets shrank to **2–136 rules**.
- **Source:** Glock, Hütter, Fürnkranz, Wöß, Dominka-Kiss, Ehrlinger, *Data
  Quality Rule Generation with LLMs*, arXiv:2609.06053, 5 Sep 2026.
  <https://arxiv.org/abs/2609.06053>
- **Strength:** Preprint, but a clean, reproducible pipeline with the most
  usable precision/coverage trade curve found. **The 50%-at-zero-FP /
  99.6%-with-6-FP contrast is the single most quotable ratio in this survey.**

### 2.4 Related: LLM drafts rules, codegen turns them into executable validators

**Quality Assessment of Tabular Data using LLMs and Code Generation**, Akella,
Kaul, Narayanam, Mehta — **EMNLP 2025 Industry Track**, pp. 2713–2748,
<https://aclanthology.org/2025.emnlp-industry.183/>. Pipeline: statistical
inlier detection/clustering → LLM proposes rules → codegen LLM writes
`check(df)` validators → code-repair loop on exceptions. Guardrails include a
conflict-resolution pass, an 8-point rubric (including "Risk-false-positive"
and "Hallucinated-overly-specific" labels), and execution on sample rows.
Coverage = share of schema columns touched by ≥1 rule, avg **0.614** with
few-shot vs **0.489** without.

*Caveat on the headline claim:* the paper says "top F1 in 26 of 30 settings";
Table 2 shows **24 outright wins + 1 tie = 25** (Nasa at 30% noise loses to
Max-Entropy). Flagged because the claim is being repeated elsewhere.

### 2.5 Data-curation tooling (same pattern, corpus scale, documented)

- **NVIDIA NeMo Curator docs:** "Order matters: Place computationally
  inexpensive filters early in your pipeline." Heuristic filtering → dedup →
  model-based filtering → blending. Explicit tiering: fastText (100K–1M
  samples) → BERT-style → LLM/reward model, with LLMs recommended only "for
  smaller, specialized or fine-tuning datasets."
  <https://docs.nvidia.com/nemo/curator/curate-text/process-data/quality-assessment>
- **NVIDIA engineering blog** (13 Nov 2024): "implement a cascading approach";
  "combining lightweight models for initial filtering with advanced models for
  final quality assessment often provides the best balance."
  <https://developer.nvidia.com/blog/mastering-llm-techniques-data-preprocessing/>
- **Data-Juicer 2.0** (arXiv:2501.14755): a runtime `Adapter` probes each
  operator on a small batch, then reorders so faster OPs precede slower ones
  (only where commutative). Combined reordering+fusion at "at most 70.22%" time
  savings (not isolated reordering).
- **Strength:** Industry docs/engineering reports — implementation evidence, not
  measured accuracy. The clearest corpus-scale expression of the pattern.

### 2.6 Where the vendors *don't* say it (important negative result)

- **Databricks:** The AI-First Data Engineering post (24 Feb 2026) says only
  "Use fuzzy matching and entity resolution to fix duplicates and
  inconsistencies at scale" — **no rules-first recommendation, no expectations,
  no validation guidance.** The AI Functions docs give operational advice
  instead: sample ≥10,000 rows before scaling ("The cost-accuracy trade-off is
  specific to each use case"), use DDL STRUCT `responseFormat`, version prompts
  like code, prefer task-specific functions over `ai_query`, tag jobs on day
  one. No per-document cost figures.
- **Snowflake:** AI Observability docs expose Groundedness/Context Relevance as
  **LLM-as-judge metrics with chain-of-thought**, default judge llama3.1-70b,
  and the docs do **not** discuss judge reliability or validation limits.
  <https://docs.snowflake.com/en/user-guide/snowflake-cortex/ai-observability/reference>
  Snowflake ships an LLM-judge grounding metric *without* the deterministic
  cross-check the rest of this survey argues for.
- **Google (Dataplex / Knowledge Catalog):** the one vendor documenting both the
  pattern and its limits. LLM (Gemini) generates DQ rules from profiling stats;
  docs warn "large language models (LLMs) are non-deterministic," call HITL
  review "the most critical step in the entire workflow," and instruct users to
  "iterate with the agent to refine the rules and eliminate false alerts."
  <https://docs.cloud.google.com/dataplex/docs/profile-ensure-data-quality>
  (last updated 2026-10-24)
- **Great Expectations ExpectAI:** AI suggests Expectations; a human approval
  workflow gates them ("selectively approve only those rules that align");
  suggestions expire in 48h. The blog claims "fewer false positives" but
  **gives no metric, no acceptance rate, no reviewer-effort number.**
  <https://greatexpectations.io/blog/gx-expectai/> (1 May 2025)

### Where evidence is thin

No vendor publishes a rule-vs-LLM coverage ratio. Nearest quantitative proxies:
LeDQeR's precision/coverage curve (§2.3), the Austrian Post error-type split
(§2.1), and the "rules 100% structured / 0% semantic" split (§2.2, preprint).
Most "hybrid" claims are architectural assertions with an ablation missing —
Cocoon, for instance, never isolates LLM vs statistical contribution.

---

## 3. Grounding / citation for data-quality findings

**Bottom line: span-grounding is converging as a documented best practice and is
now *architecturally guaranteed* at the API layer by at least one provider. But
requiring a span does NOT make the finding correct, and measured span-level
attribution accuracy is poor enough that "cite the span" must be treated as
necessary, not sufficient.**

### 3.1 Strongest guarantee available (industry, primary docs)

- **Claim:** With the Citations API, "Because the API parses citations into the
  response formats described in the following sections and extracts `cited_text`
  directly, citations are guaranteed to contain valid pointers to the provided
  documents." Granularity: `char_location` (plain text), `page_location` (PDF),
  `content_block_location` (custom). `cited_text` is API-extracted, not
  model-generated, and does not bill as output tokens.
- **Source:** Anthropic, *Citations*,
  <https://platform.claude.com/docs/en/build-with-claude/citations> (GA).
- **Strength:** Vendor documentation describing an API-level construction
  guarantee — about as strong as grounding evidence gets, because the model
  cannot fabricate a span it does not emit. **Caveat: the guarantee is that the
  pointer is valid, not that the cited text supports the claim.**

### 3.2 Grounding as detect-and-filter, not guarantee

- **LangExtract** (Google, OSS): maps every extraction to `char_interval`;
  extractions that "cannot be located in the source text will have
  `char_interval = None`," which users filter out. It auto-detects when the
  model extracted from few-shot examples rather than the input. It does **not**
  claim hallucination prevention. <https://github.com/google/langextract>
- **Databricks `ai_enrich`:** returns per-field `value` + `rationale` and
  row-level `metadata.sources`; the prompt instructs "return null rather than
  guessing" / "Preserve that distinction instead of replacing it with a guessed
  value." **Documented limitation: provenance is row-wide, not
  field-specific.**
  <https://docs.databricks.com/aws/en/large-language-models/ai-enrich-tutorial>
- **Pattern:** these are *detection* mechanisms (null the interval, null the
  value), not verification of correctness.

### 3.3 How good is span attribution actually? (the sobering numbers)

| Finding | Number | Source | Strength |
|---|---|---|---|
| Best schema-guided extraction grounding | **word-level F1 46.4%** vs page-level 84.9%; VLMs/coding agents "do not return evidence by default; they therefore score zero at both grounding levels" | *ExtractBench*, Zhang et al., arXiv:2607.29677, 31 Jul 2026 | Benchmark, 4,869 pages / 370 docs / 67 doc types |
| Table attribution | "close to chance for JSON inputs across all models"; models cite rows better than columns; worse on text than images | *ViTaB-A*, Alqurnawi et al., arXiv:2602.15769, 17 Feb 2026 | Benchmark |
| Citation support completeness | even leading models "lack complete citation support 50% of the time" on ELI5 | **ALCE**, Gao et al., **EMNLP 2023**, arXiv:2305.14627 | Peer-reviewed benchmark |
| Deep-research citations | link validity **>94%**, relevance **>80%**, but factual accuracy only **39–77%** | *Cited but Not Verified*, Onweller et al., arXiv:2605.06635, 7 May 2026 | Benchmark, 14 models |
| Attribution by construction | multi-source attribution accuracy **38% → 64%** when provenance is a structural invariant (claims tied to token spans) rather than predicted post hoc | *Attributable by Construction* (CAMS), arXiv:2606.23989, 22 Jun 2026 | Benchmark; single author |
| Traceable KG statement validation | precision **88%**, recall **44%** when grounded in retrieved passages instead of model knowledge | Adam & Kliegr, **Information Processing & Management 62 (2025) 104128**, arXiv:2409.07507 | Peer-reviewed journal |

### 3.4 Verification architectures for hallucinated findings

- **Microsoft Research, Universal Verifier** (21 Apr 2026): separates rubric
  generation from scoring and scores **twice — with and without screenshot
  evidence — treating disagreement as a hallucination signal.** Reported outcome
  FPR **0.01 internal / 0.08 Browserbase**, vs WebVoyager **0.45** and WebJudge
  "at least 22%"; κ 0.64/0.58 vs 0.44/0.26. The gain is architectural, not
  model-driven (swapping WebVoyager's backbone GPT-4o→GPT-5.2 cut FPR 0.45→0.10
  but pushed false negatives 0.24→0.44).
  <https://www.microsoft.com/en-us/research/articles/the-art-of-building-verifiers-for-computer-use-agents/>
  **Strength: official research blog with numbers; not peer-reviewed.**
- **Microsoft Tech Community, "AI Slop in Enterprise Analytics"** (2 Oct 2026):
  recommends making "claim verifiability a system requirement," doing arithmetic
  in code, and enforcing that "a citation must support the specific assertion,
  not merely be present."
  **Strength: opinion/community post. Directionally right, cite as practitioner
  guidance only.**
- **Reducto table-judge benchmark** (10 Jul 2026): LLM judges silently *fix*
  typos, so five models rejected HTML that matched the source
  ("Pre-Primany / Primany"). Recommendations: **split rubrics into separate
  calls, supply a reference output, and report false alarms alongside
  detection** so conservative judges aren't mistaken for good ones.
  <https://reducto.ai/engineering/posts/llm-judges-table-ocr-false-positives/>
  **Strength: vendor benchmark with released dataset/prompts; the failure mode
  is credible and reproducible.**

### Where evidence is thin

**No benchmark jointly scores (detected error, cited span) correctness for data
quality.** ExtractBench scores values and grounding separately; ViTaB-A scores
attribution on QA; nothing scores "this cell is wrong *and* here is the span
proving it" as one unit. A genuine gap for any claim-with-evidence system.

"Require exact spans" is documented practice at Anthropic/Google/Databricks, but
**no ablation** was found showing it reduces false data-quality findings vs
ungrounded findings. The mechanism is plausible (LangExtract's null-on-ungrounded,
Microsoft's two-pass disagreement) but unquantified for this task.

---

## 4. Measuring data cleaning quality

**Bottom line: detection F1 for good systems sits around 0.55–0.95 depending on
dataset; LLM-only detection sits around 0.35–0.75 with high variance; LLM
*correction* is far worse (0.25–0.40). Benchmarks are proliferating but most are
synthetic-error-injected, and there is active methodological criticism that
reported gains shrink under matched operating points.**

### 4.1 Benchmarks that now exist

| Benchmark | Scale | What it measures | Venue |
|---|---|---|---|
| **TableEG** (arXiv:2507.10934) | trained on 12 real datasets / 10 domains; error generator | realistic synthetic errors to benchmark detection/correction | preprint, v3 Mar 2026 |
| **TABARD** (ACL Anthology 2025.findings-emnlp.1189) | 5,295 tables, 8 anomaly types | detection precision/recall/F1 | Findings of EMNLP 2025 |
| **SEC-FinTables** | 103,395 table instances | 4 sub-tasks: error count, missing count, categorization, exact row/col positioning | Findings of ACL 2026 |
| **DCA-Bench** (arXiv:2406.07275) | 221 cases, 8 platforms | discovery of *unknown* data quality issues | **KDD 2025** |
| **Clean Me If You Can** (arXiv:2606.31983) | 9.3M / 4.3M real address records | detection + correction on real dirty data | preprint, Jul 2026 |
| **ExtractBench** (arXiv:2607.29677) | 4,869 pages | value F1, record completeness, grounding F1, cost | preprint, Jul 2026 |
| **OmniDocBench** (arXiv:2412.07626) | 1,355 PDF pages, 9 doc types | end-to-end parsing (edit distance, TEDS, CDM, reading order) | **CVPR 2025** |
| **OHRBench** (arXiv:2412.02592) | 1,261 PDFs, 8,561 pages, 8,498 QA | downstream RAG impact of OCR noise | preprint v4 |
| **InduOCRBench** | 570 PDFs / 3,402 pages, 12 industries | OCR fidelity **and** RAG impact | **ACL 2026 Industry Track** |

### 4.2 Actual accuracy numbers (the part most blog posts omit)

**Tabular error detection F1 (ForestED paper, GPT-5, Table 3):**

| Dataset | Raha | ZeroED | TreeED | ForestED |
|---|---|---|---|---|
| Rayyan | .570 | .686 | **.709** | .704 |
| Hospital | .140 | .531 | .472 | **.756** |
| Flights | .430 | .585 | .697 | **.720** |
| Beers | .680 | .678 | .782 | **.947** |
| Billionaire | .130 | .455 | .358 | **.566** |
| Movies | .250 | .698 | .583 | **.735** |
| Tax | .487 | .505 | .838 | .838 |

Mean (computed by the survey author, not a paper figure): Raha .403, ZeroED
.591, TreeED .634, ForestED .752. Reported claim: +16.1% avg F1 over best
baseline. **Source:** Wang et al., **KDD 2026** (DOI 10.1145/3770855.3817642),
arXiv:2512.07246. **Note the spread: 0.566 to 0.947 on the same method.**
TreeED/ForestED is quorum-by-EM over LLM-induced decision trees whose rule nodes
are LLM-*authored but deterministic-at-inference* Python predicates ("directly
incorporate the LLM-generated code without further processing"). **The closest
published instance of "LLM proposes checks, deterministic execution decides."**

**Other measured numbers:**

- **Cocoon hybrid:** 0.57–0.97 F1 vs HoloClean 0.00–0.63; end-to-end LLM
  baselines ~0.00.
- **Agentic cleaning:** deterministic profiling 0.561 vs best LLM 0.421;
  repair-match ≤0.028.
- **LLM-as-detector on real addresses** (Clean Me If You Can): detection
  **P 0.97–0.98 / R 0.56–0.60 / F1 0.71–0.74**, second to Raha (0.90–0.91 F1);
  **correction F1 0.24–0.29** (Baran best at 0.40). LLM was best on
  abbreviation/language/word-substitution errors. Cost: **~8h for 1M records,
  ~$650 per full-dataset run**, times out at 4.3M.
- **TABARD:** best precision **46.8–56.9%** (self-verification / SEVCOT); value
  anomalies reach 95% recall under neuro-symbolic constraint execution;
  **factual and temporal anomalies <38% recall**.
- **SEC-FinTables:** DeepSeek-R1 best localization **precision 0.420, IoU
  0.325**; GPT-5 missing-count accuracy 0.872; scaling is **non-monotonic**
  (Qwen2.5-32B beats 72B on error classification). *Figures from a search
  summary of the full text; the ACL landing page confirms the "only partial
  competence" conclusion but not the numbers.*
- **DCA-Bench:** strongest agents surface **~29.86% of issues without hints**,
  ~78.28% with the most specific hint (KDD 2025).
- **Oracle-like upper bound on simple tasks:** brand-mislabeling LLM F1
  **0.833** vs naive rule 0.721; entity matching LLM **0.948** vs Jaccard rule
  **0.950** (a tie). *When Do LLMs Actually Help?*, arXiv:2608.18158, 13 Aug
  2026 (single-author preprint).
- **Document parsing (OmniDocBench, CVPR 2025):** overall edit distance English —
  MinerU 0.150, GPT-4o 0.233, Qwen2-VL-72B 0.252; table TEDS ~72–79; handwritten
  notes: MinerU 0.984 (fails), GPT-4o 0.607, Qwen2-VL 0.168; rotated text is
  catastrophic (PaddleOCR 0.285 at Rotate270).
- **Downstream RAG impact (OHRBench):** severe semantic OCR noise → "nearly a
  50% performance decline" for most retriever–LLM pairs; best OCR still ~5 F1
  points behind ground truth; formatting noise → up to 12.7% retrieval loss
  (BGE-M3), 9.1% generation loss (Llama3.1-8B); tables/formulas/charts hardest.
- **InduOCRBench (ACL 2026 Industry):** "high OCR accuracy ≠ good RAG
  performance" — *VisualStyle* has 82.9% OCR accuracy but ~52.8–53% RAG accuracy
  (**30.1-point blind spot**) because strikethrough/color cues are discarded.
  Avg gap 16.22% vs 16.72% across two chunking/embedding pipelines, i.e. stable.

### 4.3 How teams are told to measure — and why it's hard

- **Randomised, hand-labelled samples are the standard audit.** DocETL's
  optimizer runs plans "on sampled data," rates them, and compares the top six
  pairwise; optimizer cost ($1.58 with gpt-4o-mini) "does not increase with
  dataset size, as it is done on a sample." BARGAIN/DocETL cascades calibrate on
  ≤400 oracle labels by default.
- **Operating-point matching is now required.** *Unmasking Removal-Budget
  Confounding* (arXiv:2608.06511, 6 Aug 2026) shows adaptive cleaners look
  better because they remove *fewer records*, not because they discriminate
  better; once budget and recall are matched, "differences shrink or vanish at
  low-to-moderate corruption." Recommends AUROC/AUPRC plus matched-budget and
  matched-recall controls. **Not about LLM cleaning, but the confound transfers
  directly to any precision-driven data-quality gate.** Preprint, methodological.
- **Benchmark validity is being openly criticized.** Clean Me If You Can argues
  error generators "rely on randomness or explicit patterns, in contrast to
  latent patterns that manifest in the real world," and that downstream-accuracy
  proxies are unreliable ("proxy scores such as downstream inference accuracy
  might even improve through artificial noise"). TableEG reports synthetic-vs-real
  F1 gaps of 0.05–0.23.
- **Over-cleaning is measurable and hurts fairness.** *Automated Data Cleaning
  Can Hurt Fairness in ML-based Decision Making*, Guha, Khan, Stoyanovich,
  Schelter, **ICDE 2023** (DOI 10.1109/ICDE55515.2023.00303), 26,000+
  evaluations: cleaning is usually neutral, but **when it matters it is more
  likely to worsen fairness than improve it** — outlier repair worsens 19.6% vs
  improves 8.2%; missing-value cleaning 23.6% vs 17.1%; equal-opportunity
  worsens 32.7% vs improves 7.5%.

### Where evidence is thin

- **No benchmark for document-level data quality in a RAG corpus.** OHRBench and
  InduOCRBench measure *downstream RAG* damage from parsing noise; OmniDocBench
  measures *parsing fidelity*. Nothing gives ground-truth "this document/field
  is wrong" labels at corpus scale.
- Reported F1s are **not comparable** across papers: different error generators,
  error rates (4%–52%), and label budgets. ForestED's own evaluation gives
  supervised baselines only 2 labelled tuples and evaluates ZeroED on 500
  sampled rows "due to token cost."
- **No published precision/recall for a claim-with-evidence data-quality
  reviewer** — presumably the system class this repo is building.

---

## 5. Cost / latency at corpus scale

**Bottom line: the cascade/proxy-oracle pattern is the dominant documented
answer, now with PAC-style guarantees (BARGAIN, SIGMOD'26) and production
implementations (LOTUS, Palimpzest, DocETL). Reported savings are large but
measured on academic workloads. The known failure mode is that routing collapses
under distribution shift.**

### 5.1 The strongest current result

**BARGAIN** — *Cut Costs, Not Accuracy: LLM-Powered Data Processing with
Guarantees*, Zeighami, Shankar, Parameswaran, arXiv:2509.02896, **to appear
SIGMOD'26** (venue unverified). Cheap proxy model scores everything; adaptive
sampling walks candidate thresholds from high to low, labelling only records
above each threshold, using anytime-valid confidence sequences. Guarantees
accuracy, precision, or recall hold with probability ≥ 1−δ. **"Cuts oracle usage
by up to 86% more than SUPG"** across 8 datasets; SUPG missed targets ">75% of
the time" at a 10% failure probability; GPT-4o-mini is >15× cheaper than GPT-4o.

Limits it states itself: requires a proxy confidence score (logprobs) that must
be reasonably calibrated; precision/recall modes are **binary only**; user must
supply an oracle budget; a proved negative result for very-low-prevalence recall
queries. **Strength:** peer-reviewed-venue preprint with proofs. Best evidence
in this section.

### 5.2 System-level numbers

| System | Reported | Notes |
|---|---|---|
| **LOTUS** (VLDB 18, p.4171) | defaults γ=0.9, δ=0.2; importance sample **0.01% of data, min 100**; cascade routes high-confidence to proxy, middle band to oracle | BioDEX join: **5,290 LLM calls / 2,116 s** vs 27,687 / 12,563 s for the alternative, with +37% RP@5; ~1,000× fewer calls than the gold algorithm. ArXiv group-by proxy-only was **17.4× faster but 39% less accurate** — sampling overhead can be bought, accuracy is not free |
| **Palimpzest** (CIDR 2025; arXiv:2405.14696) | one plan **3.3× faster, 2.9× cheaper** with better F1; with parallelism **90.3× speedup at 9.1× lower cost**, F1 within 83.5% of GPT-4 | optimizations: filter/convert reordering, model selection, code synthesis, prompt marshaling, input-token reduction |
| **DocETL** (arXiv:2410.12189) | optimizer cost **$1.58** (gpt-4o-mini), constant in dataset size; cascade example: 1,000 items → all proxy-scored, **137 oracle calls** to calibrate, 863 decided by proxy, **$0.42 total** vs full-oracle baseline | cascade `label_budget` default 400; supports `filter`/`resolve`/`equijoin` only; targets must be <1.0 because a target of 1 cannot be certified from finite sampling |
| **FrugalGPT** (arXiv:2305.05176) | match GPT-4 with **up to 98% cost reduction**, or +4% accuracy at equal cost | the canonical cascade result, 2023 |
| **Data-Juicer 2.0** | reordering + fusion saves "at most 70.22%" processing time on complex recipes | combined, not isolated |
| **AutoClean** (NAACL 2025 demo) | LLM agents induce **reusable cleaning rules from limited samples**, then apply them broadly | the "pay for inference once" pattern |
| **TabClean** (arXiv:2606.25388) | compiles LLM reasoning into **guarded Python programs** reused across batches; higher F1 than rule/learning/LLM baselines on 5 of 6 benchmarks, "substantial reductions in recurring runtime and API cost" | guards = dirty-pattern match + target-negative condition + supporting evidence + scope constraints; exact figures not in abstract |

### 5.3 Practical cost anchors for document pipelines

- **Anthropic contextual retrieval** (19 Sep 2024): one-time preprocessing
  **$1.02 per million document tokens** with prompt caching; retrieval failure
  reduction 49% (embeddings+BM25), 67% with reranking.
  <https://www.anthropic.com/news/contextual-retrieval>
- **ExtractBench:** per-page cost **1.0¢ (86.8% F1) / 3.1¢ (89.5%) / 8.1¢
  (95.6%)** for the LlamaExtract tiers; agentic coding agents **16.2¢ (87.1%) /
  27.8¢ (93.6%)**. Framing: "at one million pages, each cent per page equals
  $10,000."
- **Clean Me If You Can:** LLM cleaning ~8h and **~$650 per 1M records**, times
  out at 4.3M.
- **Databricks guidance:** "Run at least 10,000 rows, read the output, then run
  the rest" before scaling any AI Function.
- **Unstructured.io:** $0.001/page (Fast) to $0.01/page (Hi-Res).
  *Vendor pricing page — marketing-adjacent; order-of-magnitude anchor only.*

### 5.4 The documented failure mode for cascades

- **Claim:** confidence-based routing fails under input corruption in two
  distinct ways: (a) the routing signal breaks while the large model would still
  have been right, so escalation never happens; (b) corruption degrades both
  models so escalation no longer restores accuracy. Sequential perturbations
  yield "stable but unreliable predictions" because deferral is suppressed.
- **Source:** *Accuracy and Robustness of Model Cascades Under Data
  Perturbations*, Mitra, Kushwaha, Biessmann, arXiv:2608.17711, 18 Aug 2026,
  GREEN-AI workshop @ ECML-PKDD 2026.
- **Strength:** workshop paper, image classification not text, but the mechanism
  is general. **Implication: a cascade calibrated on a clean sample will
  silently stop escalating on a shifted corpus.**

### Where evidence is thin

Savings percentages are relative to weak baselines (SUPG) and academic datasets.
Nobody publishes "we cut our data-quality review cost from $X to $Y at N million
documents." The closest is Clean Me If You Can's $650/1M and ExtractBench's
¢/page. Latency is almost never reported per stage.

---

## 6. Documented failure modes in practice

### 6.1 Over-detection / over-editing — the best-documented failure, and it predates LLMs

> "The hit rate of editing, that is the proportion of warnings that point to
> true errors, has been estimated to be of the order of **20 to 30%**.
> Conversely, many true errors go undetected."
>
> "There exists a point in time during the editing process when just as many
> errors are introduced as are corrected."

**Source:** Statistics Canada, *Survey Methods and Practices*, ch. "Editing"
(12-539-X),
<https://www150.statcan.gc.ca/n1/pub/12-539-x/2009001/editing-verification-eng.htm>;
corroborated in UNECE *Statistical Data Editing* Vol. 2 (hit rate 20–30%,
editing consumes 20–40% of survey budget).

**Strength:** official statistical-agency methodology, decades of institutional
evidence. **The single most important calibration fact for anyone building an
LLM data-quality detector: in mature human edit systems, ~70–80% of flags are
false alarms. An LLM flagger that beats that is doing well.**

Modern formalization: *False Discovery Rate in Large-Scale Data Error
Localization* (Zenodo 10.5281/zenodo.14002948, 2024) proposes FDR control to
bound over-editing; selective-editing literature uses risk × influence scoring.

### 6.2 Benign anomaly vs true error

- **Claim:** unusual values "may represent either genuine errors or valid
  observations"; repair "requires stronger evidence because a value may be
  unusual and still valid." (arXiv:2608.14765; controlled study, no significance
  tests.)
- **Classical framing:** official statistics distinguishes an *outlier* from an
  *influential error* — "an outlier may not be caused by influential errors, and
  influential errors may not cause outliers" (Istat); statistically flagged
  values are a mixture of "representative outliers and true errors" (JRSS-A
  Euredit editorial, Charlton 2004, DOI 10.1111/j.1467-985x.2004.02051.x).
  **A 40-year-old solved-in-principle problem that LLM papers routinely
  rediscover without citing.**
- **Operational consequence measured:** outlier repair worsens fairness 19.6% vs
  improves it 8.2% (ICDE 2023, §4.3).

### 6.3 Non-determinism

- **Claim/measured:** five API LLMs **configured to be deterministic**, 8 tasks
  × 10 runs: accuracy varied **up to 15% across runs**, with best-vs-worst gaps
  up to **70%**; no model returned consistent outputs or accuracy. Suspected
  cause includes "input buffer packing across multiple jobs."
- **Source:** Atıl et al., *Non-Determinism of "Deterministic" LLM System
  Settings in Hosted Environments*, **Eval4NLP 2025**, pp. 135–148,
  <https://aclanthology.org/2025.eval4nlp-1.12/>
- **Strength:** Peer-reviewed workshop, direct measurement. **Best citation for
  "temperature=0 does not make your data-quality gate reproducible."**
- **Corroborating:** ChatGPT code generation — 75.76% / 51.00% / 47.56% of tasks
  had zero equal test outputs across requests (ACM TOSEM 10.1145/3697010);
  gpt-4o-mini produced distinct outputs in ~24% of exact repeats at temp 0
  (arXiv:2601.19934).
- **Counterweight (task-dependent):** on a narrow binary task the same prompt
  was **99.7% self-consistent** across 5 runs at temp 0.7, and majority voting
  added only **+0.005 F1 for 5× inference cost** (arXiv:2608.18158). So
  non-determinism is real but its impact depends on output-space size.
- **Vendor acknowledgement:** Google explicitly warns LLMs are non-deterministic
  and generated rules "might require adjustment" / will produce false alerts
  (Dataplex docs).

### 6.4 Distribution shift and prompt/format sensitivity

- **Cascade routing breaks under corruption** (§5.4).
- **Tabular format sensitivity:** for every model tested, "at least one-third of
  row decisions changed under some table or question rewrite"; no model was
  robust to table re-organization; decision models were far more robust than
  LLMs to question rewording. *Typed Decision Models on Tabular Data*,
  arXiv:2610.19536, 14 Oct 2026. **Implication: the same row can flip
  clean/dirty depending on how you serialize the table.**
- **Multi-row/distributional errors are the Achilles heel.** LLM+Python agents
  "struggle to detect more complex errors that require understanding data
  distribution across multiple rows"; on Hotel Bookings "all models achieve less
  than 1% improvement" and none found the injected 2016 `lead_time` bias; no
  model closed the gap to clean-data performance. *Exploring LLM Agents for
  Cleaning Tabular ML Datasets*, Bendinelli, Dox, Holz, arXiv:2503.06664,
  **ICLR 2025 Workshop on Foundation Models in the Wild**. Also observed: models
  **submit blindly** (o3-mini 84.7% of tool calls were dataset submission),
  rewrite code from scratch each call, and 7.94–13.53% of submissions were
  invalid. Also arbitrary repairs — Claude found implausible fish consumption in
  landlocked states then **multiplied by an arbitrary 0.3**; o3-mini imputed
  missing country as "Other" instead of the mode.
- **Downstream of cleaning, "quality" itself may not be what the model thinks.**
  *The Data-Quality Illusion* (Apple, arXiv:2510.00866, 1 Oct 2025):
  classifier-based quality filtering improves benchmarks but worsens loss on the
  high-quality set, learns spurious features (shorter sequences scored higher),
  and fails a "data conditioning" ordering test. "Quality filtering is about
  removing the 'bad', not imitating the 'good'."
- **Critique-induced confusion:** critics inject fabricated feedback that
  generators accept; 51.9% of debate-added operations were hallucinated, and
  95.3% of critic feedback items were accepted (arXiv:2606.02866).
- **Iterative self-correction compounds errors:** the survey warns that
  "iterative self-correction cycles can compound hallucinations or introduce new
  errors," and that synthesized rules overfit to corrupted inputs, "treating
  errors as valid" (arXiv:2601.17058).

### 6.5 Silent data loss (document pipelines specifically)

- Gemini 3.5 Flash accuracy **87.9% on short documents → 27.9% on long ones**,
  with "entire records ... dropped rather than misread" — a recall failure no
  field-level error metric catches (ExtractBench).
- InduOCRBench's "OCR Blind Spot": 82.9% OCR accuracy, ~52.8% RAG accuracy.
- Reducto: judges flag correct tables as wrong because they "silently fix typos."

### 6.6 LLM-judge false positives (the evaluator, not the cleaner)

Reducto's benchmark (538 clean + 538 single-error-injected table pairs, 7
frontier judges): strong judges have high false-positive rates on clean pairs —
Claude Opus ~6.5%, GPT-5.4 ~25.5% (figures from the article's charts per search
summary). Remedies: split rubrics, supply reference output, report FPR alongside
detection.

Microsoft Research Universal Verifier: WebVoyager 0.45 FPR, WebJudge ≥22%, vs
0.01–0.08 for the two-pass design.

Snowflake's shipped Groundedness metric is a CoT LLM judge with **no documented
reliability caveat** — a gap worth noting if vendor groundedness scores are to be
used as a data-quality control.

---

## Where the field has no consensus / evidence is thin (consolidated)

1. **"LLMs should propose, not rewrite" is practitioner convergence, not a
   demonstrated result.** No head-to-head experiment exists for cleaning. The
   survey (arXiv:2601.17058) explicitly does not endorse one mode. The evidence
   is: near-zero LLM repair-match (0.028), ~0 F1 for end-to-end LLM cleaners,
   p_r≈0.5 for repair vs ≈1.0 for detection, and 20–30% hit rates in official
   statistics.
2. **No vendor publishes a rule-vs-LLM coverage ratio.** Best numbers are
   LeDQeR's precision/coverage curve and the Austrian Post error-type split.
3. **Span grounding is not solved even where it is enforced.** Best word-level
   grounding F1 is 46.4%; table attribution is near chance on JSON. A valid
   pointer does not mean supporting evidence.
4. **No benchmark scores "finding + evidence span" jointly for data quality**,
   and none exists for document-quality ground truth in a RAG corpus.
5. **Reported F1s are not comparable** across papers (different generators,
   error rates 4–52%, label budgets).
6. **Cost numbers are academic.** $650/1M records and 1–8¢/page are the only
   corpus-scale anchors; no production case study with before/after spend.
7. **Benchmark validity is under active attack** (removal-budget confounding,
   synthetic-vs-real gaps, downstream proxies rewarding noise). Expect published
   cleaning gains to shrink under matched operating points.
8. **Date/venue caveats:** several 2026 arXiv items are preprints with unverified
   venue claims; SEC-FinTables and TABARD per-model figures come from full-text
   search summaries rather than direct reading of the tables; the Reducto FPR
   values are from charts.

## The most load-bearing citations, ranked

1. **Statistics Canada, Editing (12-539-X)** — 20–30% edit hit rate; the
   calibration baseline everyone ignores.
2. **Korom et al., DATA 2026** — the only enterprise field study with
   per-error-type F1 for rules vs LLM and an explicit hybrid recommendation.
3. **Fadlallah, arXiv:2608.14765** — clearest empirical statement that
   deterministic profiling beats LLM detection and LLM repair barely works.
4. **Parmar et al., arXiv:2606.02866** — quantified detect/correct asymmetry
   with a formal condition.
5. **Zeighami/Shankar/Parameswaran, BARGAIN (SIGMOD'26)** — cascades with
   guarantees; the right cost architecture.
6. **Glock et al., arXiv:2609.06053 (LeDQeR)** — generate-filter with the most
   usable precision/coverage trade curve.
7. **Zhang/Huang/Wu, Cocoon (ICDEW 2025)** — hybrid statistical+LLM with the
   strongest tabular F1 deltas.
8. **Atıl et al., Eval4NLP 2025** — non-determinism under "deterministic"
   settings.
9. **Guha et al., ICDE 2023** — over-cleaning measurably harms fairness.
10. **ExtractBench (arXiv:2607.29677)** — grounding is not solved; ¢/page cost
    anchors.

---

## Implication for this repository

Run deterministic validation (schema, types, ranges, functional dependencies,
arithmetic/total reconciliation, checksums, parse completeness) first and let it
own the high-confidence cases; have the LLM emit **claims with exact source
spans** that a deterministic layer can re-locate (Anthropic's `cited_text`
extraction, or LangExtract's `char_interval`-or-null pattern), never free-text
findings; verify claims with an execution/constraint check where one exists;
escalate only the residual to a stronger model or a human, with a precision
target and a measured operating point; and audit with hand-labelled random
samples, because the detector's hit rate will be in the **20–60%** range, not
the 90% range.

Every element of that is supported above — but the **combination has not been
A/B tested in the literature**, which is worth stating plainly rather than
implying consensus exists.

### What this changes about our current design

| Our assumption | What the evidence says |
|---|---|
| "LLM-only can catch broken sentences" | Deterministic profiling **beat** LLM detection (0.561 vs 0.421). Do not assume LLM is the only option. |
| "Requiring a provenance span makes a finding trustworthy" | Span validity ≠ claim support. Word-level grounding F1 is 46.4%. |
| "An audit loop where the LLM reviews rules and revises them" | Critics that also generate compound hallucinations; 51.9% of debate-added operations were fabricated. Rules must be revised through **tests**, not through model judgement. |
| "Precision 0.85+ is the bar" | Human edit systems run at **20–30% hit rate**. 0.85 is not the industry bar; it is an aspiration that must be earned per class. |
| "Cascades are safe once calibrated" | Routing silently stops escalating under distribution shift (arXiv:2608.17711). |
| "temperature=0 gives reproducible gates" | Measured run-to-run accuracy variance up to 15%, gaps up to 70% (Eval4NLP 2025). |

---

# Appendix: design discussion, October 2026

Everything above is literature. This appendix records the conclusions drawn
from it while building this repo, including the parts that are opinion rather
than evidence, and one measurement that has **not been run yet**.

## The open question: should knowledge and DB-bound content be cleaned to different standards?

The proposal was: general knowledge can tolerate some error and still be useful
in RAG; content destined for a database must be strictly verified. Only the
second should be held to a high bar.

### Finding that prompted it: the current gate rejects knowledge outright

Measured directly against the code, not inferred:

```
document: "Photosynthesis converts light energy into chemical energy in plants..."
result:   verified 0  |  quarantined 1  |  reason: unclassified
RAG search for "photosynthesis": []
```

`TemplateExtractor` knows one invoice shape, so ordinary prose becomes
`unclassified` and is withheld. The system does not "strictly clean DB content
while admitting knowledge" — it **strictly rejects both**, and knowledge loses
because it is the larger quantity.

### Proposed distinction (design opinion, not literature)

No standard taxonomy was found for "which document content becomes a DB row and
which is knowledge." That is a domain decision. But the *criterion* can be
stated:

| Question | DB (strict) | Knowledge (lenient) |
|---|---|---|
| Is the value used for aggregation or computation? | yes | no — read and quoted |
| Is a wrong value recoverable? | hard — a bad total propagates silently | easy — a reader notices a strange quoted sentence |
| Can one field be wrong without spoiling the record? | no — one bad field poisons the row | yes — one broken sentence leaves the rest useful |
| **Is it verifiable at all?** | yes — types, ranges, arithmetic, referential integrity | mostly no — meaning and style |

**The asymmetry is the last row.** DB values can be held to a high bar *because
they can be checked*. Knowledge sentences cannot be checked, so the same bar
rejects all of them. A prose sentence that is verbatim in the source may be
wrong, but the pipeline still conveys "this document said this", which is what
retrieval is for. It is quoting, not summarising.

### Gate strictness tiers

**Implemented** in `src/rag_ingest/gate_tier.py` after the measurement below was
taken. The prediction in the next paragraph held: `unclassified` was 81% of a real
646-page manual, so the tier was the next thing to build.

| Tier | Pass condition | On failure |
|---|---|---|
| `STRICT` | grounding + arithmetic + schema + required fields | quarantine |
| `STANDARD` | grounding + required fields | quarantine |
| `LENIENT` | **grounding only** (value is verbatim in the source) | quarantine |

Tier assignment comes from the ontology type the extractor claimed
(`InvoiceHeader` → STRICT, `ReportSection` → STANDARD, `ProseKnowledge` →
LENIENT). **An LLM classifier is the wrong mechanism**: the documented failure
mode is that a misclassified STRICT document silently becomes LENIENT, which is
exactly the silent-corruption path this repo exists to prevent. The mapping is a
lookup table and `assert_tier_applies` raises if a LENIENT bar is applied to a
typed record, so that direction cannot be expressed.

Arithmetic is still not implemented, so STRICT and STANDARD apply identical
checks today. The enum is complete so the arithmetic checks attach to a tier when
they arrive, rather than triggering a silent reclassification later.

This also fixes something the survey flagged: precision targets should be
**per class and per tier**, not one global 0.85. The Statistics Canada 20–30%
hit rate is a statement about *structured* editing, and does not transfer to
knowledge chunks. `ChunkRecord.gate_tier` records which bar produced each verdict
for exactly this reason.

### Not yet measured — do this before building it

The tiers are only worth building if knowledge documents are a meaningful share
of the corpus. **That is unmeasured.** Every validation so far used synthetic
invoices; real documents have never been through the pipeline.

A corpus of ~78 real PDFs (Korean business documents, manuals, reports) was
located on this machine, but processing them was blocked as PII handling, which
is correct. The run should be done deliberately by the owner:

```bash
uv run rag-goldset --corpus /path/to/corpus
```

And the numbers to read, in this order:

| Metric | What it decides |
|---|---|
| `unclassified` share | how much knowledge the gate currently discards — decides whether LENIENT is urgent |
| document-type distribution | invoice-like vs manual vs prose vs scan |
| `extraction_failed` share | how much is damaged rather than merely unknown |
| `disagreement` | whether the second parse earns its cost on this corpus |

**If `unclassified` dominates (say >60%), the tier design is the next thing to
build. If it is small, it is not a priority and the arithmetic checks matter
more.**

### Answer (2026-10-28): 81%, so it was built

Run on FAA AC 43.13-1B (646 pages, 754 chunks, `corpus/README.md`):

| Configuration | verified | unclassified |
|---|---|---|
| before the tier, `TemplateExtractor` | 0 | 613/754 (81%) |
| after the tier, `LayeredExtractor` | 722/754 (96%) |

The 32 remaining quarantines are all `multi_page_span` — chunks that span a page
boundary and therefore cannot be cited through the single-page `ProvenanceRef`
that a LENIENT check requires. That is an honest refusal, not a defect, but it is
the obvious next increment if the number matters.

Two things this measurement did **not** establish, and must not be read into it:

1. **96% is not a quality figure.** LENIENT admits quotation fidelity, which prose
   satisfies almost by construction. It says the text is in the source, not that
   it is useful, current, or correct.
2. **The DB/knowledge split is still not implemented.** All 172 table chunks in
   the manual — including `TABLE 7-1`, the torque specification table — are now
   admitted as `ProseKnowledge`. They are retrievable by text search and quotable
   with a citation, but they are **not rows**: nothing can aggregate them, and the
   `2200-3300*` footnote qualifier is preserved verbatim without being resolved.
   Routing those to STRICT and a database is the part the tier does not do, and
   the arithmetic reconciliation listed as conclusion 1 below is the missing piece
   that would make it meaningful.

## Other design conclusions reached while building

Ordered by how much they would change the system.

**1. Deterministic checks are the highest-value missing piece.**

The survey's strongest result is that deterministic profiling beat LLM detection
(0.561 vs 0.421). This repo has almost no deterministic *semantic* checks — it
verifies "the value is in the source", not "the values are consistent with each
other". Missing, in descending value:

- arithmetic reconciliation (line items vs total)
- valid ranges and date plausibility (`2026-13-45`)
- intra-document contradiction (two different totals; identifier repeated with a
  different amount)
- cross-document consistency, once a store exists

These are cheap, deterministic, require no model, and catch a class of error the
grounding gate cannot see by construction.

**2. `verified` overstates what it means.**

The field means "the value appears in the parser's text buffer where it says it
does". It is read as "the value is correct". A self-consistent OCR error passes.
`grounded` (found in source) and `checked` (passed additional validation) should
be separate fields. Recorded in the README limitations; not yet renamed.

**3. A rule-generation loop needs tests as its gate, not model judgement.**

The proposed loop — LLM writes patterns, deterministic layer applies them
offline, LLM audits the results and revises the patterns — matches published
work (LeDQeR's generate-then-filter; TreeED/ForestED's LLM-authored rules
executed deterministically at inference). The revision step is where it goes
wrong: critics that also generate compound hallucinations (51.9% of debate-added
operations fabricated, 95.3% of critic feedback accepted).

So the audit step should emit **falsification cases**, not revised rules:

```json
{"kind": "rule_false_negative", "rule_id": "...",
 "document": "...", "contradiction": {...}, "evidence": [{"page":1,"span":[..],"text":"..."}],
 "proposed_test": {"input": "...", "expected": "quarantine: arithmetic_mismatch"}}
```

Each case becomes a gold-set entry and a regression test. Rules change only by
passing those tests. This is also where `circuit.py`, `scoring.py` and
`goldset.py` already provide the machinery.

**4. Two failure modes to budget for before starting.**

- **Span grounding is not solved.** Best word-level grounding F1 is 46.4%.
  Requiring a span stops fabricated pointers; it does not stop a valid pointer
  to text that does not support the claim.
- **Non-determinism is real even at temperature 0** — measured run-to-run
  accuracy variance up to 15%, gaps up to 70% (Eval4NLP 2025). Any rule
  generation must therefore be reproducible by *disabling the model at
  inference*, not by hoping the model behaves.

**5. T2 as currently scoped may not survive the corpus measurement.**

T2 adjudicates parser disagreement. Across seven deliberately awkward synthetic
PDFs (two-column, rotated, overlapping, 12-column table, dense small text,
scan-like), Docling's layout score stayed between 0.61 and 0.99 against a 0.60
threshold — **T2 never fired once**. Synthetic vector PDFs render perfectly, so
they cannot reproduce the scan degradation the threshold was set for.

If the real corpus shows the same, T2's cost is not justified in its current
role, and its useful role is probably the one the survey points at: adjudicating
*claims* about data-quality violations rather than *parser readings*.

## What remains unknown

- Whether `LENIENT` grounding-only actually preserves retrieval quality, or lets
  garbage in. Unmeasured.
- How often tier assignment would be wrong, and what that costs.
- Whether any standard taxonomy exists for knowledge-vs-structured content in a
  RAG corpus. Not searched for; the earlier survey only established that no
  corpus-scale ground truth exists.
- Real-corpus `verification`, `unclassified`, and `disagreement` rates. **The
  single most important missing number.**
