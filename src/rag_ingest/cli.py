"""Command line entry point.

  uv run rag-goldset goldset/cases.yaml

Runs the gold set and prints the decision-level report. This is the loop you
run before and after any change to thresholds, chunking or the ontology --
it is the reason the thresholds are not vibes.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from rag_ingest.escalation import ReviewQueue
from rag_ingest.goldset import GoldExtractor, build_document, check_case, load_goldset
from rag_ingest.metrics import EscalationReport, build_report
from rag_ingest.pipeline import IngestPipeline


def _run(path: Path, today: date) -> int:
    goldset = load_goldset(path)
    failed_cases: set[str] = set()
    per_class: dict[str, list] = {}
    per_class_chunks: dict[str, int] = {}

    for case in goldset.cases:
        raw, elements = build_document(case)
        outcome = IngestPipeline(
            ReviewQueue(), GoldExtractor(case, raw), today=today
        ).ingest(raw, elements, owner_id="owner-1", retrieval_demand=5)
        hits = [record.text for record in outcome.verified_chunks]
        result = check_case(case, outcome, hits)

        status = "PASS" if result.passed else "FAIL"
        verified = len(outcome.verified_chunks)
        quarantined = len(outcome.quarantined_chunks)
        queued = len(outcome.review_items)
        print(
            f"[{status}] {case.id:<40} "
            f"verified={verified} quarantined={quarantined} queued={queued}"
        )

        if not result.passed:
            failed_cases.add(case.id)
            for failure in result.failures:
                print(f"        - {failure}")

        per_class.setdefault(case.class_name, []).extend(outcome.review_items)
        per_class_chunks[case.class_name] = (
            per_class_chunks.get(case.class_name, 0) + verified + quarantined
        )

    reports = tuple(
        build_report(name, items, chunks=per_class_chunks.get(name, 0))
        for name, items in per_class.items()
    )
    _print_metrics(EscalationReport(classes=reports))

    total = len(goldset.cases)
    print(f"\n{total - len(failed_cases)}/{total} cases passed")
    return 0 if not failed_cases else 1


def _print_metrics(report: EscalationReport) -> None:
    # Caveat that matters: this gold set is deliberately adversarial -- most
    # cases exist to fail. These rates validate *mechanisms*, not production
    # health. Real rates come from live traffic, where a healthy corpus mostly
    # verifies and a class trending up is the signal.
    print("\nEscalation by class (plan section 5b) -- gold set is adversarial:")
    for entry in report.classes:
        flag = "  <-- ontology needs rework" if entry.suspect_ontology else ""
        print(
            f"  {entry.document_class:<12} chunks={entry.chunks:<4} "
            f"rate={entry.escalation_rate:.0%}{flag}"
        )
    print(f"  {'(global)':<12} rate={report.global_rate:.0%}")


def _print_score_card(card) -> None:
    """Report T2's record: precision separately from coverage and harm."""
    print("\nT2 adjudication quality (plan section 5c):")
    print(f"  scored cases : {card.total_scored}")
    print(f"  excluded     : {card.excluded}")
    print(f"  picks        : {card.picks}")
    print(f"  abstentions  : {card.abstentions}")
    precision = "n/a" if card.precision is None else f"{card.precision:.0%}"
    print(f"  precision    : {precision}")
    print(f"  coverage     : {card.coverage:.0%}")
    print(f"  harm rate    : {card.harm_rate:.0%}")
    print(f"  verdict      : {card.verdict()}")


def _score_labels(labels_path: Path) -> int:
    """Score recorded corrections against what T2 chose.

    The labels file is what ``ReviewQueue`` produced -- corrections joined to
    review items. Scoring against owner corrections only; verifier guesses are
    not ground truth.
    """
    import json

    from rag_ingest.escalation import Correction, ReviewItem
    from rag_ingest.scoring import score

    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    corrections = [
        Correction.model_validate(entry) for entry in payload.get("corrections", [])
    ]
    items = {
        entry["chunk_id"]: ReviewItem.model_validate(entry)
        for entry in payload.get("review_items", [])
    }

    card = score(corrections, items)
    _print_score_card(card)

    # Show what the breaker would do with this evidence, using the same
    # thresholds production uses. The report should answer "would T2 be
    # stopped right now?" without anyone having to reason it out from the rates.
    from rag_ingest.circuit import T2CircuitBreaker
    from rag_ingest.scoring import feed_breaker

    breaker = T2CircuitBreaker(inner=None)
    feed_breaker(card, breaker)
    _print_breaker_status(breaker.status)
    return 0


def _print_breaker_status(status) -> None:
    print("\nCircuit breaker:")
    print(f"  state        : {status.state.value}")
    if status.trip_reason.value != "none":
        print(f"  tripped by   : {status.trip_reason.value}")
    print(f"  labelled     : {status.labels_seen}")


def _ingest_corpus(
    directory: Path,
    *,
    use_second_parse: bool,
    owner_id: str,
    today: date,
    parser: str = "auto",
    extract: bool = True,
) -> int:
    """Parse a directory of PDFs and run them through the cascade.

    ``parser`` selects the primary stream:

    * ``auto`` -- Docling when installed, pdfplumber otherwise.
    * ``docling`` -- layout model + TableFormer. Handles scans and complex
      layouts; can invent a grid.
    * ``pdfplumber`` -- raw text layer + geometric table detection. Faithful to
      digital text; blind to anything unruled.

    With ``use_second_parse`` the *other* parser becomes the alternate stream,
    which is the only pairing that produces genuinely informative disagreements.
    """
    from rag_ingest.adapters.openrouter import OpenRouterAdjudicator
    from rag_ingest.extractors import TemplateExtractor
    from rag_ingest.ingest_runner import DocumentSource, IngestRunner, describe

    primary_name, primary_parse = _select_parser(parser)
    if primary_parse is None:
        print(
            "error: docling is not installed; run `uv add docling` or pass "
            "--parser pdfplumber",
            file=sys.stderr,
        )
        return 2

    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".pdf")
    if not paths:
        print(f"error: no PDFs in {directory}", file=sys.stderr)
        return 2

    #: Alternates per document, consumed by the producer for that doc only.
    alternates: dict[str, object] = {}
    sources: list[DocumentSource] = []
    parse_failures = 0

    for path in paths:
        try:
            raw, elements = primary_parse(str(path), doc_id=path.stem)
        except Exception as exc:  # one bad file must not stop the scan
            print(f"  [parse-failed] {path.name}: {type(exc).__name__}", file=sys.stderr)
            parse_failures += 1
            continue

        if use_second_parse:
            try:
                alternates[path.stem] = _second_parse_for(str(path), primary_name)
            except Exception as exc:  # alternate is optional
                print(
                    f"  [second-parse-failed] {path.name}: {type(exc).__name__}",
                    file=sys.stderr,
                )

        sources.append(
            DocumentSource(
                doc_id=path.stem, path=str(path), raw=raw, elements=tuple(elements)
            )
        )

    producer = (
        _PerDocumentProducer(primary_name=primary_name, alternates=alternates)
        if use_second_parse
        else None
    )

    runner = IngestRunner(
        extractor=TemplateExtractor() if extract else _DisabledExtraction(),
        adjudicator=OpenRouterAdjudicator(),
        producer=producer,
        today=today,
    )
    report = runner.run(sources, owner_id=owner_id)

    print(f"primary parser : {primary_name}")
    if use_second_parse:
        alternate_name = _other_parser_name(primary_name)
        print(f"alternate      : {alternate_name} ({len(alternates)} aligned)")
    print(describe(report))
    if parse_failures:
        print(f"parse failures : {parse_failures}")
    return 0 if not report.failed else 1


class _DisabledExtraction:
    """No extraction: every chunk becomes unclassified. Useful for parser-only runs."""

    def extract(self, chunk):
        return None


def _select_parser(parser: str):
    """Resolve the primary parse function, or (name, None) when unavailable."""
    name = parser.lower()
    if name in ("auto", "docling"):
        try:
            from rag_ingest.adapters.docling import pdf_to_raw as docling_to_raw
        except ImportError:
            if name == "docling":
                return "docling", None
        else:
            return "docling", docling_to_raw

    from rag_ingest.adapters.pdfplumber import pdf_to_raw as plumber_to_raw

    return "pdfplumber", plumber_to_raw


def _other_parser_name(primary: str) -> str:
    return "pdfplumber" if primary == "docling" else "docling"


def _second_parse_for(path: str, primary_name: str):
    """Parse with the parser that is *not* primary, for cross-origin alignment.

    Returns a ``SecondParse``; imported lazily because the adapters it needs are
    optional dependencies.
    """
    from rag_ingest.candidates import SecondParse

    if primary_name == "docling":
        from rag_ingest.adapters.pdfplumber import pdf_to_raw as plumber_to_raw

        _, elements = plumber_to_raw(path, doc_id="alternate", render_images=False)
        return SecondParse(name="pdfplumber", elements=tuple(elements))

    from rag_ingest.adapters.docling import pdf_to_raw as docling_to_raw

    # No images needed for an alternate: it supplies text, not evidence.
    _, elements = docling_to_raw(path, doc_id="alternate", render_images=False)
    return SecondParse(name="docling", elements=tuple(elements))


class _PerDocumentProducer:
    """Dispatches to the alternate parse belonging to the chunk's document.

    A single shared producer would align every chunk against one document's
    second parse, silently mismatching whole documents whenever their page
    counts or layouts differ.
    """

    def __init__(self, *, primary_name: str, alternates: dict) -> None:
        self._primary_name = primary_name
        self._alternates = alternates

    def produce(self, chunk):
        from rag_ingest.candidates import AlternateParseProducer

        alternate = self._alternates.get(chunk.doc_id)
        if alternate is None:
            return []
        return AlternateParseProducer(
            primary_name=self._primary_name, alternate=alternate
        ).produce(chunk)




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-goldset", description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="goldset/cases.yaml",
        type=Path,
        help="path to the gold-set YAML (default: goldset/cases.yaml)",
    )
    parser.add_argument(
        "--score",
        type=Path,
        metavar="LABELS.json",
        help="score T2 against exported review corrections instead of running the gold set",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        metavar="DIR",
        help="ingest every PDF in DIR and report per-document accounting",
    )
    parser.add_argument(
        "--second-parse",
        action="store_true",
        help="with --corpus, enable the alternate-parse producer (bbox alignment)",
    )
    parser.add_argument(
        "--parser",
        default="auto",
        choices=["auto", "docling", "pdfplumber"],
        help="primary parse stream for --corpus (default: auto = docling if installed)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="with --corpus, skip extraction so every chunk becomes unclassified",
    )
    parser.add_argument(
        "--owner",
        default="owner-1",
        help="document owner id recorded on review items (default: owner-1)",
    )
    args = parser.parse_args(argv)

    if args.score is not None:
        if not args.score.exists():
            print(f"error: labels file not found at {args.score}", file=sys.stderr)
            return 2
        return _score_labels(args.score)

    if args.corpus is not None:
        if not args.corpus.is_dir():
            print(f"error: corpus directory not found at {args.corpus}", file=sys.stderr)
            return 2
        return _ingest_corpus(
            args.corpus,
            use_second_parse=args.second_parse,
            owner_id=args.owner,
            today=date.today(),
            parser=args.parser,
            extract=not args.no_extract,
        )

    if not args.path.exists():
        print(f"error: gold set not found at {args.path}", file=sys.stderr)
        return 2

    return _run(args.path, date.today())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
