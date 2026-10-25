"""Plan section 5b: per-class escalation rate is the health signal."""

from __future__ import annotations

from datetime import date

from rag_ingest.escalation import EscalationReason, ReviewItem
from rag_ingest.metrics import EscalationReport, build_report


def _item(reason: EscalationReason) -> ReviewItem:
    return ReviewItem(
        item_id=f"i-{reason}",
        doc_id="d1",
        chunk_id="c1",
        reason=reason,
        question="q",
        owner_id="o",
        created_on=date(2026, 10, 25),
    )


def test_global_rate_can_hide_a_bad_class():
    # Global 10% looks acceptable; the contracts class is at 100%.
    healthy = build_report("invoices", [], chunks=90)
    broken = build_report(
        "contracts", [_item(EscalationReason.UNCLASSIFIED)] * 10, chunks=10
    )
    report = EscalationReport(classes=(healthy, broken))

    assert report.global_rate == 0.1
    assert broken.escalation_rate == 1.0
    assert "contracts" in report.classes_to_rework


def test_class_with_mostly_schema_failures_is_flagged_for_ontology_rework():
    report = build_report(
        "contracts",
        [_item(EscalationReason.UNCLASSIFIED), _item(EscalationReason.SCHEMA_MISMATCH)],
        chunks=10,
    )

    assert report.suspect_ontology is True


def test_class_failing_on_grounding_is_not_an_ontology_problem():
    # Grounding failures are a parser/OCR problem, not a schema problem -- the
    # ontology is fine, the extraction is lying.
    report = build_report(
        "invoices", [_item(EscalationReason.UNGROUNDED)] * 5, chunks=10
    )

    assert report.suspect_ontology is False


def test_no_escalations_is_never_flagged():
    assert build_report("invoices", [], chunks=10).suspect_ontology is False


def test_zero_chunk_class_does_not_divide_by_zero():
    report = build_report("empty", [], chunks=0)

    assert report.escalation_rate == 0.0
    assert EscalationReport(classes=(report,)).global_rate == 0.0


def test_reason_breakdown_is_counted():
    report = build_report(
        "x",
        [_item(EscalationReason.UNGROUNDED), _item(EscalationReason.UNGROUNDED)],
        chunks=5,
    )

    assert report.by_reason[EscalationReason.UNGROUNDED] == 2
