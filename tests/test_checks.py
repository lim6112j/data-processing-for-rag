"""Deterministic semantic checks: the values agree with each other.

survey.md conclusion 1: grounding verifies "the value is in the source", which
a self-consistent OCR error and an implausible date both pass by construction.
The checks here are the missing class -- cheap, model-free, and invisible to
the grounding gate. They run AFTER the gate: a value must be provably in the
source before asking whether it is plausible.
"""

from __future__ import annotations

from rag_ingest.checks import check_extraction
from rag_ingest.docmodel import ProvenanceRef
from rag_ingest.ontology import (
    AbstainReason,
    FieldAbstention,
    FieldValue,
    InvoiceHeader,
    LineItem,
    ReportSection,
)

REF = ProvenanceRef(page=1, char_start=0, char_end=10, element_id="e")


def _value(value: str) -> FieldValue:
    return FieldValue(value=value, provenance=REF)


def _abstain() -> FieldAbstention:
    return FieldAbstention(reason=AbstainReason.NOT_PRESENT)


class TestDatePlausibility:
    def test_impossible_month_is_flagged(self):
        extraction = InvoiceHeader(
            invoice_number=_value("INV-1"),
            issue_date=_value("2026-13-45"),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        failures = check_extraction(extraction)

        assert len(failures) == 1
        assert failures[0].field == "issue_date"
        assert failures[0].check_name == "date_plausibility"

    def test_impossible_feb_30_is_flagged(self):
        extraction = InvoiceHeader(
            invoice_number=_value("INV-1"),
            issue_date=_value("2026-02-30"),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        failures = check_extraction(extraction)

        assert failures[0].check_name == "date_plausibility"

    def test_valid_date_passes(self):
        extraction = InvoiceHeader(
            invoice_number=_value("INV-1"),
            issue_date=_value("2026-02-28"),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        assert check_extraction(extraction) == ()

    def test_leap_day_2024_passes(self):
        extraction = InvoiceHeader(
            invoice_number=_value("INV-1"),
            issue_date=_value("2024-02-29"),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        assert check_extraction(extraction) == ()

    def test_non_leap_century_feb_29_is_flagged(self):
        extraction = InvoiceHeader(
            invoice_number=_value("INV-1"),
            issue_date=_value("1900-02-29"),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        failures = check_extraction(extraction)

        assert failures, "1900-02-29 is not a leap day"
        assert failures[0].check_name == "date_plausibility"

    def test_ocr_swapped_digits_are_flagged(self):
        # The plausible-looking failure: 2026-05-96 OCRs from 2026-05-96's
        # neighbour digit. Grounding passes (the string is verbatim); only the
        # semantic check can see it.
        extraction = InvoiceHeader(
            invoice_number=_value("INV-1"),
            issue_date=_value("2026-05-96"),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        assert check_extraction(extraction)

    def test_non_iso_shape_is_ignored_not_failed(self):
        # The extractor only emits ISO dates, but a check must not guess what
        # an unparseable value *means*. Suspicion without a parse is silence.
        extraction = InvoiceHeader(
            invoice_number=_value("INV-1"),
            issue_date=_value("May 96"),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        assert check_extraction(extraction) == ()

    def test_abstention_is_not_checked(self):
        extraction = InvoiceHeader(
            invoice_number=_value("INV-1"),
            issue_date=_abstain(),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        assert check_extraction(extraction) == ()


class TestCheckAppliesToReportSection:
    def test_report_finding_date_shape_is_checked(self):
        extraction = ReportSection(
            section_title=_value("Findings"),
            primary_finding=_value("2026-13-45"),
        )

        failures = check_extraction(extraction)

        assert failures
        assert failures[0].field == "primary_finding"


class TestArithmeticReconciliation:
    """survey.md conclusion 1, ranked first: line items vs total.

    The rule for running at all: reconciliation fires ONLY when every line
    item parses. Summing a subset and calling the mismatch a finding would be
    the check hallucinating completeness -- the same discipline as the date
    check's shape guard.
    """

    def _header(self, total: str, *line_amounts: str) -> InvoiceHeader:
        return InvoiceHeader(
            invoice_number=_value("INV-1"),
            issue_date=_abstain(),
            total_amount=_value(total),
            currency=_abstain(),
            line_items=tuple(LineItem(amount=_value(a)) for a in line_amounts),
        )

    def test_lines_that_do_not_sum_to_total_are_flagged(self):
        extraction = self._header("1234.56", "100.00", "200.00")

        failures = check_extraction(extraction)

        assert failures, "100 + 200 != 1234.56"
        assert failures[0].check_name == "arithmetic_reconciliation"
        assert failures[0].field == "total_amount"

    def test_lines_that_sum_exactly_pass(self):
        extraction = self._header("300.00", "100.00", "200.00")

        assert check_extraction(extraction) == ()

    def test_rounding_within_a_cent_passes(self):
        extraction = self._header("300.01", "100.00", "200.005")

        assert check_extraction(extraction) == ()

    def test_unparseable_line_silences_the_check(self):
        # One damaged line ("l,234.56") means the sum is unknown. Flagging
        # anyway would quarantine every invoice with one bad line -- the
        # review queue is for questions with answers, not noise.
        extraction = InvoiceHeader(
            invoice_number=_value("INV-1"),
            issue_date=_abstain(),
            total_amount=_value("1234.56"),
            currency=_abstain(),
            line_items=(LineItem(amount=_value("l,234.56")),),
        )

        assert check_extraction(extraction) == ()

    def test_no_lines_means_nothing_to_reconcile(self):
        extraction = self._header("1234.56")

        assert check_extraction(extraction) == ()

    def test_abstained_total_with_parseable_lines_stays_silent(self):
        extraction = InvoiceHeader(
            invoice_number=_value("INV-1"),
            issue_date=_abstain(),
            total_amount=_abstain(),
            currency=_abstain(),
            line_items=(LineItem(amount=_value("100.00")),),
        )

        assert check_extraction(extraction) == ()

    def test_unparseable_total_also_stays_silent(self):
        extraction = InvoiceHeader(
            invoice_number=_value("INV-1"),
            issue_date=_abstain(),
            total_amount=_value("TBD"),
            currency=_abstain(),
            line_items=(LineItem(amount=_value("100.00")),),
        )

        assert check_extraction(extraction) == ()
