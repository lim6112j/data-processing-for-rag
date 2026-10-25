"""The gate's tests. The central one is `test_schema_valid_but_ungrounded_value_is_rejected`.

The plan's claim: schema validation passes a value that contradicts its source,
and provenance re-derivation is the only check that catches it. That test proves
it with a concrete OCR case.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_ingest.docmodel import Element, ElementKind, ProvenanceRef, RawDocument
from rag_ingest.grounding import GroundingGate, GroundingStatus
from rag_ingest.ontology import (
    AbstainReason,
    FieldAbstention,
    FieldValue,
    InvoiceHeader,
    Unclassified,
)


def _doc(text: str, doc_id: str = "d1") -> tuple[RawDocument, ProvenanceRef]:
    raw, _ = RawDocument.from_elements(
        doc_id,
        [Element(element_id="p1", kind=ElementKind.PARAGRAPH, page=1, text=text)],
    )
    ref = ProvenanceRef(page=1, char_start=0, char_end=len(text), element_id="p1")
    return raw, ref


def _abstain(reason: AbstainReason = AbstainReason.NOT_PRESENT) -> FieldAbstention:
    return FieldAbstention(reason=reason)


def _value(value: str, ref: ProvenanceRef) -> FieldValue:
    return FieldValue(value=value, provenance=ref)


class TestCentralClaim:
    def test_schema_valid_but_ungrounded_value_is_rejected(self):
        # Arrange: OCR read "l,234.56" (lowercase L). The raw source says so.
        raw, ref = _doc("Invoice INV-9 total l,234.56 due")

        # A constrained decoder cannot emit invalid JSON, so it emits a legal
        # Decimal -- here it "corrects" the value into a plausible number that
        # is not what the source says.
        extraction = InvoiceHeader(
            invoice_number=_value("INV-9", ref),
            issue_date=_abstain(),
            total_amount=_value("1334.56", ref),
            currency=_abstain(),
        )

        # Act: schema validation already passed (the model constructed).
        result = GroundingGate(raw).check(extraction)

        # Assert: the gate catches what the schema could not.
        assert result.status is GroundingStatus.MISMATCH
        assert not result.verified
        mismatch_fields = {i.field for i in result.issues if i.status is GroundingStatus.MISMATCH}
        assert mismatch_fields == {"total_amount"}

    def test_numeric_equality_tolerates_formatting_but_not_wrong_value(self):
        raw, ref = _doc("Invoice total 1,234.56 KRW")

        # Two grounded fields: partial extraction is INCOMPLETE, not verified.
        right = InvoiceHeader(
            invoice_number=_abstain(),
            issue_date=_abstain(),
            total_amount=_value("1234.56", ref),
            currency=_value("KRW", ref),
        )
        wrong = InvoiceHeader(
            invoice_number=_abstain(),
            issue_date=_abstain(),
            total_amount=_value("1234.57", ref),
            currency=_value("KRW", ref),
        )

        gate = GroundingGate(raw)
        assert gate.check(right).status is GroundingStatus.GROUNDED
        assert gate.check(wrong).status is GroundingStatus.MISMATCH

    def test_ocr_confusion_match_is_grounded_but_flagged(self):
        # Raw text carries the OCR damage; the value reads through it.
        raw, ref = _doc("Total l,234.56")

        extraction = InvoiceHeader(
            invoice_number=_value("l,234", ref),
            issue_date=_abstain(),
            total_amount=_value("1234.56", ref),
            currency=_abstain(),
        )

        result = GroundingGate(raw).check(extraction)

        assert result.status is GroundingStatus.GROUNDED
        assert result.needed_normalization is True


class TestProvenanceIntegrity:
    def test_value_without_resolvable_span_is_unverifiable_not_passed(self):
        raw, _ = _doc("Some text")

        extraction = InvoiceHeader(
            invoice_number=_value("INV-1", ProvenanceRef(page=9, char_start=0, char_end=5)),
            issue_date=_abstain(),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        result = GroundingGate(raw).check(extraction)

        assert result.status is GroundingStatus.UNVERIFIABLE
        assert not result.verified

    def test_malformed_span_is_unverifiable(self):
        raw, _ = _doc("Some text")

        extraction = InvoiceHeader(
            invoice_number=_value("Some", ProvenanceRef(page=1, char_start=5, char_end=5)),
            issue_date=_abstain(),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        assert GroundingGate(raw).check(extraction).status is GroundingStatus.UNVERIFIABLE

    def test_span_beyond_buffer_is_unverifiable(self):
        raw, _ = _doc("short")

        extraction = InvoiceHeader(
            invoice_number=_value("short", ProvenanceRef(page=1, char_start=0, char_end=9999)),
            issue_date=_abstain(),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        assert GroundingGate(raw).check(extraction).status is GroundingStatus.UNVERIFIABLE

    def test_field_without_provenance_cannot_be_represented(self):
        # The schema makes an ungrounded value unrepresentable.
        with pytest.raises(ValidationError):
            FieldValue(value="INV-1")  # type: ignore[call-arg]


class TestAbstentionAndScope:
    def test_all_abstentions_are_legitimate_but_not_verified(self):
        raw, _ = _doc("nothing useful here")

        extraction = InvoiceHeader(
            invoice_number=_abstain(AbstainReason.ILLEGIBLE),
            issue_date=_abstain(),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        result = GroundingGate(raw).check(extraction)

        assert result.status is GroundingStatus.ABSTAINED
        assert not result.verified

    def test_partial_abstention_can_still_ground(self):
        raw, ref = _doc("Invoice INV-9 total 10.00")

        extraction = InvoiceHeader(
            invoice_number=_value("INV-9", ref),
            issue_date=_abstain(),
            total_amount=_value("10.00", ref),
            currency=_abstain(),
        )

        assert GroundingGate(raw).check(extraction).status is GroundingStatus.GROUNDED

    def test_unclassified_is_not_a_validation_failure(self):
        raw, ref = _doc("A haiku about procurement")

        extraction = Unclassified(observed_shape="poem", provenance=ref)
        result = GroundingGate(raw).check(extraction)

        assert result.status is GroundingStatus.UNCLASSIFIED
        # Legitimate, but not retrievable as verified content.
        assert not result.verified

    def test_unclassified_without_span_is_unverifiable(self):
        raw, _ = _doc("text")

        extraction = Unclassified(
            observed_shape="poem", provenance=ProvenanceRef(page=2, char_start=0, char_end=3)
        )

        assert GroundingGate(raw).check(extraction).status is GroundingStatus.UNVERIFIABLE


class TestNumericParsing:
    def test_currency_symbols_are_stripped(self):
        raw, ref = _doc("Total ₩1,234.56")

        extraction = InvoiceHeader(
            invoice_number=_abstain(),
            issue_date=_abstain(),
            total_amount=_value("1234.56", ref),
            currency=_value("₩", ref),
        )

        assert GroundingGate(raw).check(extraction).status is GroundingStatus.GROUNDED

    def test_non_numeric_claim_does_not_match_numeric_span_by_luck(self):
        raw, ref = _doc("Total 0")

        extraction = InvoiceHeader(
            invoice_number=_value("INV-1", ref),
            issue_date=_abstain(),
            total_amount=_value("99", ref),
            currency=_abstain(),
        )

        assert GroundingGate(raw).check(extraction).status is GroundingStatus.MISMATCH


class TestPartialExtractionIsNotVerification:
    """A found field is not a verified record.

    This was a real bug: an invoice with only its identifier extracted passed
    as GROUNDED and entered retrieval claiming to be a total it did not carry.
    """

    def test_single_grounded_field_is_incomplete(self):
        raw, ref = _doc("Invoice INV-9 total 10.00")

        extraction = InvoiceHeader(
            invoice_number=_value("INV-9", ref),
            issue_date=_abstain(),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        result = GroundingGate(raw).check(extraction)

        assert result.status is GroundingStatus.INCOMPLETE
        assert not result.verified

    def test_two_grounded_fields_verify(self):
        raw, ref = _doc("Invoice INV-9 total 10.00")

        extraction = InvoiceHeader(
            invoice_number=_value("INV-9", ref),
            issue_date=_abstain(),
            total_amount=_value("10.00", ref),
            currency=_abstain(),
        )

        assert GroundingGate(raw).check(extraction).status is GroundingStatus.GROUNDED

    def test_incomplete_is_not_mismatch(self):
        # Nothing contradicted the source; the record is simply thin. Conflating
        # the two would send it down the terminal no-T2 path.
        raw, ref = _doc("Invoice INV-9")

        extraction = InvoiceHeader(
            invoice_number=_value("INV-9", ref),
            issue_date=_abstain(),
            total_amount=_abstain(),
            currency=_abstain(),
        )

        assert GroundingGate(raw).check(extraction).status is not GroundingStatus.MISMATCH
