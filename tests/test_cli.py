"""The CLI is the loop you run before and after changing any threshold."""

from __future__ import annotations

from pathlib import Path

from rag_ingest.cli import main

GOLDSET = Path(__file__).parent.parent / "goldset" / "cases.yaml"


def test_default_goldset_passes(capsys):
    exit_code = main([str(GOLDSET)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "6/6 cases passed" in captured.out
    assert "PASS" in captured.out


def test_missing_file_returns_usage_error(capsys):
    exit_code = main(["/nonexistent/cases.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "gold set not found" in captured.err


def test_failing_case_returns_nonzero(tmp_path, capsys):
    # A gold set whose expectation is deliberately wrong must fail loudly --
    # otherwise a regression could pass unnoticed.
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        """
cases:
  - id: deliberately-wrong
    description: expects a verified chunk from an unreadable page
    class: invoice
    scores: { ocr: 0.1, layout: 0.2, parse: 0.1 }
    pages:
      - "illegible"
    expected:
      verified: 1
""",
        encoding="utf-8",
    )

    exit_code = main([str(broken)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL" in captured.out
    assert "0/1 cases passed" in captured.out


def test_metrics_are_reported_per_class(capsys):
    main([str(GOLDSET)])

    captured = capsys.readouterr()
    assert "Escalation by class" in captured.out
    assert "invoice" in captured.out
    assert "adversarial" in captured.out


def test_ontology_rework_flag_only_for_schema_shaped_failures(capsys):
    main([str(GOLDSET)])

    captured = capsys.readouterr()
    # The unknown class fails entirely on unclassified content, so its ontology
    # is the problem. The invoice class fails on OCR/grounding, which is not.
    assert "unknown" in captured.out
    lines = [line for line in captured.out.splitlines() if "ontology needs rework" in line]
    assert any("unknown" in line for line in lines)
    assert not any("invoice" in line for line in lines)


def test_score_mode_reads_labels_and_reports(tmp_path, capsys):
    import json

    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "corrections": [
                    {
                        "doc_id": "d1",
                        "chunk_id": "c1",
                        "field": "structure",
                        "validator_error": "structure",
                        "before": None,
                        "after": "c1:pdfplumber",
                        "reviewer_id": "owner",
                        "reviewer_role": "owner",
                        "captured_on": "2026-10-25",
                        "adjudication_choice": "c1:pdfplumber",
                    }
                ],
                "review_items": [
                    {
                        "item_id": "i1",
                        "doc_id": "d1",
                        "chunk_id": "c1",
                        "reason": "structure_ambiguous",
                        "question": "q",
                        "owner_id": "owner",
                        "created_on": "2026-10-25",
                        "candidate_ids": ["c1:docling", "c1:pdfplumber"],
                        "adjudicated_choice": "c1:pdfplumber",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--score", str(labels)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "T2 adjudication quality" in out
    assert "precision" in out
    assert "reliable" in out


def test_score_mode_missing_file_is_a_usage_error(capsys):
    exit_code = main(["--score", "/nonexistent/labels.json"])

    assert exit_code == 2
    assert "labels file not found" in capsys.readouterr().err


def test_score_report_shows_breaker_state(tmp_path, capsys):
    import json

    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "corrections": [
                    {
                        "doc_id": "d1",
                        "chunk_id": "c1",
                        "field": "s",
                        "validator_error": "s",
                        "before": None,
                        "after": "c1:b",
                        "reviewer_id": "o",
                        "reviewer_role": "owner",
                        "captured_on": "2026-10-25",
                        "adjudication_choice": "c1:a",
                    }
                ],
                "review_items": [
                    {
                        "item_id": "i1",
                        "doc_id": "d1",
                        "chunk_id": "c1",
                        "reason": "structure_ambiguous",
                        "question": "q",
                        "owner_id": "o",
                        "created_on": "2026-10-25",
                        "candidate_ids": ["c1:a", "c1:b"],
                        "adjudicated_choice": "c1:a",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    main(["--score", str(labels)])

    out = capsys.readouterr().out
    assert "Circuit breaker:" in out
    assert "closed" in out


def test_tripping_breaker_is_reported(tmp_path, capsys):
    import json

    corrections = []
    items = []
    for i in range(12):
        cid = f"c{i}"
        corrections.append(
            {
                "doc_id": "d",
                "chunk_id": cid,
                "field": "s",
                "validator_error": "s",
                "before": None,
                "after": f"{cid}:b",
                "reviewer_id": "o",
                "reviewer_role": "owner",
                "captured_on": "2026-10-25",
                "adjudication_choice": f"{cid}:a",
            }
        )
        items.append(
            {
                "item_id": f"i{i}",
                "doc_id": "d",
                "chunk_id": cid,
                "reason": "structure_ambiguous",
                "question": "q",
                "owner_id": "o",
                "created_on": "2026-10-25",
                "candidate_ids": [f"{cid}:a", f"{cid}:b"],
                "adjudicated_choice": f"{cid}:a",
            }
        )
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"corrections": corrections, "review_items": items}), encoding="utf-8"
    )

    main(["--score", str(path)])

    out = capsys.readouterr().out
    assert "open" in out
    assert "low_precision" in out


def test_corpus_mode_requires_a_directory(capsys):
    exit_code = main(["--corpus", "/nonexistent/dir"])

    assert exit_code == 2
    assert "corpus directory not found" in capsys.readouterr().err


def test_corpus_mode_with_no_pdfs_is_a_usage_error(tmp_path, capsys):
    exit_code = main(["--corpus", str(tmp_path)])

    assert exit_code == 2
    assert "no PDFs" in capsys.readouterr().err


def test_corpus_mode_ingests_real_pdfs(tmp_path, capsys):
    from tests.test_second_parse_e2e import _make_pdf

    _make_pdf(tmp_path / "a.pdf")
    _make_pdf(tmp_path / "b.pdf")

    exit_code = main(["--corpus", str(tmp_path)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "documents      : 2" in out
    assert "disagreement" in out


def test_corpus_mode_isolates_a_corrupt_file(tmp_path, capsys):
    from tests.test_second_parse_e2e import _make_pdf

    _make_pdf(tmp_path / "good.pdf")
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")

    exit_code = main(["--corpus", str(tmp_path)])

    out = capsys.readouterr().out
    # The good file still processed; the bad one was reported and skipped.
    assert "documents      : 1" in out
    assert exit_code == 0
