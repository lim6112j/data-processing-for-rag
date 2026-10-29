"""pdfplumber's geometric table finder boxes plain text lines as one-column
"tables" (FAA safety-tip lists do this). A one-column table is not a table:
it has no column structure to preserve, and routing it as rows starves both
discovery (32 junk row:text clusters) and the prose path that could have
quoted it."""

from rag_ingest.adapters.pdfplumber import page_to_elements


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def bbox(self):  # pdfplumber Table.bbox()
        return (0.0, 0.0, 100.0, 100.0)


class _FakeFinder:
    def __init__(self, tables):
        self._tables = tables

    def __getitem__(self, index):
        return self._tables[index]


class _FakePage:
    """Duck-typed pdfplumber page: only the surface page_to_elements uses."""

    def __init__(self, tables, text):
        self._tables = tables
        self._text = text

    def extract_tables(self):
        return self._tables

    def find_tables(self):
        return _FakeFinder([_FakeTable(()) for _ in self._tables])

    def extract_text(self):
        return self._text

    def extract_words(self):
        return [
            {"x0": 10.0, "top": 10.0, "x1": 90.0, "bottom": 90.0},
        ]


class TestOneColumnTablesAreProse:
    def test_single_column_table_is_skipped(self):
        page = _FakePage(
            tables=[[["SAFETY TIPS."], ["Always ground the aircraft."]]],
            text="SAFETY TIPS.\nAlways ground the aircraft.",
        )
        elements = page_to_elements(page, page_no=3)

        assert [e.kind.name for e in elements] == ["PARAGRAPH"]

    def test_real_two_column_table_survives(self):
        page = _FakePage(
            tables=[[["부서", "상태"], ["품질팀", "진행중"]]],
            text="부서 상태",
        )
        elements = page_to_elements(page, page_no=1)

        assert [e.kind.name for e in elements] == ["TABLE", "PARAGRAPH"]
        assert elements[0].table_rows == (("부서", "상태"), ("품질팀", "진행중"))
