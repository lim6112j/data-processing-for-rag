"""Build the multi-document Korean business corpus (born-digital).

Run: uv run python corpus/korean/make_corpus.py

Three monthly reports sharing the same inventory table shape across
documents -- the cross-document recurrence discovery is supposed to
surface -- plus one warehouse variant whose row shape differs, keeping
the per-table role separation honest at corpus scale.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT = "Kor"
pdfmetrics.registerFont(TTFont(FONT, "/Library/Fonts/Arial Unicode.ttf"))
OUT_DIR = Path(__file__).parent

MONTHS = [
    ("2026-08", [("IV-001", "육안검사기", "KS-A 12mm", "개", "120", "30"),
                 ("IV-002", "베어링6204", "20x47x14", "개", "2,100", "500"),
                 ("IV-003", "윤활유ISO32", "드럼", "통", "15", "3"),
                 ("IV-004", "볼트M8", "M8x40", "개", "7,900", "1,000")]),
    ("2026-09", [("IV-001", "육안검사기", "KS-A 12mm", "개", "135", "30"),
                 ("IV-002", "베어링6204", "20x47x14", "개", "2,250", "500"),
                 ("IV-003", "윤활유ISO32", "드럼", "통", "14", "3"),
                 ("IV-004", "볼트M8", "M8x40", "개", "8,300", "1,000")]),
    ("2026-10", [("IV-001", "육안검사기", "KS-A 12mm", "개", "150", "30"),
                 ("IV-002", "베어링6204", "20x47x14", "개", "2,400", "500"),
                 ("IV-003", "윤활유ISO32", "드럼", "통", "12", "3"),
                 ("IV-004", "볼트M8", "M8x40", "개", "8,600", "1,000")]),
]

# A different shape: shipment log, one row per outbound order.
SHIPMENTS = [
    ("출고일자", "출고번호", "수량", "납품처"),
    ("2026-10-12", "SH-1001", "40", "부산공장"),
    ("2026-10-14", "SH-1002", "120", "인천공장"),
    ("2026-10-18", "SH-1003", "75", "광주공장"),
]


def _draw_table(c, rows, top, col_widths):
    y = top
    row_h = 20
    for _r, row in enumerate(rows):
        x = 50
        for i, cell in enumerate(row):
            c.setFont(FONT, 10)
            c.drawString(x + 4, y - 14, cell)
            c.rect(x, y - row_h, col_widths[i], row_h)
            x += col_widths[i]
        y -= row_h
    return y


def make_monthly(month, rows, filename):
    path = OUT_DIR / filename
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont(FONT, 14)
    c.drawString(50, 790, f"부품 재고 관리 대장 ({month})")
    c.setFont(FONT, 10.5)
    c.drawString(50, 765, "본 대장은 자재창고의 월간 재고 현황을 기록한다.")
    c.drawString(
        50, 749, "안전재고는 최소 보유 수량이며, 재고가 안전재고 이하로 내려가면 발주를 개시한다."
    )
    _draw_table(c, [list(r) for r in rows], 720, [70, 110, 90, 50, 80, 80])
    c.showPage()
    c.save()
    return path


def make_shipment_log():
    path = OUT_DIR / "shipment-log.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont(FONT, 14)
    c.drawString(50, 790, "출고 기록 (2026년 10월)")
    c.setFont(FONT, 10.5)
    c.drawString(50, 768, "출고 수량은 개 단위로 기록한다.")
    _draw_table(c, [list(r) for r in SHIPMENTS], 740, [90, 90, 60, 90])
    c.showPage()
    c.save()
    return path


if __name__ == "__main__":
    for month, rows in MONTHS:
        print("wrote", make_monthly(month, rows, f"inventory-{month}.pdf"))
    print("wrote", make_shipment_log())
