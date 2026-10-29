"""Build the Korean business corpus PDF (born-digital, real text layer).

Run: uv run python corpus/korean/make_pdf.py

Purpose: the FAA corpus is numeric-dense English; this document is the
untested territory -- Korean headers, composite specifications, a closed
status set, and prose paragraphs around the tables. Everything a Korean
business document actually looks like.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT = "Kor"
pdfmetrics.registerFont(TTFont(FONT, "/Library/Fonts/Arial Unicode.ttf"))

OUT = Path(__file__).parent / "korean-business-sample.pdf"

INVENTORY = [
    ["품목코드", "품목명", "규격", "단위", "재고", "안전재고"],
    ["IV-001", "육안검사기", "KS-A 12mm", "개", "150", "30"],
    ["IV-002", "베어링6204", "20x47x14", "개", "2,400", "500"],
    ["IV-003", "윤활유ISO32", "드럼", "통", "12", "3"],
    ["IV-004", "볼트M8", "M8x40", "개", "8,600", "1,000"],
    ["IV-005", "필터HF-35", "HF-35", "개", "44", "10"],
    ["IV-006", "스패너12호", "12mm", "개", "75", "20"],
]

INSPECTION = [
    ["검사일자", "부품코드", "상태", "불량수", "메모"],
    ["2026-10-02", "AN310-4", "PASS", "0", "정상"],
    ["2026-10-02", "MS20365-3", "FAIL", "3", "토크부족"],
    ["2026-10-03", "NAS1223", "HOLD", "0", "재검사대기"],
    ["2026-10-04", "AN310-5", "PASS", "0", "정상"],
    ["2026-10-04", "MS20365-4", "FAIL", "12", "나사손상"],
    ["2026-10-05", "NAS1224", "PASS", "0", "정상"],
]


def draw_table(c, rows, top, col_widths):
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


c = canvas.Canvas(str(OUT), pagesize=A4)
width, _height = A4

# -- page 1: intro prose + inventory table -----------------------------------
c.setFont(FONT, 14)
c.drawString(50, 790, "부품 재고 관리 대장 (2026년 10월)")
c.setFont(FONT, 10.5)
for i, line in enumerate([
    "본 대장은 자재창고의 월간 재고 현황을 기록한다.",
    "안전재고는 최소 보유 수량이며, 재고가 안전재고 이하로 내려가면 발주를 개시한다.",
    "규격 표기는 한국산업규격(KS) 및 제조사 표기를 혼용한다.",
]):
    c.drawString(50, 760 - 16 * i, line)
draw_table(c, INVENTORY, 690, [70, 110, 90, 50, 80, 80])
c.setFont(FONT, 10.5)
c.drawString(50, 540, "비고: IV-002와 IV-004는 금월 대량 사용 품목이다.")
c.showPage()

# -- page 2: inspection table + closing prose --------------------------------
c.setFont(FONT, 14)
c.drawString(50, 790, "부품 검사 기록")
c.setFont(FONT, 10.5)
c.drawString(50, 770, "상태는 PASS(합격), FAIL(불합격), HOLD(보류) 중 하나로 기록한다.")
draw_table(c, INSPECTION, 740, [90, 100, 60, 60, 100])
c.setFont(FONT, 10.5)
for i, line in enumerate([
    "불합격 판정이 된 부품은 즉시 격리된다.",
    "토크부족은 규정 조임 토크 미달을 의미하며 재조임 후 재검사한다.",
    "보류 상태의 부품은 판정 전까지 출고할 수 없다.",
    "본 기록은 품질관리팀이 작성하며 월말에 품질보고서로 종합한다.",
]):
    c.drawString(50, 560 - 16 * i, line)
c.showPage()
c.save()
print(f"wrote {OUT}")
