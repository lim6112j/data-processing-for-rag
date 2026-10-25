"""Minimal PDF generator shared by tests that need a real file."""

from __future__ import annotations

from pathlib import Path


def make_pdf(path: Path) -> None:
    """Write a one-page PDF with a ruled table and a heading."""
    content = b"""BT /F1 12 Tf 72 720 Td (Quarterly report) Tj ET
0.5 w
72 690 m 400 690 l S
72 670 m 400 670 l S
72 650 m 400 650 l S
72 690 m 72 650 l S
180 690 m 180 650 l S
400 690 m 400 650 l S
BT /F1 10 Tf 78 674 Td (Item) Tj ET
BT /F1 10 Tf 186 674 Td (Amount) Tj ET
BT /F1 10 Tf 78 654 Td (Widget) Tj ET
BT /F1 10 Tf 186 654 Td (1200) Tj ET
"""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n"
        f"{xref_at}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))
