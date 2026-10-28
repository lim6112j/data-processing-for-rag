"""Mechanical defect marking: see the damage, never fix it.

The cleaning policy this module serves was decided, and the literature supports
it. survey.md §1: the best LLM repair configuration matched a reference only
0.028 of the time, while the best detector (deterministic profiling, no model)
reached 0.561. survey.md §6.1: in mature human edit systems only 20-30% of edit
warnings point at true errors, and "there exists a point in time during the
editing process when just as many errors are introduced as are corrected".

So damaged text is admitted -- it is what the source says, and a quotation is
the one thing this pipeline can vouch for -- but it is *marked*. An unmarked
defect does not disappear: it moves to the consumer, where an answering model
will silently correct it with no record, no provenance and no way to audit.
A mark keeps the correction decision with a human.

No model participates. Same rule as everywhere in T0/T1.
"""

from __future__ import annotations

import re

from rag_ingest.ontology import DefectCode

#: Text that looks like it *wants* to be an amount but is damaged -- e.g. OCR
#: read a lowercase l for the leading 1 ("l,234.56"). Shared with the extractor,
#: which needs the same pattern for the opposite purpose (refusing to pretend an
#: unreadable amount was absent).
DAMAGED_AMOUNT = re.compile(r"[lIO|][,.]\d{3}|\d[,.]\d{3}[,.][lIO]")

#: Each rule maps a text pattern to the mark it earns. Closed by construction:
#: an unrecognised pattern produces no mark, and the enum is closed so the queue
#: and the store can treat these as triageable facts rather than free text.
_RULES: tuple[tuple[re.Pattern[str], DefectCode], ...] = (
    (DAMAGED_AMOUNT, DefectCode.SUSPECTED_OCR_DAMAGE),
)


def scan_defects(text: str) -> tuple[DefectCode, ...]:
    """The defects visible in this text, in rule order.

    Deliberately returns *what was seen*, not a verdict: the caller decides what
    a mark means for admission. This module only makes sure the mark exists
    before the text enters retrieval.
    """
    return tuple(code for pattern, code in _RULES if pattern.search(text))


__all__ = ["DAMAGED_AMOUNT", "scan_defects"]
