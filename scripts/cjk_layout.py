"""CJK paragraph defaults and PDF character-spacing diagnostics.

Coordinates are PDF points in pdfplumber's (left, top, right, bottom) system.
Findings are candidates for visual review, never proof of semantic correctness.
"""
from __future__ import annotations

from collections import Counter
import re
from statistics import median


def make_cjk_styles(regular_font: str, bold_font: str) -> dict:
    """Call after registering static regular/bold fonts with ReportLab.

    Uses character-aware wrapping with no word-space justification. Actual code
    should use a separate preformatted flowable, not this prose formatter.
    """
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics

    for name in (regular_font, bold_font):
        pdfmetrics.getFont(name)  # Fail immediately on missing registrations.
    pdfmetrics.registerFontFamily(regular_font, normal=regular_font, bold=bold_font,
                                  italic=regular_font, boldItalic=bold_font)
    body = ParagraphStyle("CJKBody", fontName=regular_font, fontSize=10.5,
                          leading=17, alignment=TA_LEFT, wordWrap="CJK",
                          splitLongWords=1, spaceAfter=6,
                          allowWidows=0, allowOrphans=0)
    return {
        "body": body,
        "cell": ParagraphStyle("CJKCell", parent=body, fontSize=9, leading=14),
        "caption": ParagraphStyle("CJKCaption", parent=body, fontSize=9, leading=14),
        "prompt": ParagraphStyle("CJKPrompt", parent=body, leftIndent=7, rightIndent=7),
        "bullet": ParagraphStyle("CJKBullet", parent=body, leftIndent=14,
                                 firstLineIndent=0, bulletIndent=2),
    }


def is_cjk(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in text)


def inside(char: dict, bbox) -> bool:
    x, y = (char["x0"] + char["x1"]) / 2, (char["top"] + char["bottom"]) / 2
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def spacing_findings(chars: list[dict], page_width: float, page_height: float,
                     table_boxes=(), exclusions=()) -> list[dict]:
    """Find oversized horizontal gaps in CJK lines outside ruled tables.

    Uses non-space glyph edges, not extracted whitespace counts. Font-relative
    and page-relative thresholds avoid flagging ordinary word spaces. Table
    cells are checked individually so gutters are not mistaken for bad spacing.
    """
    visible = [c for c in chars if c.get("text", "").strip()
               and .04 * page_height <= c["top"] <= .94 * page_height
               and not any(inside(c, b) for b in table_boxes)
               and not any(inside(c, b) for b in exclusions)]
    lines = []
    for char in sorted(visible, key=lambda c: (c["top"], c["x0"])):
        # Superscripts form their own small-font lines and are not gap anchors.
        if lines and abs(char["top"] - lines[-1][0]["top"]) <= 1.5:
            lines[-1].append(char)
        else:
            lines.append([char])
    findings = []
    for line in lines:
        if sum(is_cjk(c["text"]) for c in line) < 8:
            continue
        em = median(c["size"] for c in line)
        line = sorted((c for c in line if c["size"] >= .85 * em), key=lambda c: c["x0"])
        gaps = []
        for left, right in zip(line, line[1:]):
            gap = right["x0"] - left["x1"]
            if gap > max(3 * em, .035 * page_width):
                gaps.append({"gap_pt": round(gap, 2), "gap_em": round(gap / em, 2),
                             "between": left["text"] + " | " + right["text"]})
        # Short mixed lines also matter: the reported dataset-name line has
        # only one Chinese character. Handled separately below.
        if gaps:
            findings.append({"kind": "excessive-line-gap",
                             "bbox": [min(c["x0"] for c in line), min(c["top"] for c in line),
                                      max(c["x1"] for c in line), max(c["bottom"] for c in line)],
                             "text": "".join(c["text"] for c in line), "gaps": gaps})
    for line in lines:
        if not 1 <= sum(is_cjk(c["text"]) for c in line) < 8 or len(line) < 15:
            continue
        line.sort(key=lambda c: c["x0"])
        em = median(c["size"] for c in line)
        gaps = [b["x0"] - a["x1"] for a, b in zip(line, line[1:])]
        if sum(g > max(6 * em, .08 * page_width) for g in gaps) >= 2:
            findings.append({"kind": "sparse-mixed-line",
                             "bbox": [line[0]["x0"], min(c["top"] for c in line),
                                      line[-1]["x1"], max(c["bottom"] for c in line)],
                             "text": "".join(c["text"] for c in line),
                             "max_gap_pt": round(max(gaps), 2)})
    return findings


def audit_layout(document, page_limit: int, exclusions=()) -> dict:
    findings = []
    fonts = Counter()
    for page_no, page in enumerate(document.pages[:page_limit], 1):
        tables = page.find_tables()
        page_exclusions = [e["bbox"] for e in exclusions if e["page"] == page_no]
        page_findings = spacing_findings(page.chars, page.width, page.height,
                                         [t.bbox for t in tables], page_exclusions)
        # Check inside each table cell, rather than suppressing table text.
        for table in tables:
            for cell in table.cells:
                if not cell:
                    continue
                cc = [c for c in page.chars if inside(c, cell)]
                for f in spacing_findings(cc, cell[2] - cell[0], page.height,
                                          exclusions=page_exclusions):
                    f["region"] = "table-cell"
                    page_findings.append(f)
        for f in page_findings:
            f["page"] = page_no
        findings.extend(page_findings)
        for char in page.chars:
            if is_cjk(char["text"]) and .04 * page.height < char["top"] < .94 * page.height:
                fonts[char["fontname"]] += 1
    total = sum(fonts.values())
    thin_fonts = [name for name, count in fonts.items()
                  if count >= max(50, total * .1) and re.search(r"Thin|Hairline|ExtraLight|UltraLight", name, re.I)]
    return {"pages_scanned": page_limit, "findings": findings,
            "font_glyph_counts": dict(fonts), "thin_body_fonts": thin_fonts,
            "exclusions": list(exclusions)}
