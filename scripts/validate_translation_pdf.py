#!/usr/bin/env python3
"""Deterministic preflight for translated-paper PDFs.

This script does not replace visual review. It renders every page, creates
contact sheets, and fails on text-layer corruption, render-count mismatch,
near-empty pages, oversized CJK line gaps, thin body fonts, or a mismatched
source-page appendix. Dependencies: pypdf, pdfplumber, Pillow, and pdftoppm.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader
import pdfplumber

from cjk_layout import audit_layout


LITERAL_MARKERS = (
    "<sub>",
    "</sub>",
    "<sup>",
    "</sup>",
    "<super>",
    "</super>",
    "&lt;sub&gt;",
    "&lt;/sub&gt;",
    "&lt;sup&gt;",
    "&lt;/sup&gt;",
    "&lt;super&gt;",
    "&lt;/super&gt;",
    "[symbol]sub",
    "[symbol]/sub",
)


def pixels(image: Image.Image) -> list[int]:
    getter = getattr(image, "get_flattened_data", image.getdata)
    return list(getter())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Final translated PDF")
    parser.add_argument(
        "--translated-pages",
        type=int,
        help="Number of generated translation pages to scan; defaults to all pages or the source appendix offset",
    )
    parser.add_argument("--source-pdf", type=Path, help="Original PDF copied as the final appendix")
    parser.add_argument(
        "--pdftoppm",
        type=Path,
        help="Path to pdftoppm; otherwise resolve it from PATH",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        help="Fresh directory for rendered pages, contact sheets, and audit.json",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--layout-exclusions", type=Path,
                        help="JSON list of {page, bbox: [left, top, right, bottom], reason}; only visually confirmed intentional regions")
    return parser.parse_args()


def resolve_pdftoppm(value: Path | None) -> Path:
    if value:
        candidate = value.resolve()
    else:
        found = shutil.which("pdftoppm")
        if not found:
            raise RuntimeError("pdftoppm was not found; pass --pdftoppm or load the PDF workspace dependencies")
        candidate = Path(found).resolve()
    if not candidate.is_file():
        raise RuntimeError(f"pdftoppm does not exist: {candidate}")
    return candidate


def text_layer_audit(reader: PdfReader, page_limit: int) -> dict:
    findings: list[dict] = []
    page_text: list[str] = []
    for page_no, page in enumerate(reader.pages[:page_limit], start=1):
        text = page.extract_text() or ""
        page_text.append(text)
        if "\x00" in text:
            findings.append({"page": page_no, "kind": "nul", "count": text.count("\x00")})
        if "\ufffd" in text:
            findings.append({"page": page_no, "kind": "replacement-character", "count": text.count("\ufffd")})
        lower = text.lower()
        for marker in LITERAL_MARKERS:
            count = lower.count(marker)
            if count:
                findings.append({"page": page_no, "kind": "literal-marker", "marker": marker, "count": count})
    return {"pages_scanned": page_limit, "findings": findings, "page_text": page_text}


def render_pdf(pdf: Path, executable: Path, audit_dir: Path, dpi: int) -> list[Path]:
    audit_dir.mkdir(parents=True, exist_ok=False)
    prefix = audit_dir / "page"
    completed = subprocess.run(
        [str(executable), "-png", "-r", str(dpi), str(pdf), str(prefix)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"pdftoppm failed with {completed.returncode}")
    return sorted(audit_dir.glob("page-*.png"))


def raster_metrics(path: Path) -> dict:
    with Image.open(path) as source:
        image = source.convert("L")
        width, height = image.size
        values = pixels(image)
        ink_fraction = sum(value < 245 for value in values) / len(values)
        band = max(3, round(min(width, height) * 0.008))
        boxes = {
            "top": (0, 0, width, band),
            "bottom": (0, height - band, width, height),
            "left": (0, 0, band, height),
            "right": (width - band, 0, width, height),
        }
        edges = {}
        for name, box in boxes.items():
            crop = image.crop(box)
            edge_values = pixels(crop)
            edges[name] = sum(value < 225 for value in edge_values) / len(edge_values)
    return {
        "width": width,
        "height": height,
        "ink_fraction": round(ink_fraction, 6),
        "edge_ink": {name: round(value, 6) for name, value in edges.items()},
    }


def make_contact_sheets(pngs: list[Path], audit_dir: Path) -> list[Path]:
    cols, rows = 3, 3
    cell_width, cell_height = 360, 510
    outputs = []
    for start in range(0, len(pngs), cols * rows):
        chunk = pngs[start : start + cols * rows]
        sheet = Image.new("RGB", (cols * cell_width, rows * cell_height), (228, 232, 237))
        draw = ImageDraw.Draw(sheet)
        for offset, path in enumerate(chunk):
            with Image.open(path) as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((cell_width - 16, cell_height - 32))
            x = (offset % cols) * cell_width
            y = (offset // cols) * cell_height
            sheet.paste(thumb, (x + (cell_width - thumb.width) // 2, y + 22))
            draw.text((x + 7, y + 5), f"p{start + offset + 1}", fill="black")
        output = audit_dir / f"contact-{start // (cols * rows) + 1:02d}.jpg"
        sheet.save(output, quality=90, optimize=True)
        outputs.append(output)
    return outputs


def source_appendix_audit(final: PdfReader, source_path: Path | None, translated_pages: int) -> dict | None:
    if not source_path:
        return None
    source = PdfReader(str(source_path))
    expected_offset = len(final.pages) - len(source.pages)
    if translated_pages != expected_offset:
        return {
            "source_pages": len(source.pages),
            "appendix_offset": expected_offset,
            "streams_exact": False,
            "error": f"translated page count {translated_pages} does not match appendix offset {expected_offset}",
        }
    exact = all(
        final.pages[translated_pages + index].get_contents().get_data()
        == source.pages[index].get_contents().get_data()
        for index in range(len(source.pages))
    )
    return {"source_pages": len(source.pages), "appendix_offset": translated_pages, "streams_exact": exact}


def main() -> int:
    args = parse_args()
    pdf = args.pdf.resolve()
    if not pdf.is_file():
        raise RuntimeError(f"PDF does not exist: {pdf}")
    source = args.source_pdf.resolve() if args.source_pdf else None
    if source and not source.is_file():
        raise RuntimeError(f"source PDF does not exist: {source}")

    reader = PdfReader(str(pdf))
    default_translated = len(reader.pages) - len(PdfReader(str(source)).pages) if source else len(reader.pages)
    translated_pages = args.translated_pages if args.translated_pages is not None else default_translated
    if translated_pages < 1 or translated_pages > len(reader.pages):
        raise RuntimeError("--translated-pages is outside the final PDF page range")

    audit_dir = (
        args.audit_dir.resolve()
        if args.audit_dir
        else pdf.with_name(f"{pdf.stem}-audit")
    )
    executable = resolve_pdftoppm(args.pdftoppm)
    text_report = text_layer_audit(reader, translated_pages)
    page_text = text_report.pop("page_text")
    exclusions = []
    if args.layout_exclusions:
        exclusions = json.loads(args.layout_exclusions.read_text(encoding="utf-8"))
        if not isinstance(exclusions, list):
            raise ValueError("layout exclusions must be a list")
        for item in exclusions:
            if (not isinstance(item, dict) or not isinstance(item.get("page"), int)
                or not 1 <= item["page"] <= translated_pages
                or not isinstance(item.get("reason"), str) or not item["reason"].strip()
                or not isinstance(item.get("bbox"), list) or len(item["bbox"]) != 4):
                raise ValueError("each exclusion needs a translated-page number, four coordinates, and a reason")
            left, top, right, bottom = item["bbox"]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in item["bbox"]) or not (0 <= left < right and 0 <= top < bottom):
                raise ValueError("invalid exclusion bbox")
    with pdfplumber.open(pdf) as layout_document:
        for item in exclusions:
            page = layout_document.pages[item["page"] - 1]
            if item["bbox"][2] > page.width or item["bbox"][3] > page.height:
                raise ValueError("exclusion bbox is outside the page")
        layout_report = audit_layout(layout_document, translated_pages, exclusions)
    pngs = render_pdf(pdf, executable, audit_dir, args.dpi)
    if len(pngs) != len(reader.pages):
        raise RuntimeError(f"rendered {len(pngs)} pages for a {len(reader.pages)}-page PDF")

    pages = []
    for index, path in enumerate(pngs, start=1):
        metrics = raster_metrics(path)
        text_chars = len((page_text[index - 1] if index <= translated_pages else (reader.pages[index - 1].extract_text() or "")).strip())
        metrics.update(
            {
                "page": index,
                "text_chars": text_chars,
                "blank_suspect": metrics["ink_fraction"] < 0.0015 and text_chars < 20,
                "edge_suspect": max(metrics["edge_ink"].values()) > 0.08,
            }
        )
        pages.append(metrics)

    contacts = make_contact_sheets(pngs, audit_dir)
    appendix = source_appendix_audit(reader, source, translated_pages)
    report = {
        "pdf": str(pdf),
        "pages": len(reader.pages),
        "translated_pages": translated_pages,
        "text_layer": text_report,
        "layout": layout_report,
        "blank_suspects": [page["page"] for page in pages if page["blank_suspect"]],
        "edge_suspects": [page["page"] for page in pages if page["edge_suspect"]],
        "source_appendix": appendix,
        "contact_sheets": [str(path) for path in contacts],
        "manual_review_required": True,
        "page_metrics": pages,
    }
    (audit_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    hard_failures = []
    if text_report["findings"]:
        hard_failures.append("text-layer findings")
    if report["blank_suspects"]:
        hard_failures.append("blank/near-blank page suspects")
    if appendix and not appendix.get("streams_exact"):
        hard_failures.append("source appendix mismatch")
    if layout_report["findings"]:
        hard_failures.append("oversized line gaps require correction or documented visual review")
    if layout_report["thin_body_fonts"]:
        hard_failures.append("thin body font requires a readable regular-weight font")

    print(json.dumps({key: value for key, value in report.items() if key != "page_metrics"}, ensure_ascii=False, indent=2))
    print("Open every page at readable size before delivery; contact sheets and exit 0 do not certify visual quality.", file=sys.stderr)
    if report["edge_suspects"]:
        print("Edge suspects require visual crop review.", file=sys.stderr)
    if hard_failures:
        print("HARD FAIL: " + "; ".join(hard_failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
