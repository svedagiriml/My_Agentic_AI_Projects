"""PDF -> Markdown conversion.

Primary converter: **pymupdf4llm** – fast, CPU only, detects tables and emits them
as GitHub-flavoured pipe tables, and returns per-page output so we can keep page
numbers for citations. (PyMuPDF is AGPL – fine for an open-source portfolio repo;
switch to `PDF_CONVERTER=docling` (MIT) if that matters for you.)

Post-processing:
  * page markers  `<!-- page: N -->` are inserted so the chunker can attach page numbers
  * running headers / footers (lines repeated on >50 % of pages) are removed
  * hyphenated line breaks are re-joined, `<br>` inside table cells become spaces
"""
from __future__ import annotations

import collections
import os
import re
import tempfile

PAGE_MARKER = "<!-- page: {} -->"
PAGE_MARKER_RE = re.compile(r"^<!-- page: (\d+) -->$")


def _convert_pymupdf(path: str) -> list[str]:
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(path, page_chunks=True, show_progress=False, table_strategy="lines_strict")
    return [p["text"] for p in pages]


def _convert_docling(path: str) -> list[str]:
    from docling.document_converter import DocumentConverter

    doc = DocumentConverter().convert(path).document
    return [doc.export_to_markdown()]  # docling exports a single page stream


def _strip_running_lines(pages: list[str]) -> list[str]:
    if len(pages) < 3:
        return pages
    counts: collections.Counter[str] = collections.Counter()
    for p in pages:
        for line in {ln.strip() for ln in p.splitlines() if ln.strip()}:
            if not line.startswith("|") and len(line) < 120:
                counts[line] += 1
    running = {ln for ln, n in counts.items() if n / len(pages) > 0.5}
    running |= {ln for ln in counts if re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", ln, re.I)}
    return ["\n".join(ln for ln in p.splitlines() if ln.strip() not in running) for p in pages]


def clean_markdown(md: str) -> str:
    md = md.replace(" ", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    md = re.sub(r"(\w)-\n(\w)", r"\1\2", md)               # de-hyphenate
    md = re.sub(r"<br\s*/?>", " ", md)                     # cell line breaks
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def pdf_bytes_to_markdown(data: bytes, filename: str, converter: str | None = None) -> str:
    converter = converter or os.getenv("PDF_CONVERTER", "pymupdf4llm")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        pages = _convert_docling(path) if converter == "docling" else _convert_pymupdf(path)
    finally:
        os.unlink(path)
    pages = _strip_running_lines(pages)
    out = []
    for i, text in enumerate(pages, start=1):
        out.append(PAGE_MARKER.format(i))
        out.append(text)
    return clean_markdown("\n\n".join(out))
