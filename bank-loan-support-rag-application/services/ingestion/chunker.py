"""Structure-aware Markdown chunker.

Pipeline:  markdown --parse--> typed blocks --assemble--> semantic chunks

1. **Parse** the markdown into typed blocks while tracking the heading hierarchy
   and the current page:
      heading | paragraph | list | table | page-marker
   Lines that are entirely bold and short (`**Loan Tenor:**`) are treated as
   pseudo-headings, because PDF->MD conversion of bank MITCs renders most section
   titles that way.

2. **Assemble** chunks, one semantic unit per chunk:
   * paragraph -> its own chunk. Tiny fragments (< CHUNK_MIN_TOKENS) are merged
     with the *next* block in the same section so no chunk is meaningless on its
     own; oversized paragraphs are split on sentence boundaries.
   * list      -> its own chunk, together with its lead-in sentence
     ("The loan will be disbursed only on the following conditions:").
   * table     -> its own chunk (never mixed with prose). See `_table_chunks`.
   Every chunk is prefixed with a *contextual header* – document title + heading
   path – so that a chunk such as "0.35% of the loan amount" still carries the
   meaning "SBI Home Loan > Processing Fee" when embedded or BM25-indexed.

Table handling
--------------
* Tables that continue across a page break (same column count, repeated or no
  header) are stitched back together first.
* Empty columns/rows are dropped; the header row is detected from the `|---|`
  separator.
* The short paragraph right before a table (or ending with ':') is used as its
  caption.
* `display_text` keeps the **markdown table** (what the LLM reads – it reasons
  well over pipe tables). `text` (what is embedded and BM25-indexed) is a
  **row-linearised** version: `Loan Amount: Up to Rs.30 lakh | LTV Ratio: 90%`.
  Linearisation pairs every cell with its column header, which embeds far better
  than a bare grid of numbers and lets BM25 match "LTV for 50 lakh loan".
* Tables longer than TABLE_MAX_ROWS_PER_CHUNK are split into row groups; each
  part repeats the header row and caption so it remains self-describing.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from common.schemas import Chunk, ChunkType, LoanProduct
from services.ingestion.pdf_to_markdown import PAGE_MARKER_RE

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
BOLD_HEADING_RE = re.compile(r"^\*\*([^*]{2,90}?)\*\*:?\s*$")
LIST_RE = re.compile(r"^\s*(?:[-*+•]|\d{1,2}[.)]|\([a-z0-9ivx]{1,4}\)|[a-z][.)])\s+")
TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\d₹])")


def estimate_tokens(text: str) -> int:
    return max(1, int(len(re.findall(r"\S+", text)) * 1.3))


# --------------------------------------------------------------------------- blocks
@dataclass
class Block:
    kind: str                     # paragraph | list | table
    lines: list[str]
    section: list[str]
    page: int | None

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


@dataclass
class TableData:
    header: list[str]
    rows: list[list[str]]
    caption: str | None = None
    page: int | None = None
    section: list[str] = field(default_factory=list)


def parse_blocks(md: str) -> list[Block]:
    blocks: list[Block] = []
    section: list[tuple[int, str]] = []
    page: int | None = None
    buf: list[str] = []
    buf_kind: str | None = None
    buf_page: int | None = None

    def flush() -> None:
        nonlocal buf, buf_kind
        if buf and buf_kind and any(x.strip() for x in buf):
            blocks.append(Block(buf_kind, buf, [s for _, s in section], buf_page))
        buf, buf_kind = [], None

    def start(kind: str) -> None:
        nonlocal buf_kind, buf_page
        flush()
        buf_kind, buf_page = kind, page

    def set_heading(level: int, title: str) -> None:
        nonlocal section
        title = re.sub(r"[*_`]", "", title).strip().rstrip(":")
        section = [(lvl, t) for lvl, t in section if lvl < level] + [(level, title)]

    for raw in md.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if m := PAGE_MARKER_RE.match(stripped):
            # A table may continue on the next page: keep the table buffer open
            # (and _stitch_tables re-joins it if a blank line slipped in).
            if buf_kind != "table":
                flush()
            page = int(m.group(1))
            continue
        if not stripped:
            flush()
            continue
        if m := HEADING_RE.match(stripped):
            flush()
            set_heading(len(m.group(1)), m.group(2))
            continue
        if (m := BOLD_HEADING_RE.match(stripped)) and len(m.group(1).split()) <= 10:
            flush()
            set_heading(7, m.group(1))
            continue
        if stripped.startswith("|"):
            if buf_kind != "table":
                start("table")
            buf.append(stripped)
            continue
        if LIST_RE.match(line):
            if buf_kind != "list":
                start("list")
            buf.append(stripped)
            continue
        if buf_kind == "list" and raw.startswith(("  ", "\t")):
            buf[-1] += " " + stripped          # continuation of a list item
            continue
        if buf_kind != "paragraph":
            start("paragraph")
        buf.append(stripped)
    flush()
    return blocks


# --------------------------------------------------------------------------- tables
def _cells(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [re.sub(r"\s+", " ", c.replace("**", "")).strip() for c in row.split("|")]


def parse_table(block: Block) -> TableData:
    header: list[str] | None = None
    rows: list[list[str]] = []
    pending: list[list[str]] = []
    for line in block.lines:
        if TABLE_SEP_RE.match(line):
            if pending and header is None:
                header = pending.pop()
            rows.extend(pending)
            pending = []
            continue
        cells = _cells(line)
        if header is not None and cells == header:   # repeated header after page break
            continue
        pending.append(cells)
    rows.extend(pending)
    if header is None:
        header = rows.pop(0) if rows else []
    # Converter artefact repair: pymupdf4llm emits "Col2", "Col3"... for header cells it
    # could not read, and sometimes glues a stray text line above the table into the
    # header. If the header is mostly placeholders and the first row is fully populated,
    # that first row is the real header.
    placeholders = sum(bool(re.fullmatch(r"Col\d+", h)) or not h for h in header)
    if rows and placeholders and placeholders >= len(header) / 2 and all(rows[0]):
        header = rows.pop(0)
    width = max([len(header)] + [len(r) for r in rows]) if (header or rows) else 0
    header = header + [""] * (width - len(header))
    rows = [r + [""] * (width - len(r)) for r in rows]
    # drop fully empty columns and rows
    keep = [i for i in range(width) if header[i] or any(r[i] for r in rows)]
    header = [header[i] or f"Column {n + 1}" for n, i in enumerate(keep)]
    rows = [[r[i] for i in keep] for r in rows if any(r[i] for i in keep)]
    return TableData(header=header, rows=rows, page=block.page, section=block.section)


def table_to_markdown(t: TableData, rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(t.header) + " |", "|" + "|".join("---" for _ in t.header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def linearize_rows(t: TableData, rows: list[list[str]]) -> str:
    lines = []
    for r in rows:
        pairs = [f"{h}: {v}" for h, v in zip(t.header, r) if v]
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)


# --------------------------------------------------------------------------- chunker
@dataclass
class _Draft:
    kind: ChunkType
    section: list[str]
    page: int | None
    text: str
    display: str
    table_meta: dict[str, object] | None = None


class StructureAwareChunker:
    def __init__(self, min_tokens: int = 40, max_tokens: int = 350, table_max_rows: int = 15) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.table_max_rows = table_max_rows

    # ---- public -----------------------------------------------------------
    def chunk(self, md: str, *, doc_id: str, doc_title: str, source_file: str,
              product: LoanProduct) -> list[Chunk]:
        blocks = self._stitch_tables(parse_blocks(md))
        drafts = self._assemble(blocks)
        return self._finalize(drafts, doc_id, doc_title, source_file, product)

    # ---- stages -----------------------------------------------------------
    @staticmethod
    def _stitch_tables(blocks: list[Block]) -> list[Block]:
        """Merge a table that was split by a page break into the previous table
        (consecutive table blocks on different pages with the same column count)."""
        out: list[Block] = []
        for b in blocks:
            prev = out[-1] if out else None
            if (b.kind == "table" and prev is not None and prev.kind == "table" and b.page != prev.page
                    and len(_cells(b.lines[0])) == len(_cells(prev.lines[0]))):
                out[-1].lines.extend(b.lines)
                continue
            out.append(b)
        return out

    def _assemble(self, blocks: list[Block]) -> list[_Draft]:
        drafts: list[_Draft] = []
        i = 0
        carry: Block | None = None   # tiny paragraph waiting to be merged forward
        while i < len(blocks):
            b = blocks[i]
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None

            if b.kind == "paragraph":
                # Caption / lead-in detection: short paragraph ending with ':' or
                # short paragraph right before a table, in the same section.
                is_leadin = nxt is not None and nxt.section == b.section and nxt.kind in ("table", "list") and (
                    b.text.endswith(":") or estimate_tokens(b.text) <= 30)
                if is_leadin:
                    nxt.lines = ([f"__CAPTION__{b.text}"] if nxt.kind == "table" else [b.text]) + nxt.lines
                    i += 1
                    continue
                if carry is not None and carry.section == b.section:
                    b = Block("paragraph", carry.lines + b.lines, b.section, carry.page)
                    carry = None
                elif carry is not None:
                    drafts.extend(self._paragraph_drafts(carry))
                    carry = None
                tiny = estimate_tokens(b.text) < self.min_tokens
                if tiny and nxt is not None and nxt.section == b.section and nxt.kind == "paragraph":
                    carry = b                                    # merge forward
                elif tiny and drafts and drafts[-1].kind is ChunkType.paragraph \
                        and drafts[-1].section == b.section \
                        and estimate_tokens(drafts[-1].text + " " + b.text) <= self.max_tokens:
                    tail = re.sub(r"\s+", " ", b.text)
                    merged = f"{drafts[-1].text} {tail}"                     # merge backward
                    drafts[-1].text = drafts[-1].display = merged
                else:
                    drafts.extend(self._paragraph_drafts(b))
            else:
                if carry is not None:
                    drafts.extend(self._paragraph_drafts(carry))
                    carry = None
                drafts.extend(self._table_drafts(b) if b.kind == "table" else self._list_drafts(b))
            i += 1
        if carry is not None:
            drafts.extend(self._paragraph_drafts(carry))
        return drafts

    def _split_sentences(self, text: str) -> list[str]:
        sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
        groups, cur = [], ""
        for s in sents:
            if cur and estimate_tokens(cur + " " + s) > self.max_tokens:
                groups.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            groups.append(cur)
        return groups

    def _paragraph_drafts(self, b: Block) -> list[_Draft]:
        text = re.sub(r"\s+", " ", b.text)
        parts = [text] if estimate_tokens(text) <= self.max_tokens else self._split_sentences(text)
        return [_Draft(ChunkType.paragraph, b.section, b.page, p, p) for p in parts]

    def _list_drafts(self, b: Block) -> list[_Draft]:
        lead = [] if LIST_RE.match(b.lines[0]) else [b.lines[0]]
        items = b.lines[len(lead):]
        out, cur = [], list(lead)
        for it in items:
            if len(cur) > len(lead) and estimate_tokens("\n".join(cur + [it])) > self.max_tokens:
                out.append(cur)
                cur = list(lead)        # repeat the lead-in in every part
            cur.append(it)
        out.append(cur)
        return [_Draft(ChunkType.list, b.section, b.page, "\n".join(p), "\n".join(p)) for p in out]

    def _table_drafts(self, b: Block) -> list[_Draft]:
        caption = None
        if b.lines and b.lines[0].startswith("__CAPTION__"):
            caption = b.lines[0].removeprefix("__CAPTION__")
            b = Block("table", b.lines[1:], b.section, b.page)
        t = parse_table(b)
        t.caption = caption
        if not t.rows:
            return []
        table_key = hashlib.sha1(("|".join(t.header) + str(t.rows[:2])).encode()).hexdigest()[:8]
        groups = [t.rows[i:i + self.table_max_rows] for i in range(0, len(t.rows), self.table_max_rows)]
        drafts = []
        for n, rows in enumerate(groups, start=1):
            cap = caption or (b.section[-1] if b.section else "Table")
            part = f" (part {n}/{len(groups)})" if len(groups) > 1 else ""
            display = f"Table: {cap}{part}\n\n{table_to_markdown(t, rows)}"
            text = f"Table: {cap}{part}\nColumns: {', '.join(t.header)}\n{linearize_rows(t, rows)}"
            drafts.append(_Draft(ChunkType.table, b.section, b.page, text, display, {
                "table_id": table_key, "part": n, "total_parts": len(groups),
                "n_rows": len(rows), "columns": t.header, "caption": cap,
            }))
        return drafts

    def _finalize(self, drafts: list[_Draft], doc_id: str, doc_title: str, source_file: str,
                  product: LoanProduct) -> list[Chunk]:
        chunks: list[Chunk] = []
        seen: dict[str, int] = {}
        for pos, d in enumerate(drafts):
            path = " > ".join(d.section)
            header = f"[{doc_title}]" + (f" {path}" if path else "")
            text = f"{header}\n{d.text}"
            display = f"{header}\n{d.display}"
            digest = hashlib.sha256(display.encode("utf-8")).hexdigest()
            dup = seen.get(digest, 0)
            seen[digest] = dup + 1
            # Content-addressed ID: unchanged content => unchanged ID (this is what
            # makes the CDC diff cheap). `dup` disambiguates identical passages.
            cid = hashlib.sha256(f"{doc_id}|{digest}|{dup}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::{cid}", doc_id=doc_id, source_file=source_file, product=product,
                chunk_type=d.kind, section_path=d.section, page=d.page, position=pos, text=text,
                display_text=display, content_hash=digest, token_count=estimate_tokens(text),
                table_meta=d.table_meta,
            ))
        return chunks
