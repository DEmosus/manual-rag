"""STEP 1 - PDF pages 4-8  ->  outputs/pages.json (clean rows) + outputs/chunks.json (searchable pieces).

Run:  python pdf_to_chunks.py
"""
import json
import re
import sys
from pathlib import Path

import config

# lines that are page furniture, not content: running header/footer and the side block "No. 47 (cont)"
FURNITURE = re.compile(r"^(No\.|47|\(cont\)|No\.\s*47|Page \d+ of \d+.*)$")
GLYPHS = {"\uf0a3": "\u2264"}                       # font stores "<=" as a private-use character
CLAUSE = re.compile(r"^(\d+(?:\.\d+)+|\d+\.)\s+([A-Z].*)$")   # "1.1 Text", "3.1.1 Text", "5. Gas Cutting"
TABLE_HEAD = re.compile(r"^Standard\s+Limit\b")


def read_rows(pdf_path, first, last):
    """Return {page_number: [text rows]}. Lines on the same baseline are joined, so a table row such as
    'Strength Members 150 um 300 um' (which the PDF stores as 3 separate cells) stays one row."""
    import pymupdf
    if not Path(pdf_path).exists():
        sys.exit(f"PDF not found: {pdf_path}. Put the supplied file in materials/.")
    pages = {}
    with pymupdf.open(pdf_path) as doc:
        for n in range(first, last + 1):                        # page n is index n-1
            items = []
            for block in doc[n - 1].get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    if text and not FURNITURE.match(text):
                        x0, y0, _, y1 = line["bbox"]
                        items.append((y0, y1, x0, text))
            items.sort()
            rows, refs = [], []                                  # refs: (vertical centre, height) of each row's first line
            for y0, y1, x0, text in items:
                centre, height = (y0 + y1) / 2, y1 - y0
                for i, (rc, rh) in enumerate(refs):
                    if abs(centre - rc) < 0.5 * min(height, rh):
                        rows[i].append((x0, text))
                        break
                else:
                    rows.append([(x0, text)])
                    refs.append((centre, height))
            order = sorted(range(len(rows)), key=lambda i: refs[i][0])
            lines = [" ".join(t for _, t in sorted(rows[i])) for i in order]
            for bad, good in GLYPHS.items():
                lines = [ln.replace(bad, good) for ln in lines]
            pages[n] = [re.sub(r"\s+", " ", ln).strip() for ln in lines]
    return pages


def make_chunks(pages):
    """Cut pages into chunks. One chunk = one numbered clause (never spanning pages); a 'Label: / Standard Limit /
    rows' block is its own 'table' chunk so values keep their column headers."""
    chunks, pending_headings, last_section = [], [], ""
    for page, rows in pages.items():
        clauses = []                                             # [(section or None, [lines])]
        for row in rows:
            m = CLAUSE.match(row)
            if m:
                clauses.append((m.group(1).rstrip("."), [row]))
            elif clauses:
                clauses[-1][1].append(row)
            else:
                clauses.append((None, [row]))                    # continuation of the previous page's clause
        counts = {}

        def add(section, lines, kind, heading):
            counts[section] = counts.get(section, 0) + 1
            chunks.append({"id": f"p{page}-{section}-{counts[section]:02d}", "page": page, "pages": [page],
                           "section": section, "heading": " > ".join(heading), "kind": kind,
                           "text": "\n".join(lines), "file": None})

        for section, lines in clauses:
            if section is None:
                section = last_section or "front"
            else:
                last_section = section
                if len(lines) == 1 and len(lines[0]) < 90 and not lines[0].endswith("."):
                    pending_headings.append(lines[0])            # heading-only clause: becomes context for the next chunk
                    continue
            heading, pending_headings = pending_headings, []
            starts = [i - 1 for i, ln in enumerate(lines) if i > 0 and TABLE_HEAD.match(ln) and lines[i - 1].endswith(":")]
            if not starts:
                add(section, lines, "text", heading)
                continue
            intro = lines[: starts[0]]
            if len(intro) > 2:
                add(section, intro, "text", heading)
                intro = intro[:1]
            for s, e in zip(starts, starts[1:] + [len(lines)]):
                add(section, intro + lines[s:e], "table", heading)
    return chunks


if __name__ == "__main__":
    pages = read_rows(config.PDF_PATH, config.FIRST_PAGE, config.LAST_PAGE)
    chunks = make_chunks(pages)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/pages.json").write_text(json.dumps(pages, indent=1, ensure_ascii=False), encoding="utf-8")
    Path("outputs/chunks.json").write_text(json.dumps(chunks, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"pages {config.FIRST_PAGE}-{config.LAST_PAGE}: {len(chunks)} chunks -> outputs/chunks.json")
