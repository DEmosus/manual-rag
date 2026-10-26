"""EXTENSION 2 helpers - turn the wiki/*.md files into chunks (same shape as PDF chunks) and check the wiki.

Wiki format: every fact is one bullet ending  (Source: ..., PDF page N.)  ; a table is a markdown table directly
under a cited line.   Run `python wiki_tools.py` to validate the wiki."""
import json
import re
import sys
from pathlib import Path

import config
from retrieve import normalize

CITE = re.compile(r"\s*\(Source:[^)]*?PDF pages? ([0-9][0-9, and]*)\.?\)\s*$")
TOPICS = ["welder-qualifications.md", "materials-and-standards.md", "surface-quality-and-cutting.md"]


def split_citation(line):
    m = CITE.search(line)
    if not m:
        return line.strip(), []
    return line[: m.start()].strip(), sorted({int(n) for n in re.findall(r"\d+", m.group(1))})


def table_text(rows):
    """markdown table -> the same 'Label: / header / rows' text the PDF table chunks use"""
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(re.fullmatch(r":?-{2,}:?", x) for x in c)]
    return "\n".join([f"{cells[0][0]}:", " ".join(cells[0][1:]), *[" ".join(r) for r in cells[1:]]]) if len(cells) > 1 else None


def load_wiki_chunks(wiki_dir="wiki"):
    chunks = []
    for path in sorted(Path(wiki_dir).glob("*.md")):
        if path.name == "index.md":                              # navigation only, no facts
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), path.stem)
        heading, n, i = title, 0, 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if line.startswith("#"):
                heading = title if line.startswith("# ") else line.lstrip("#").strip()
                continue
            if not line or line.startswith(("Related:", "|", "---")):
                continue
            body, pages = split_citation(line[2:] if line.startswith("- ") else line)
            if not pages:
                continue
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            kind, text = "text", body
            if j < len(lines) and lines[j].strip().startswith("|"):  # cited caption directly above a table
                k = j
                while k < len(lines) and lines[k].strip().startswith("|"):
                    k += 1
                kind, text, i = "table", table_text(lines[j:k]), k
            n += 1
            chunks.append({"id": f"{path.stem}-{n:02d}", "page": min(pages), "pages": pages, "section": heading,
                           "heading": title if heading == title else f"{title} > {heading}", "kind": kind,
                           "text": text, "file": f"wiki/{path.name}"})
    return chunks


def validate_wiki(wiki_dir="wiki", pages_file="outputs/pages.json"):
    """Return a list of problems (empty = OK): files, links, citations, page range, numbers traceable to the PDF page."""
    wiki, problems = Path(wiki_dir), []
    pages = {int(k): normalize(" ".join(v)) for k, v in json.loads(Path(pages_file).read_text(encoding="utf-8")).items()}
    for name in ["index.md", *TOPICS]:
        if not (wiki / name).exists():
            problems.append(f"missing file {name}")
    for p in wiki.glob("*.md"):
        text = p.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)#]+\.md)\)", text):
            if not (wiki / target).exists():
                problems.append(f"{p.name}: broken link {target}")
        if p.name in TOPICS:
            if not any(t in text for t in TOPICS if t != p.name):
                problems.append(f"{p.name}: links to no other topic")
            cited = False
            for ln in text.splitlines():
                s = ln.strip()
                if not s or s.startswith(("#", "Related:", "---")):
                    continue
                if s.startswith("|"):
                    if not cited:
                        problems.append(f"{p.name}: table without a cited line above it")
                    continue
                cited = bool(CITE.search(s))
                if not cited:
                    problems.append(f"{p.name}: no citation: {s[:50]!r}")
    index = (wiki / "index.md").read_text(encoding="utf-8") if (wiki / "index.md").exists() else ""
    problems += [f"index.md does not link {t}" for t in TOPICS if t not in index]
    for c in load_wiki_chunks(wiki_dir):
        if any(p not in pages for p in c["pages"]):
            problems.append(f"{c['id']}: cites a page outside {config.FIRST_PAGE}-{config.LAST_PAGE}")
            continue
        source = " ".join(pages[p] for p in c["pages"])
        for num in re.findall(r"\d+(?:\.\d+)*", c["text"]):
            if not re.search(rf"(?<![\d.]){re.escape(num)}(?!\d)", source):
                problems.append(f"{c['id']}: number {num} not found on cited page(s) {c['pages']}")
    return problems


if __name__ == "__main__":
    found = validate_wiki()
    print(f"wiki chunks: {len(load_wiki_chunks())}; problems: {len(found)}")
    for p in found:
        print(" -", p)
    sys.exit(1 if found else 0)
