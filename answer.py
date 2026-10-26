"""STEP 3 - turn the top-3 chunks into an answer, or abstain. Identical for every retriever and for the wiki.

Idea: weigh each content word of the question by how rare it is in the manual (rare = important; a word that appears
nowhere, like 'price' or 'telephone', is the most important of all). Coverage = share of that weight found in a chunk.
If even the best chunk covers < 50 %, the manual does not say - abstain. Otherwise copy the best sentence(s)."""
import math
import re
from collections import Counter

import config
from retrieve import search_text, tokens

ABBREVIATION_END = re.compile(r"(?:\b(?:ref|Ref|No|e\.g|i\.e|Fig|approx)\.|^\d+(?:\.\d+)*\.)$")


def build_idf(chunks):
    df = Counter()
    for c in chunks:
        df.update(set(tokens(search_text(c))))
    n = len(chunks)
    idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
    idf["__unseen__"] = math.log(n + 1) + 1                      # words absent from the manual weigh the most
    return idf


def coverage(terms, token_set, idf):
    weight = lambda t: idf.get(t, idf["__unseen__"])             # noqa: E731
    total = sum(weight(t) for t in terms)
    return sum(weight(t) for t in terms if t in token_set) / total if total else 0.0


def split_sentences(text):
    """split after . ? ! + capital letter, but not after 'ref.', clause numbers like '5.', 'e.g.' ..."""
    flat = re.sub(r"\s+", " ", text).strip()
    out = []
    for part in re.split(r"(?<=[.?!])\s+(?=[A-Z(\u201c\"\u2018'])", flat):
        if out and ABBREVIATION_END.search(out[-1]):
            out[-1] += " " + part
        else:
            out.append(part)
    return out


def _table_answer(chunk, terms, idf):
    """keep the header ('Standard Limit') and only the row(s) that match the question"""
    lines = chunk["text"].split("\n")
    head = next((i for i, ln in enumerate(lines) if ln.startswith("Standard")), None)
    if head in (None, 0):
        return " ".join(lines)
    rows = lines[head + 1:]
    scores = [coverage(terms, set(tokens(r)), idf) for r in rows]
    best = max(scores, default=0)
    picked = [r for r, s in zip(rows, scores) if best > 0 and s == best] or rows
    return " ".join([lines[head - 1], lines[head], *picked])


def _sentence_answer(chunk, terms, idf):
    sents = split_sentences(chunk["text"])
    scores = [coverage(terms, set(tokens(s)), idf) for s in sents]
    best = max(scores, default=0)
    if best <= 0:
        return " ".join(sents)
    keep = [i for i, s in enumerate(scores) if s > 0 and s >= config.SENTENCE_CUTOFF * best]
    keep = sorted(sorted(keep, key=lambda i: (-scores[i], i))[: config.MAX_SENTENCES])
    return " ".join(sents[i] for i in keep)


def answer_question(question, retrieved, idf, use_threshold=True):
    """retrieved: list from retrieve.py. use_threshold=False lets an outside judge (Laya) make the answer/abstain
    call; the sentence-picking below is unchanged."""
    abstain = {"decision": "abstain", "answer": config.ABSTAIN_MESSAGE, "pages": [], "citations": [],
               "chunk_id": None, "coverage": 0.0}
    terms = sorted(set(tokens(question)))
    if not retrieved or not terms:
        return abstain
    scored = [(coverage(terms, set(tokens(search_text(r["chunk"]))), idf), -r["rank"], r["chunk"]) for r in retrieved]
    best_cov, _, chunk = max(scored, key=lambda t: (t[0], t[1]))
    if use_threshold and best_cov < config.COVERAGE_THRESHOLD:
        return {**abstain, "coverage": round(best_cov, 4)}
    text = _table_answer(chunk, terms, idf) if chunk["kind"] == "table" else _sentence_answer(chunk, terms, idf)
    cite = {"chunk_id": chunk["id"], "section": chunk["section"], "pdf_pages": chunk["pages"], "wiki_file": chunk["file"]}
    return {"decision": "answer", "answer": text, "pages": sorted(chunk["pages"]), "citations": [cite],
            "chunk_id": chunk["id"], "coverage": round(best_cov, 4)}
