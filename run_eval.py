"""STEP 4 - run a system on the dev or test questions, score it, save outputs/*.json, print a table.

  python run_eval.py dev  --system pdf  --method keyword      # Task 1 (also: embedding | all | selected)
  python run_eval.py dev  --system wiki                        # Extension 2: search the wiki instead of the PDF
  python run_eval.py dev  --system laya                        # Extension 1: Laya decides answer/abstain
  python run_eval.py test --system pdf                         # final run, refuses to overwrite (use --force)
  python run_eval.py report                                    # comparison tables -> outputs/results.md
"""
import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import config
from answer import answer_question, build_idf
from pdf_to_chunks import make_chunks, read_rows
from retrieve import make_search, normalize
from wiki_tools import load_wiki_chunks

OUT = Path("outputs")


# ---------- loading ----------
def load_questions(split):
    return [q for q in json.loads(Path("data/questions.json").read_text(encoding="utf-8")) if q["split"] == split]


def load_keys():
    return json.loads(Path("data/answer_keys.json").read_text(encoding="utf-8"))


def load_chunks(system):
    if system == "wiki":
        return load_wiki_chunks("wiki")
    if not (OUT / "chunks.json").exists():                       # build them if step 1 was not run yet
        OUT.mkdir(exist_ok=True)
        pages = read_rows(config.PDF_PATH, config.FIRST_PAGE, config.LAST_PAGE)
        (OUT / "chunks.json").write_text(json.dumps(make_chunks(pages), indent=1, ensure_ascii=False), encoding="utf-8")
    return json.loads((OUT / "chunks.json").read_text(encoding="utf-8"))


# ---------- scoring ----------
def contains_all(text, needles):
    haystack = normalize(text)
    return all(normalize(n) in haystack for n in needles)


def judge(question, retrieved, result, key, laya_error=False):
    """Hit = a top-3 chunk is on an expected page AND contains the key evidence text (a page match alone is not enough).
    Correct answer = answered, contains the key strings, cites an expected page. Correct abstention = abstained."""
    expected = set(question["relevant_pdf_pages"])
    if expected:
        hit = any(set(r["chunk"]["pages"]) & expected and contains_all(r["chunk"]["text"], key["evidence"]) for r in retrieved)
        ok = (result["decision"] == "answer" and contains_all(result["answer"], key["answer"])
              and bool(set(result["pages"]) & expected))
        return {"answerable": True, "retrieval_hit": hit, "correct_answer": ok, "correct_abstention": None,
                "failure": None if ok else ("abstained_on_answerable" if result["decision"] == "abstain" else "wrong_answer")}
    ok = result["decision"] == "abstain" and not laya_error     # an error-triggered abstention never counts as correct
    return {"answerable": False, "retrieval_hit": None, "correct_answer": None, "correct_abstention": ok,
            "failure": None if ok else ("laya_error" if laya_error else "answered_unanswerable")}


def summarize(results):
    ans = [r for r in results if r["judgment"]["answerable"]]
    una = [r for r in results if not r["judgment"]["answerable"]]
    return {"answerable": len(ans), "retrieval_hits": sum(bool(r["judgment"]["retrieval_hit"]) for r in ans),
            "correct_answers": sum(bool(r["judgment"]["correct_answer"]) for r in ans),
            "unanswerable": len(una), "correct_abstentions": sum(bool(r["judgment"]["correct_abstention"]) for r in una),
            "laya_errors": sum(1 for r in results if r.get("laya", {}).get("error"))}


# ---------- running ----------
def run_info():
    versions = {}
    for pkg in ("pymupdf", "numpy", "sentence-transformers", "torch", "laya", "anthropic"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "none"
    return {"python": platform.python_version(), "packages": versions, "git_commit": commit}


def evaluate(split, system, method, force=False):
    path = OUT / f"{split}_{system}_{method}.json"
    if split == "test" and path.exists() and not force:
        sys.exit(f"{path} already exists (final result). Use --force to overwrite.")
    chunks = load_chunks(system)
    idf = build_idf(chunks)
    info = {"split": split, "system": system, "method": method, "settings": {
        k: getattr(config, k) for k in ("TOP_K", "COVERAGE_THRESHOLD", "MAX_SENTENCES", "SENTENCE_CUTOFF")},
        "n_chunks": len(chunks), **run_info()}
    if system == "laya":
        info["laya"] = {"repo": config.LAYA_REPO, "subfolder": config.LAYA_SUBFOLDER}
    try:                                                         # set up the model parts; may legitimately be unavailable
        search = make_search(method, chunks)
        gate = None
        if system == "laya":
            from laya_gate import LayaGate
            gate = LayaGate()
    except Exception as exc:                                     # ImportError, no network, no weights ...
        path.write_text(json.dumps({"status": "not_run", "error": repr(exc), **info}, indent=1), encoding="utf-8")
        print(f"[{split}/{system}/{method}] NOT RUN: {exc}")
        return None
    keys, results = load_keys(), []
    for q in load_questions(split):
        retrieved = search(q["question"], config.TOP_K)
        record = {"id": q["id"], "question": q["question"], "expected_pages": q["relevant_pdf_pages"]}
        laya_error = False
        if gate is None:
            result = answer_question(q["question"], retrieved, idf)
        else:
            laya = gate.decide(q["question"], retrieved)          # Laya replaces the coverage gate; sentence picking unchanged
            record["laya"] = {"input": laya["input"], "output": laya["output"], "label": laya["label"], "error": laya["error"]}
            laya_error = laya["error"] is not None
            result = answer_question(q["question"], retrieved, idf, use_threshold=False) if laya["label"] == "answer" \
                else {"decision": "abstain", "answer": config.ABSTAIN_MESSAGE, "pages": [], "citations": [],
                      "chunk_id": None, "coverage": 0.0}
        record["retrieved"] = [{"rank": r["rank"], "score": round(r["score"], 4), "id": r["chunk"]["id"],
                                "page": r["chunk"]["page"], "file": r["chunk"]["file"], "text": r["chunk"]["text"]}
                               for r in retrieved]
        record.update(result)
        record["judgment"] = judge(q, retrieved, result, keys.get(q["id"], {"evidence": [], "answer": []}), laya_error)
        results.append(record)
    summary = summarize(results)
    path.write_text(json.dumps({"info": info, "summary": summary, "results": results}, indent=1, ensure_ascii=False,
                               default=str), encoding="utf-8")
    print(f"\n=== {split} / {system} / {method}  ->  {path}")
    for r in results:
        j = r["judgment"]
        ok = j["correct_answer"] or j["correct_abstention"]
        print(f"[{'OK  ' if ok else 'FAIL'}] {r['id']} hit={j['retrieval_hit']} {r['decision']:<7} cov={r['coverage']:<6} "
              f"top3={[x['id'] for x in r['retrieved']]}\n        {r['answer'][:150]!r}")
    print("summary:", summary)
    return summary


# ---------- report ----------
def cell(split, system, method, key, denom):
    p = OUT / f"{split}_{system}_{method}.json"
    if not p.exists():
        return "not run"
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("status") == "not_run":
        return "not run (model unavailable)"
    return f"{data['summary'][key]} / {data['summary'][denom]}"


def laya_errors(method):
    p = OUT / f"test_laya_{method}.json"
    if not p.exists():
        return "not run"
    data = json.loads(p.read_text(encoding="utf-8"))
    return "not run (model unavailable)" if data.get("status") == "not_run" else str(data["summary"]["laya_errors"])


def report():
    s = config.SELECTED_METHOD
    lines = ["# Results (generated by `python run_eval.py report` from outputs/*.json)", "",
             "## Task 1 - keyword vs embedding", "| | Keyword | Embedding |", "|---|---|---|"]
    for label, split, key, denom in [("Dev retrieval hits", "dev", "retrieval_hits", "answerable"),
                                     ("Test retrieval hits", "test", "retrieval_hits", "answerable"),
                                     ("Test correct supported answers", "test", "correct_answers", "answerable"),
                                     ("Test correct missing-evidence responses", "test", "correct_abstentions", "unanswerable")]:
        lines.append(f"| {label} | {cell(split, 'pdf', 'keyword', key, denom)} | {cell(split, 'pdf', 'embedding', key, denom)} |")
    lines += ["", f"## Extension 1 - Laya gate (retrieval = {s})", "| Test | Task 1 baseline | With Laya |", "|---|---|---|",
              f"| Correct supported answers | {cell('test', 'pdf', s, 'correct_answers', 'answerable')} | {cell('test', 'laya', s, 'correct_answers', 'answerable')} |",
              f"| Correct abstentions | {cell('test', 'pdf', s, 'correct_abstentions', 'unanswerable')} | {cell('test', 'laya', s, 'correct_abstentions', 'unanswerable')} |",
              f"| Laya runtime errors | n/a | {laya_errors(s)} |",
              "", f"## Extension 2 - wiki (retrieval = {s})", "| Test | PDF RAG | Wiki RAG |", "|---|---|---|",
              f"| Correct supported answers | {cell('test', 'pdf', s, 'correct_answers', 'answerable')} | {cell('test', 'wiki', s, 'correct_answers', 'answerable')} |",
              f"| Correct abstentions | {cell('test', 'pdf', s, 'correct_abstentions', 'unanswerable')} | {cell('test', 'wiki', s, 'correct_abstentions', 'unanswerable')} |",
              f"| (dev) retrieval hits | {cell('dev', 'pdf', s, 'retrieval_hits', 'answerable')} | {cell('dev', 'wiki', s, 'retrieval_hits', 'answerable')} |"]
    text = "\n".join(lines) + "\n"
    (OUT / "results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", choices=["dev", "test", "report"])
    ap.add_argument("--system", choices=["pdf", "wiki", "laya"], default="pdf")
    ap.add_argument("--method", choices=["keyword", "embedding", "all", "selected"], default="selected")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.split == "report":
        report()
        sys.exit(0)
    if a.system != "pdf" and a.method in ("all", "embedding") and a.system == "laya":
        a.method = "selected"
    methods = ["keyword", "embedding"] if a.method == "all" else [config.SELECTED_METHOD if a.method == "selected" else a.method]
    for m in methods:
        evaluate(a.split, a.system, m, a.force)
