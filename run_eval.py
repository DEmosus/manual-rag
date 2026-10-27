"""STEP 4 - run a system on the dev or test questions, score it, save outputs/*.json, print a table.

  python run_eval.py dev  --system pdf  --method all          # Task 1: keyword + embedding (embedding needs extras)
  python run_eval.py dev  --system wiki                        # Extension 2: search the wiki instead of the PDF
  python run_eval.py dev  --system laya                        # Extension 1: Laya decides answer/abstain
  python run_eval.py test --system pdf                         # final run (needs a dev run with identical code/settings)
  python run_eval.py report                                    # comparison tables -> outputs/results.md

Freeze rule: `test` only runs if a dev result exists whose fingerprint (hash of config, retrieval, answer code, questions,
keys and - per system - the wiki files or the Laya prompt) equals the current one, and it never overwrites a finished
test result without --force. So: tune on dev, run dev one last time, then run test.
"""
import argparse
import hashlib
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
BASE_FILES = ["config.py", "retrieve.py", "answer.py", "pdf_to_chunks.py", "data/questions.json", "data/answer_keys.json"]


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


def fingerprint(system):
    """sha256 of everything that must not change between the last dev run and the test run"""
    files = list(BASE_FILES)
    if system == "wiki":
        files += sorted(str(p) for p in Path("wiki").glob("*.md")) + ["prompts/wiki_prompt.md"]
    if system == "laya":
        files += ["laya_gate.py", "prompts/laya_question.json"]
    per_file = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
    return {"combined": hashlib.sha256(json.dumps(per_file, sort_keys=True).encode()).hexdigest(), "files": per_file}


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
            "laya_errors": sum(1 for r in results if (r.get("laya_decision") or {}).get("error"))}


def check_contract(rec):
    """the application-output rules from examples/decision_contract.json"""
    assert rec["decision"] in ("answer", "abstain")
    if rec["decision"] == "answer":
        assert rec["supporting_pdf_pages"], "answer needs supporting_pdf_pages"
    else:
        assert rec["answer"] == config.ABSTAIN_MESSAGE and rec["supporting_pdf_pages"] == []
    assert all(isinstance(c["pdf_page"], int) and 4 <= c["pdf_page"] <= 8 for c in rec["retrieved_chunks"])


def write_readable(path, results, system, method):
    """the task's example format: Question / Method / Retrieved / Answer / Source"""
    lines = []
    for r in results:
        lines += [f"Question: {r['question']}", f"Method: {method} ({system})", "Retrieved:"]
        for c in r["retrieved_chunks"]:
            where = f"{c['wiki_file']}, " if c["wiki_file"] else ""
            lines.append(f"  {c['rank']}. [{where}PDF page {c['pdf_page']}, {c['chunk_id']}, score {c['score']}] "
                         + c["text"].replace("\n", " "))
        lines.append(f"Answer: {r['answer']}")
        cite = r["citations"][0] if r["citations"] else None
        if cite:
            wiki = f"{cite['wiki_file']}; " if cite["wiki_file"] else ""
            lines.append(f"Source: {wiki}PDF page {r['supporting_pdf_pages'][0]}, Section {cite['section']}")
        else:
            lines.append("Source: none (abstained)")
        j = r["judgment"]
        lines += [f"Check: {'CORRECT' if (j['correct_answer'] or j['correct_abstention']) else 'FAILED: ' + str(j['failure'])}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


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
    fp = fingerprint(system)
    if split == "test":                                          # ---- the freeze guard
        if path.exists() and not force:
            sys.exit(f"{path} already exists (final result). Use --force to overwrite.")
        dev = OUT / f"dev_{system}_{method}.json"
        if not dev.exists():
            sys.exit(f"run the dev split first: python run_eval.py dev --system {system} --method {method}")
        if json.loads(dev.read_text(encoding="utf-8"))["info"]["fingerprint"]["combined"] != fp["combined"]:
            sys.exit("code/settings/wiki/prompt changed since the last dev run. Re-run dev, then test (no tuning on test).")
    OUT.mkdir(exist_ok=True)
    chunks = load_chunks(system)
    idf = build_idf(chunks)
    info = {"split": split, "system": system, "method": method, "fingerprint": fp, "settings": {
        k: getattr(config, k) for k in ("TOP_K", "COVERAGE_THRESHOLD", "MAX_SENTENCES", "SENTENCE_CUTOFF")},
        "n_chunks": len(chunks), **run_info()}
    if method == "embedding":
        info["embedding_model"] = config.EMBEDDING_MODEL
    if system == "laya":
        info["laya"] = {"repo": config.LAYA_REPO, "subfolder": config.LAYA_SUBFOLDER}
    try:                                                         # set up the model parts; may legitimately be unavailable
        search = make_search(method, chunks)
        gate = None
        if system == "laya":
            from laya_gate import LayaGate
            gate = LayaGate()
    except Exception as exc:                                     # ImportError, no network, no weights ...
        path.write_text(json.dumps({"status": "not_run", "error": repr(exc), "info": info}, indent=1), encoding="utf-8")
        print(f"[{split}/{system}/{method}] NOT RUN: {exc}")
        return None
    keys, results, laya_log = load_keys(), [], []
    for q in load_questions(split):
        retrieved = search(q["question"], config.TOP_K)
        laya_error, laya_decision = False, None
        if gate is None:
            result = answer_question(q["question"], retrieved, idf)
        else:
            laya = gate.decide(q["question"], retrieved)          # Laya replaces the coverage gate; sentence picking unchanged
            laya_log.append({"question_id": q["id"], **laya})     # raw input/output/error are kept in a separate log file
            laya_decision = {"label": laya["label"], "error": laya["error"]}
            laya_error = laya["error"] is not None
            result = answer_question(q["question"], retrieved, idf, use_threshold=False) if laya["label"] == "answer" \
                else {"decision": "abstain", "answer": config.ABSTAIN_MESSAGE, "pages": [], "citations": [],
                      "chunk_id": None, "coverage": 0.0}
        rec = {"question_id": q["id"], "split": split, "system": system, "method": method, "question": q["question"],
               "expected_pdf_pages": q["relevant_pdf_pages"],
               "retrieved_chunks": [{"rank": r["rank"], "chunk_id": r["chunk"]["id"], "pdf_page": r["chunk"]["page"],
                                     "score": round(r["score"], 4), "wiki_file": r["chunk"]["file"], "text": r["chunk"]["text"]}
                                    for r in retrieved],
               "decision": result["decision"], "answer": result["answer"], "supporting_pdf_pages": result["pages"],
               "citations": result["citations"], "coverage": result["coverage"], "laya_decision": laya_decision,
               "judgment": judge(q, retrieved, result, keys.get(q["id"], {"evidence": [], "answer": []}), laya_error)}
        check_contract(rec)
        results.append(rec)
    summary = summarize(results)
    path.write_text(json.dumps({"info": info, "summary": summary, "results": results}, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    write_readable(path.with_suffix(".md"), results, system, method)
    if gate is not None:
        (OUT / f"{split}_laya_{method}_laya_log.json").write_text(json.dumps(
            {"laya_version": info["packages"]["laya"], "checkpoint": f"{config.LAYA_REPO} / {config.LAYA_SUBFOLDER}",
             "question_prompt_sha256": hashlib.sha256(Path("prompts/laya_question.json").read_bytes()).hexdigest(),
             "entries": laya_log}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n=== {split} / {system} / {method}  ->  {path}  (+ readable .md)")
    for r in results:
        j = r["judgment"]
        ok = j["correct_answer"] or j["correct_abstention"]
        print(f"[{'OK  ' if ok else 'FAIL'}] {r['question_id']} hit={j['retrieval_hit']} {r['decision']:<7} cov={r['coverage']:<6} "
              f"top3={[x['chunk_id'] for x in r['retrieved_chunks']]}\n        {r['answer'][:150]!r}")
    print("summary:", summary)
    return summary


# ---------- report ----------
def load_run(split, system, method):
    p = OUT / f"{split}_{system}_{method}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cell(split, system, method, key, denom=None):
    data = load_run(split, system, method)
    if data is None:
        return "not run"
    if data.get("status") == "not_run":
        return "not run (model unavailable)"
    return f"{data['summary'][key]} / {data['summary'][denom]}" if denom else str(data["summary"][key])


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
              f"| Correct supported answers / 3 | {cell('test', 'pdf', s, 'correct_answers', 'answerable')} | {cell('test', 'laya', s, 'correct_answers', 'answerable')} |",
              f"| Correct abstentions / 2 | {cell('test', 'pdf', s, 'correct_abstentions', 'unanswerable')} | {cell('test', 'laya', s, 'correct_abstentions', 'unanswerable')} |",
              f"| Laya runtime errors | N/A | {cell('test', 'laya', s, 'laya_errors')} |",
              "", f"## Extension 2 - wiki (retrieval = {s})", "| Test | PDF RAG | Wiki RAG |", "|---|---|---|",
              f"| Correct supported answers / 3 | {cell('test', 'pdf', s, 'correct_answers', 'answerable')} | {cell('test', 'wiki', s, 'correct_answers', 'answerable')} |",
              f"| Correct abstentions / 2 | {cell('test', 'pdf', s, 'correct_abstentions', 'unanswerable')} | {cell('test', 'wiki', s, 'correct_abstentions', 'unanswerable')} |",
              f"| (dev) retrieval hits / 5 | {cell('dev', 'pdf', s, 'retrieval_hits', 'answerable')} | {cell('dev', 'wiki', s, 'retrieval_hits', 'answerable')} |"]
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
    if a.system == "laya" and a.method in ("all", "embedding"):
        a.method = "selected"                                    # Laya extends the SELECTED Task 1 retriever only
    methods = ["keyword", "embedding"] if a.method == "all" else [config.SELECTED_METHOD if a.method == "selected" else a.method]
    for m in methods:
        evaluate(a.split, a.system, m, a.force)
