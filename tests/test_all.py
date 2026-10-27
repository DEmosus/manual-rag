import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from answer import answer_question, build_idf, split_sentences  # noqa: E402
from laya_gate import LayaGate, build_state  # noqa: E402
from pdf_to_chunks import make_chunks, read_rows  # noqa: E402
from retrieve import keyword_search, make_embedding_search, stem, tokens  # noqa: E402
from run_eval import judge, summarize  # noqa: E402
from wiki_tools import load_wiki_chunks, split_citation, validate_wiki  # noqa: E402

MSG = config.ABSTAIN_MESSAGE


def chunk(i, text, page=5, kind="text", heading=""):
    return {"id": f"c{i}", "page": page, "pages": [page], "section": "s", "heading": heading, "kind": kind,
            "text": text, "file": None}


# ---- text helpers
def test_tokens_stem_and_stopwords():
    assert tokens("approves approved approve") == ["approv"] * 3
    assert tokens("qualification qualified") == ["qualif"] * 2
    assert tokens("what is the difference between") == []
    assert stem("types") == "type"


def test_sentence_split_keeps_ref_and_clause_numbers():
    assert split_sentences("5. Gas Cutting Rough (ref. A6). Next one.") == ["5. Gas Cutting Rough (ref. A6).", "Next one."]


# ---- chunker
PAGES = {4: ["1. Scope", "Intro.", "1.1 First line", "more text."],
         5: ["continuation text.", "3. Qual", "3.1 Welders", "3.1.1 Welders shall qualify.", "5. Gas Cutting",
             "Roughness must meet:", "Free Edges:", "Standard Limit", "Strength Members 150 \u00b5m 300 \u00b5m",
             "Others 500 \u00b5m 1000 \u00b5m", "Welding Edges:", "Standard Limit", "Strength Members 400 \u00b5m 800 \u00b5m"]}


def test_chunker_pages_headings_tables():
    cs = {c["id"]: c for c in make_chunks(PAGES)}
    assert {c["page"] for c in cs.values()} == {4, 5}
    assert cs["p4-1.1-01"]["text"].endswith("more text.")
    assert cs["p5-3.1.1-01"]["heading"] == "3. Qual > 3.1 Welders" and "3.1 Welders" not in cs["p5-3.1.1-01"]["text"]
    tables = [c for c in cs.values() if c["kind"] == "table"]
    assert len(tables) == 2 and tables[0]["text"].startswith("5. Gas Cutting")
    assert "Free Edges:\nStandard Limit\nStrength Members 150" in tables[0]["text"]


def test_missing_pdf_exits(tmp_path):
    with pytest.raises(SystemExit):
        read_rows(tmp_path / "nope.pdf", 4, 8)


# ---- retrieval
CH = [chunk(0, "welders qualified standard"), chunk(1, "materials approved society"),
      chunk(2, "welders qualified standard society"), chunk(3, "unrelated words")]


def test_keyword_ranking_ties_and_empty():
    out = keyword_search(CH, "welders standard society", 3)
    assert [r["chunk"]["id"] for r in out] == ["c2", "c0", "c1"] and out[0]["score"] == 3
    assert keyword_search(CH, "what is the", 3) == []
    assert [r["rank"] for r in out] == [1, 2, 3]


def test_embedding_cosine_with_fake_encoder():
    vocab = ["weld", "steel", "gas"]
    enc = lambda texts: np.array([[t.lower().count(w) for w in vocab] for t in texts], float)  # noqa: E731
    cs = [chunk(0, "weld weld"), chunk(1, "steel gas"), chunk(2, "gas gas gas")]
    out = make_embedding_search(cs, enc)("gas", 3)
    assert [r["chunk"]["id"] for r in out] == ["c2", "c1", "c0"] and out[0]["score"] == pytest.approx(1.0)
    assert make_embedding_search(cs, enc)("  ", 3) == []
    with pytest.raises(ValueError):
        make_embedding_search(cs, lambda t: np.full((len(t), 2), np.nan))


# ---- answer / abstain
AC = [chunk(0, "3.1.1 Welders are to be qualified in accordance with UR W32 or other recognized standard. Other text."),
      chunk(1, "4.1 All materials used should be manufactured at a works approved by the Society.", 5),
      chunk(2, "5. Gas Cutting\nRoughness:\nFree Edges:\nStandard Limit\nStrength Members 150 \u00b5m 300 \u00b5m\n"
               "Others 500 \u00b5m 1000 \u00b5m", 8, "table")]


def run(q, **kw):
    return answer_question(q, keyword_search(AC, q, 3), build_idf(AC), **kw)


def test_answer_copies_sentence_with_page():
    a = run("Which qualification standard is named for welders?")
    assert a["decision"] == "answer" and a["pages"] == [5] and "UR W32" in a["answer"]


def test_abstains_when_topic_matches_but_fact_missing():
    a = run("What is the purchase price of a steel plate from the supplier?")
    assert a["decision"] == "abstain" and a["answer"] == MSG and a["pages"] == [] and a["citations"] == []


def test_table_answer_keeps_header_and_matching_row_only():
    a = run("What are the free edges roughness values for strength members?")
    assert "Standard Limit Strength Members 150 \u00b5m 300 \u00b5m" in a["answer"] and "Others" not in a["answer"]


def test_threshold_can_be_handed_to_outside_judge():
    q = "What is the purchase price of a steel plate from the supplier?"
    assert run(q)["decision"] == "abstain" and run(q, use_threshold=False)["decision"] == "answer"


# ---- scoring
KEY = {"evidence": ["UR W32"], "answer": ["UR W32"]}
Q = {"id": "q", "question": "?", "relevant_pdf_pages": [5], "split": "dev"}
R = lambda text, page: [{"chunk": chunk(0, text, page), "rank": 1, "score": 1}]  # noqa: E731


def test_judge_hit_needs_page_and_text():
    good = {"decision": "answer", "answer": "UR W32", "pages": [5]}
    assert judge(Q, R("x UR W32", 5), good, KEY)["correct_answer"]
    assert not judge(Q, R("x UR W32", 6), good, KEY)["retrieval_hit"]
    assert not judge(Q, R("other", 5), good, KEY)["retrieval_hit"]
    j = judge(Q, R("x UR W32", 5), {"decision": "abstain", "answer": MSG, "pages": []}, KEY)
    assert j["failure"] == "abstained_on_answerable" and not j["correct_answer"]


def test_judge_unanswerable_and_laya_error_not_counted():
    uq = {"id": "u", "question": "?", "relevant_pdf_pages": [], "split": "test"}
    abst = {"decision": "abstain", "answer": MSG, "pages": []}
    assert judge(uq, [], abst, {})["correct_abstention"]
    err = judge(uq, [], abst, {}, laya_error=True)
    assert not err["correct_abstention"] and err["failure"] == "laya_error"
    assert summarize([{"judgment": err, "laya_decision": {"label": "abstain", "error": "boom"}}])["laya_errors"] == 1


# ---- Extension 1 (logic only; a fake agent stands in, this is NOT a Laya run)
class FakeAgent:
    def __init__(self, label=None, boom=False):
        self.label, self.boom = label, boom

    def predict(self, state, questions):
        if self.boom:
            raise RuntimeError("model crashed")
        assert questions["enough_evidence"]["type"] == "choice" and set(questions["enough_evidence"]["criteria"]) == {"answer", "abstain"}
        return {"answers": {"enough_evidence": {"choice": self.label}}}


def test_laya_gate_labels_errors_and_input():
    ret = keyword_search(AC, "welders", 3)
    assert "Question: welders" in build_state("welders", ret) and "PDF page" in build_state("welders", ret)
    assert LayaGate(FakeAgent("answer")).decide("welders", ret)["label"] == "answer"
    bad = LayaGate(FakeAgent("maybe")).decide("welders", ret)
    assert bad["label"] == "abstain" and "unexpected label" in bad["error"]
    assert LayaGate(FakeAgent(boom=True)).decide("welders", ret)["error"] is not None


# ---- Extension 2 helpers
CITE = "(Source: IACS Rec. 47, PDF page {p}.)"


def build_wiki(tmp, body):
    names = ["welder-qualifications.md", "materials-and-standards.md", "surface-quality-and-cutting.md"]
    (tmp / "index.md").write_text("\n".join(f"- [t]({n})" for n in names))
    for n in names:
        other = next(o for o in names if o != n)
        (tmp / n).write_text(f"# T\n\nRelated: [o]({other})\n\n## S\n{body}\n")
    pages = tmp / "pages.json"
    pages.write_text(json.dumps({"4": ["welders UR W32 limit 150"], "5": ["text 300 mm"]}))
    return pages


def test_wiki_loader_and_validator(tmp_path):
    ok = "- Welders use UR W32. " + CITE.format(p=4)
    pages = build_wiki(tmp_path, ok)
    assert validate_wiki(tmp_path, pages) == []
    assert split_citation("x (Source: A, PDF pages 4 and 5.)") == ("x", [4, 5])
    (tmp_path / "welder-qualifications.md").write_text(
        "# T\n\n## S\n- Claim. " + CITE.format(p=4) + "\n\nCaption. " + CITE.format(p=5) +
        "\n\n| Free Edges | Standard | Limit |\n|---|---|---|\n| Strength | 150 | 300 |\n")
    cs = [c for c in load_wiki_chunks(tmp_path) if c["file"].endswith("welder-qualifications.md")]
    assert [c["kind"] for c in cs] == ["text", "table"]
    assert cs[1]["text"] == "Free Edges:\nStandard Limit\nStrength 150 300" and cs[1]["pages"] == [5]


def test_wiki_validator_catches_problems(tmp_path):
    bad = ("- No citation.\n- Out of range. " + CITE.format(p=9) + "\n- Invented 999 mm. " + CITE.format(p=4)
           + "\n- Link [x](missing.md). " + CITE.format(p=4))
    problems = " ".join(validate_wiki(tmp_path, build_wiki(tmp_path, bad)))
    assert "no citation" in problems and "outside" in problems and "number 999" in problems and "broken link" in problems


# ---- needs the real PDF (skipped if it is not in materials/)
@pytest.mark.skipif(not Path(config.PDF_PATH).exists(), reason="PDF not present")
def test_real_pdf_pages_and_gas_cutting_table():
    pages = read_rows(config.PDF_PATH, 4, 8)
    assert list(pages) == [4, 5, 6, 7, 8]
    cs = make_chunks(pages)
    assert {c["page"] for c in cs} == {4, 5, 6, 7, 8}
    free = next(c for c in cs if c["kind"] == "table" and "Free Edges" in c["text"])
    assert "Standard Limit" in free["text"] and "Strength Members 150 \u00b5m 300 \u00b5m" in free["text"]
    top = keyword_search(cs, "Which qualification standard is named for welders?", 3)[0]["chunk"]
    assert top["page"] == 5 and "UR W32" in top["text"]


# ---- Extension 1 wiring end to end (fake agent: checks OUR code path only, it is NOT a Laya run)
needs_pdf = pytest.mark.skipif(not Path(config.PDF_PATH).exists(), reason="PDF not present")


@needs_pdf
@pytest.mark.parametrize("agent_kwargs, expect_errors", [({"label": "abstain"}, 0), ({"boom": True}, 5)])
def test_laya_pipeline_path(monkeypatch, tmp_path, agent_kwargs, expect_errors):
    import laya_gate
    import run_eval
    monkeypatch.setattr(run_eval, "OUT", tmp_path)
    real = laya_gate.LayaGate
    monkeypatch.setattr(laya_gate, "LayaGate", lambda: real(FakeAgent(**agent_kwargs)))
    summary = run_eval.evaluate("dev", "laya", "keyword")
    log = json.loads((tmp_path / "dev_laya_keyword_laya_log.json").read_text())
    assert summary["laya_errors"] == expect_errors and summary["correct_answers"] == 0
    assert log["entries"][0]["input"].startswith("Question:") and len(log["entries"]) == 5     # Laya input saved separately
    assert (log["entries"][0]["error"] is not None) == bool(expect_errors)
    assert len(log["question_prompt_sha256"]) == 64


@needs_pdf
def test_output_follows_decision_contract(monkeypatch, tmp_path):
    import run_eval
    monkeypatch.setattr(run_eval, "OUT", tmp_path)
    run_eval.evaluate("dev", "pdf", "keyword")
    rec = json.loads((tmp_path / "dev_pdf_keyword.json").read_text())["results"][2]
    for field in ("question_id", "retrieved_chunks", "decision", "answer", "supporting_pdf_pages"):
        assert field in rec
    assert len(rec["retrieved_chunks"]) == 3 and {"chunk_id", "pdf_page", "text"} <= set(rec["retrieved_chunks"][0])
    assert (tmp_path / "dev_pdf_keyword.md").read_text().startswith("Question:")   # readable example format
    with pytest.raises(AssertionError):                                           # contract: answer needs pages
        run_eval.check_contract({**rec, "decision": "answer", "supporting_pdf_pages": []})


@needs_pdf
def test_freeze_guard(monkeypatch, tmp_path):
    import run_eval
    monkeypatch.setattr(run_eval, "OUT", tmp_path)
    with pytest.raises(SystemExit):                                               # no dev run yet
        run_eval.evaluate("test", "pdf", "keyword")
    run_eval.evaluate("dev", "pdf", "keyword")
    run_eval.evaluate("test", "pdf", "keyword")                                   # allowed: dev fingerprint matches
    with pytest.raises(SystemExit):                                               # finished test result is not overwritten
        run_eval.evaluate("test", "pdf", "keyword")
    (tmp_path / "test_pdf_keyword.json").unlink()
    monkeypatch.setattr(config, "COVERAGE_THRESHOLD", 0.9)                        # settings changed after dev...
    monkeypatch.setattr(run_eval, "fingerprint", lambda s: {"combined": "different", "files": {}})
    with pytest.raises(SystemExit):                                               # ...so test is refused
        run_eval.evaluate("test", "pdf", "keyword")


# ---- final-version behaviour
@needs_pdf
def test_stale_not_run_marker_does_not_block_test_run(monkeypatch, tmp_path):
    import run_eval
    monkeypatch.setattr(run_eval, "OUT", tmp_path)
    run_eval.evaluate("dev", "pdf", "keyword")
    fp = run_eval.fingerprint("pdf")
    (tmp_path / "test_pdf_keyword.json").write_text(json.dumps({"status": "not_run", "error": "x", "info": {"fingerprint": fp}}))
    assert run_eval.evaluate("test", "pdf", "keyword")["answerable"] == 3          # marker overwritten, no --force needed
    with pytest.raises(SystemExit):                                                 # a real result is still protected
        run_eval.evaluate("test", "pdf", "keyword")


def test_report_fills_markers_and_handles_missing_runs(monkeypatch, tmp_path):
    import run_eval
    monkeypatch.setattr(run_eval, "OUT", tmp_path)
    doc = tmp_path / "doc.md"
    doc.write_text(f"before\n{run_eval.START}\nold\n{run_eval.END}\nafter\n")
    run_eval.inject(doc, run_eval.results_body())
    text = doc.read_text()
    assert "old" not in text and "not run yet" in text and text.startswith("before") and text.rstrip().endswith("after")


def test_hf_revision_falls_back_without_network(monkeypatch):
    import run_eval
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert run_eval.hf_revision("x/y").startswith("unknown")


def test_summary_counts_top3_on_expected_page():
    rec = {"judgment": {"answerable": True, "retrieval_hit": True, "correct_answer": True},
           "expected_pdf_pages": [5], "retrieved_chunks": [{"pdf_page": 5}, {"pdf_page": 4}, {"pdf_page": 5}]}
    s = summarize([rec])
    assert s["top3_on_expected_page"] == 2 and s["top3_total"] == 3
