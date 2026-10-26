# EXPLAINER - how everything works and how the files connect

This explains the project from zero. No machine-learning background is assumed.

## 1. The problem in plain words

You have a long technical document. Someone asks, "Which qualification standard is named for welders?". A tool should:

1. **Find** the few places in the document that probably contain the answer (this is *retrieval*).
2. **Answer** using only those places, and say which PDF page it came from.
3. **Refuse** when the document does not contain the fact (for example "What is the purchase price of a steel plate from
   EXTAA supplier A?"), instead of making something up.

"Finding text first, then answering from it" is called **RAG** (retrieval-augmented generation). In this assignment the
"answer" is simply a sentence copied from the document, so no text-generating AI is required.

## 2. The whole flow on one page

```
 materials/IACS...pdf
        │  pdf_to_chunks.py   (read pages 4-8, clean, cut into pieces)
        ▼
 outputs/chunks.json  ← 22 "chunks": small pieces of text, each knowing its PDF page
        │
        ├────────────── retrieve.py ──────────────┐   two ways to pick the 3 best chunks
        │      keyword_search     embedding search │
        ▼                                          ▼
   top-3 chunks (text, page, score, rank)  ◄───────┘
        │
        │  answer.py   (is there enough evidence? then copy the best sentence; else abstain)
        ▼
   answer + pages + citation   or   "The supplied pages do not specify an answer."
        │
        │  run_eval.py   (loop over the questions, score each one, save everything, print tables)
        ▼
   outputs/dev_*.json, test_*.json, results.md
```

The two extensions plug into this same line:

* **Extension 2 (wiki)** replaces the *left end*: instead of PDF chunks, the chunks come from `wiki/*.md`
  (loaded by `wiki_tools.py`). Everything after that is the same code.
* **Extension 1 (Laya)** replaces the *"enough evidence?" decision* in `answer.py` with Laya's decision
  (called from `laya_gate.py`). Retrieval and sentence copying are unchanged.

Because both extensions reuse the same retrieval, answering and scoring code, any difference in the results is caused
by the thing that changed, not by accident.

## 3. The files, one by one

### `config.py` - the settings
A list of constants: the PDF path, pages 4-8, `TOP_K = 3`, the abstention sentence, `COVERAGE_THRESHOLD = 0.5`,
`MAX_SENTENCES = 5`, the embedding model name, which retriever is `SELECTED_METHOD`, and which Laya checkpoint to use.
Every other file imports it (`import config`). It is the single place you change behaviour. *Rule:* change it only while
looking at the dev questions; after you run the test questions, leave it alone.

### `pdf_to_chunks.py` - STEP 1: PDF to chunks
Input: the PDF. Output: `outputs/pages.json` (the cleaned text rows of each page) and `outputs/chunks.json`.

* `read_rows` opens only pages 4-8 (code index n-1, but all stored page numbers are the real 1-based ones). The PDF
  stores each table cell as a separate line, so lines at the same height are joined back into one row
  (`Strength Members 150 µm 300 µm`). It removes page furniture (the running header, footer and the side block
  "No. 47 (cont)") and repairs one font glitch (the "≤" sign is stored as a private-use character).
* `make_chunks` cuts the rows into chunks:
  * every numbered clause (`1.1`, `3.1.1`, `5. Gas Cutting`) starts a new chunk; a chunk never crosses a page;
  * a clause that is only a heading (e.g. "3.1 Qualification of welders") is not a chunk of its own; it becomes the
    `heading` of the next chunk so retrieval still "sees" it;
  * a block shaped like `Free Edges: / Standard Limit / rows` becomes a **table chunk**, with the clause heading in
    front, so the numbers always stay attached to their column titles.

A chunk is a small dictionary: `id` (like `p5-3.1.1-01`), `page`, `pages`, `section`, `heading`, `kind`
("text" or "table"), `text`, `file` (the wiki file, or `None` for the PDF).

### `retrieve.py` - STEP 2: pick the 3 best chunks
Two retrievers with the same output: a list of `{"chunk", "score", "rank", "method"}`.

* **Text helpers** (`normalize`, `stem`, `tokens`): lower-case, unify odd characters, drop common words
  ("the", "of", ...), and shorten words so `approves/approved/approve` match.
* **`keyword_search` (the baseline):** count how many distinct words the question and the chunk share; highest count
  wins; ties are broken by document order so results never change between runs.
* **`make_embedding_search`:** an embedding model turns text into a list of numbers (a vector) so that texts with similar
  meaning get similar vectors. Each chunk is turned into a vector once; the question is turned into a vector; chunks are
  ranked by **cosine similarity** (how closely two vectors point in the same direction). `load_encoder` loads the
  pretrained model `sentence-transformers/all-MiniLM-L6-v2` and raises a clear error if it cannot (it never silently uses
  another model). The encoder is passed in as a function, which is why the tests can use a tiny fake one.
* `make_search(method, chunks)` returns whichever one you ask for, with the same call signature.

### `answer.py` - STEP 3: answer or abstain
Input: the question and the 3 retrieved chunks. Output: `{decision, answer, pages, citations, chunk_id, coverage}`.

1. **Weights.** `build_idf` looks at all chunks being searched and gives each word a weight: rare words weigh more.
   A word that appears nowhere (like "price" or "telephone") gets the biggest weight.
2. **Coverage.** For a chunk, coverage = (weight of the question's words found in the chunk) ÷ (weight of all the
   question's words). So 1.0 means every question word is there; 0.15 means mostly not.
3. **Decision.** Take the best coverage among the three chunks. Below `COVERAGE_THRESHOLD` (0.5) → abstain: the answer is
   the fixed sentence, with no pages. A high rank alone never counts as evidence.
4. **Copy the answer.** From the best chunk, score each sentence the same way and copy the best one(s) (up to 5,
   keeping those at least 60 % as good as the best). For a table chunk, keep the header line and only the matching row.
5. **Cite.** Return the chunk's page(s) and its id/section (and the wiki file for wiki chunks).

It never knows which retriever produced the chunks, so the keyword and embedding results are comparable. With
`use_threshold=False` it skips step 3 so that someone else (Laya) can make that decision; steps 4-5 are unchanged.

### `run_eval.py` - STEP 4: run and score
The command-line entry point (`python run_eval.py dev|test|report ...`). For one split and one system it:

1. loads the questions for that split (`data/questions.json`, exactly as supplied);
2. loads chunks (PDF chunks, or wiki chunks for `--system wiki`);
3. for every question: retrieve → answer (or retrieve → Laya → answer) → **judge**;
4. writes `outputs/<split>_<system>_<method>.json` with, for each question, the three retrieved chunks (text, page,
   score), the answer, the citations, the judgment, and run info (Python/package versions, git commit, settings);
5. prints a short table. `test` runs refuse to overwrite an existing result unless `--force`.

If a model cannot be set up (no `sentence-transformers`, no `laya`, no internet), the file records
`"status": "not_run"` plus the error, instead of pretending. `report` reads those files and builds `outputs/results.md`.

**How scoring works** (`judge`, using `data/answer_keys.json`; keys are only used here, never by the retrievers):

* *Retrieval hit* (answerable questions only): at least one top-3 chunk is on an expected page **and** contains the key
  evidence text. A matching page number alone is not enough.
* *Correct supported answer*: it answered, the answer contains the key strings, and it cites an expected page. (Answers
  are copied from retrieved text, so they are supported by it by construction.)
* *Correct abstention* (unanswerable questions only): it abstained. If Laya crashed and the system abstained because of
  the crash, that does **not** count as a correct abstention; it is counted as a Laya error.

### `wiki_tools.py` - Extension 2 helpers
* `load_wiki_chunks` reads `wiki/*.md` (not `index.md`, which is navigation) and produces chunks in the same dictionary
  shape as the PDF chunks: one chunk per cited bullet, one per cited table. It strips the `(Source: ... PDF page N.)`
  tag from the text but remembers the page(s) in `pages` and the file name in `file`.
* `validate_wiki` (run it with `python wiki_tools.py`) checks: all four files exist; links resolve; every fact line has a
  citation; cited pages are 4-8; every number in a claim appears on the cited PDF page (in `outputs/pages.json`).

### `make_wiki.py` - Extension 2 generation (optional)
Fills `prompts/wiki_prompt.md` with the page text, sends it to an LLM through the Anthropic API, splits the reply into files
and saves them in `wiki_generated/` with the model name. It needs an API key, so it was **not** used for the committed
wiki (see section 7).

### `laya_gate.py` - Extension 1
Wraps Laya (section 8). `build_state` writes the question and the three chunks (with page numbers) as one text;
`LayaGate.decide` asks Laya one typed `choice` question with two labels, `answer` / `abstain`, and returns the label, the
exact input sent, the raw output, and any error. Any failure (or an unexpected label) becomes `abstain` + a recorded error.

### The folders
`data/` (questions as supplied; scoring keys) · `wiki/` (the wiki) · `prompts/` (the wiki prompt) · `outputs/` (everything
produced) · `materials/` (put the PDF here; git-ignored) · `tests/test_all.py` (18 tests: chunking, both retrievers with a
fake encoder, answer/abstain, scoring, the wiki helpers, the Laya gate and its wiring with a fake agent).

## 4. Following one question through the code (dev q3, real values)

Question: *"Which qualification standard is named for welders?"* (expected page 5)

1. **Chunks.** `pdf_to_chunks.py` made chunk `p5-3.1.1-01` from clause 3.1.1; its `heading` is
   "3. Qualification of personnel and procedures > 3.1 Qualification of welders" and its text starts "3.1.1 Welders are to
   be qualified in accordance with UR W32 (ref.A18)...".
2. **Retrieve.** After `tokens(...)`, the question becomes {qualif, standard, name, welder}. Keyword search counts shared
   words per chunk: `p5-3.1.1-01` shares 3 (qualif and welder via its heading + text, standard), the next two chunks share 2.
   Top 3 = `p5-3.1.1-01` (3.0), `p4-1.4-01` (2.0), `p5-2.3-01` (2.0).
3. **Decide.** Coverage of the best chunk = 0.63 ≥ 0.5 → answer.
4. **Copy.** Sentences of that chunk are scored; the best two are copied: "3.1.1 Welders are to be qualified in accordance with
   UR W32 (ref.A18) or other recognized standard accepted by the Classification Society. Subcontractors are to keep records
   of welders qualification and ...", citing PDF page 5.
5. **Judge.** The chunk is on page 5 and contains "UR W32" → hit. The answer contains "UR W32" and cites page 5 → correct.

And a question that must be refused - test q9 *"What is the purchase price of a steel plate from EXTAA supplier A?"*:
retrieval still returns three chunks (it always returns three), e.g. `p4-1.1-01`; but "purchase", "price" and "EXTAA" occur
nowhere in the manual, so they carry the biggest weights and are all missing. Best coverage = 0.15 < 0.5 → the answer is the
fixed refusal sentence with no pages. That is the whole point of separating *ranking* from *evidence check*.

## 5. Dev questions, test questions, and why they are kept apart
Tuning on the questions you are graded on would inflate the score. So: the five **dev** questions (q1-q5) were used to
improve the system; then settings were left alone and the five **test** questions (q6-q10) were run once. Three changes came
from dev (written in README). `run_eval.py test` will not overwrite a finished test result without `--force`, which makes
accidental re-tuning visible. Caveat (also in the README): this project is a rebuild of an earlier version whose test
questions had already been run, so this rebuild's test run is a reproduction, not a blind trial.

## 6. Reading the results (and why the failures happened)
* **PDF q6 failed although retrieval hit.** The right chunk was found, but sentence selection picked the first sentence of
  3.1.2 (it contains "fully mechanized or fully automatic processes", which the question repeats) instead of the later
  sentence "Records of training and operation experience ...". Word counting cannot tell what is being asked. This is a
  weakness of the *answer* step, not of retrieval.
* **Wiki q8 failed.** In the wiki, the gas-cutting values live in a table chunk whose text is only the label, the header and
  the rows; the cited intro bullet ("The roughness of the cut edges ...") contains more of the question's words, so the
  answer step chose it and returned a sentence without the numbers. In the PDF the table chunk includes the heading and
  intro, so this did not happen.
* **Wiki dev q2 missed** because the wiki splits text into many tiny bullets and three of them tied on score.
* **q9 and q10 passed** for the reasons in section 4.

## 7. Extension 2 in detail - the LLM wiki
*What it is:* a handful of short, linked Markdown notes written from the same five pages: `welder-qualifications.md`,
`materials-and-standards.md`, `surface-quality-and-cutting.md`, plus `index.md` (links only).

*Rules it follows* (`prompts/wiki_prompt.md`): use only the source; one fact per bullet; every bullet ends with
`(Source: IACS Rec. 47 Part A, Sec. X, PDF page N.)`; keep the manual's numbers, units and table headings; link related pages.

*How it is searched:* `wiki_tools.load_wiki_chunks` turns each bullet into a chunk (and each table into one table chunk).
`run_eval.py --system wiki` then uses the **same retriever and the same answer step** as the PDF system. Wiki answers carry
both the wiki file name (`file`) and the original PDF page (`pages`), as required.

*How it is checked:* `python wiki_tools.py` (links, citations, page range, numbers traceable to the cited page). One
mistake was caught while building it: a bullet said "the two tables below", which is not in the source; it was reworded.

*Provenance:* the committed wiki was written by Claude in a chat session, not by `make_wiki.py`, and Claude had seen the
questions beforehand. So the wiki-vs-PDF comparison is **not blind**, and because the wiki copies the manual's wording, its
main effect is changing the chunk size (one bullet vs one clause), not adding knowledge. Five pages also say nothing about
how this would scale.

## 8. Extension 1 in detail - Laya as the "enough evidence?" judge
*What Laya is* (from its documentation): an open model that reads a text plus typed questions and answers them in a single
pass, without writing text. A `choice` question returns one of your named options. Usage:

```python
import laya
agent  = laya.load("convaiinnovations/laya", subfolder="typed-decisions")
result = agent.predict(state, {"enough_evidence": {"type": "choice", "instructions": "...",
                               "criteria": {"answer": "...", "abstain": "..."}}})
label  = result["answers"]["enough_evidence"]["choice"]       # "answer" or "abstain"
```

*How it is wired* (`laya_gate.py` + `run_eval.py --system laya`):
1. Retrieval is the selected Task 1 method, unchanged (top-3 chunks).
2. The state sent to Laya is `Question: ...` followed by `Evidence 1 (PDF page N): ...` for the three chunks.
3. Label `answer` → `answer_question(..., use_threshold=False)`: the normal sentence-copying code runs, but the
   built-in 50 % coverage gate is switched off, because Laya is now the judge of sufficiency.
   Label `abstain` → the fixed refusal sentence, no pages.
4. If Laya cannot load, crashes, or returns an unknown label → abstain and store the error. Such abstentions are counted as
   "Laya runtime errors", never as correct abstentions.
5. Every question's Laya input, raw output, label and error are saved in the result JSON, together with the Laya package version.

*Why "with Laya" is empty in the results:* Laya needs `torch` and its weights from Hugging Face; the sandbox had neither, so
the run is recorded as `not run`. What *was* tested: the gate's logic and the full pipeline path using a **fake agent**
(clearly a code-path test only, not a Laya result). When you run it for real, the baseline and "with Laya" columns use the
same questions and the same retrieved chunks.

*What to expect and not expect:* Laya can only change answer-vs-abstain. It cannot fix wrong-sentence cases like q6. And its
authors say the base model is near chance zero-shot and that the strong accuracy belongs to the fine-tuned typed-decisions
checkpoint, trained on other kinds of data - so the effect on this task is unknown until it is run.

## 9. Things you can do next
* Run the embedding comparison: `pip install -r requirements-extras.txt`, then `python run_eval.py dev --system pdf --method embedding`
  and `... test ...`. (If you want the selection to follow the embedding dev result, decide before looking at its test
  result, and change `SELECTED_METHOD` in `config.py`.)
* Run Laya: same install, then `python run_eval.py dev --system laya`, then `test`.
* Add questions to `data/questions.json` with matching keys in `data/answer_keys.json`.
* Change a setting: edit `config.py`, re-run `dev`, look at `outputs/dev_*.json` (it lists the retrieved text and the coverage).

## 10. Glossary
**Chunk** a small piece of the document used as the unit of search · **Retrieval** choosing chunks for a question ·
**Top-3** the three best-ranked chunks · **Keyword baseline** ranking by shared words · **Embedding** numbers representing
the meaning of a text · **Cosine similarity** how alike two embeddings are · **IDF weight** a word's rarity score ·
**Coverage** share of the question's weight found in a chunk · **Abstain** refuse because the evidence is missing ·
**Retrieval hit** a top-3 chunk with the key evidence on an expected page · **Dev / test** the tuning questions / the
final questions · **Wiki** short linked, cited notes written from the source · **Laya** a model that returns typed
decisions (choice / score / yes-no) in one pass.
