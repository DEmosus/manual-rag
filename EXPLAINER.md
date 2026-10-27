# EXPLAINER - how the project works and how every file connects

This document explains the whole project from zero, in plain language. It assumes no machine-learning background. For setup and results
see [README.md](README.md); for the short submission report see [REPORT.md](REPORT.md).

**Contents:** 1 The problem · 2 The big picture · 3 The data that flows between files · 4 File by file · 5 Two questions traced through the code ·
6 Dev and test: why they are kept apart · 7 How scoring works · 8 Extension 2: the wiki · 9 Extension 1: Laya · 10 Choosing the retriever ·
11 Why the failures happened · 12 How to change things safely · 13 Glossary

---

## 1. The problem in plain words

There is a long technical document and someone asks, *"Which qualification standard is named for welders?"* A trustworthy tool should:

1. **Find** the few places in the document that probably contain the answer. This is **retrieval**.
2. **Answer** using only those places, and say which PDF page it came from.
3. **Refuse** if the document does not contain the fact (for example *"What is the purchase price of a steel plate from EXTAA supplier A?"*)
   instead of guessing.

"Find text first, then answer from it" is called **RAG** (retrieval-augmented generation). In this assignment an answer may simply be a
sentence copied from the document, so no text-writing AI is needed. The project compares ways of doing step 1, adds an AI judge for step 3
(Laya), and tests a different source of text for step 1 (a wiki).

## 2. The big picture

```
materials/IACS...pdf
      │ pdf_to_chunks.py     read pages 4-8, clean, cut into pieces
      ▼
outputs/chunks.json  ── 22 "chunks": small pieces of text, each knowing its PDF page
      │
      │ retrieve.py          pick the 3 most relevant chunks   (two methods: keyword | embedding)
      ▼
top-3 chunks  (text, page, score, rank)
      │
      │ answer.py            enough evidence?  yes -> copy the best sentence(s) + cite pages ; no -> refuse
      ▼
answer + pages               or   "The supplied pages do not specify an answer."
      │
      │ run_eval.py          loop over questions, score each, save everything, print tables
      ▼
outputs/*.json  outputs/*.md  outputs/results.md  (and the tables inside README.md / REPORT.md)
```

The two extensions plug into the **same line**:

* **Extension 2 (wiki)** replaces the *source of chunks*: `wiki_tools.py` turns `wiki/*.md` into chunks of the same shape. Everything after
  that (retrieval, answer, scoring) is the same code.
* **Extension 1 (Laya)** replaces the *"enough evidence?" decision* inside `answer.py`: `laya_gate.py` asks the Laya model instead of using the
  50 % coverage rule. Retrieval and the sentence-copying are unchanged.

Because everything else is held fixed, a difference in results can be traced to the one thing that changed.

## 3. The data that flows between files

**A chunk** is a plain Python dictionary (the same shape for PDF chunks and wiki chunks):

| Field | Meaning |
|---|---|
| `id` | e.g. `p5-3.1.1-01` (page 5, clause 3.1.1, first piece) or `welder-qualifications-06` for the wiki |
| `page`, `pages` | the original 1-based PDF page (wiki chunks: the page(s) they cite) |
| `section` | the clause number (PDF) or the wiki heading |
| `heading` | parent headings, used only to help search ("3. Qualification ... > 3.1 Qualification of welders") |
| `kind` | `"text"` or `"table"` |
| `text` | the text answers are copied from |
| `file` | the wiki file (`wiki/welder-qualifications.md`), or `None` for the PDF |

**A retrieved item** is `{"chunk", "score", "rank", "method"}`. **An answer** is `{"decision", "answer", "pages", "citations", "chunk_id",
"coverage"}`. **A result record** (saved in `outputs/`) follows the task's output contract: `question_id`, `retrieved_chunks`, `decision`,
`answer`, `supporting_pdf_pages` (+ citations, judgment).

## 4. File by file

### `config.py` - the settings
Plain constants: the PDF path, pages 4-8, `TOP_K = 3`, the abstention sentence, `COVERAGE_THRESHOLD = 0.5`, `MAX_SENTENCES = 5`,
`SENTENCE_CUTOFF = 0.6`, the embedding model name, `SELECTED_METHOD`, and the Laya checkpoint. Every other file does `import config`.
This is the single place that controls behaviour. *Rule:* change it only while looking at the dev questions.

### `pdf_to_chunks.py` - Step 1: PDF to chunks
Input: the PDF. Output: `outputs/pages.json` (cleaned rows per page) and `outputs/chunks.json`.

* `read_rows` opens **only pages 4-8** (page *n* is code index *n-1*, but stored numbers stay the real 1-based ones). Problems it solves:
  * The PDF stores each table cell as a separate line. Lines at the same height are joined back into one row, so
    `Strength Members 150 µm 300 µm` stays one row.
  * Page furniture (running header/footer and the side block "No. 47 (cont)") sits at the same height as body text, so it is removed *before*
    grouping, and a line only joins a row if it is vertically aligned with that row's first line.
  * The font stores "≤" as a private-use character (U+F0A3); it is replaced. This is the only manual text correction.
* `make_chunks` cuts the rows into chunks:
  * every numbered clause (`1.1`, `3.1.1`, `5. Gas Cutting`) starts a new chunk, and a chunk never crosses a page;
  * a clause that is only a heading (e.g. "3.1 Qualification of welders") is not a chunk; it becomes the `heading` of the next one;
  * text at the top of a page before any clause (page 7) is a continuation chunk that inherits the previous section;
  * a block shaped like `Free Edges: / Standard Limit / rows` becomes a **table chunk** with the clause heading in front, so numbers stay
    attached to their column titles. Result: 22 chunks.

### `retrieve.py` - Step 2: pick the 3 best chunks
Two retrievers with the same output.

* **Text helpers.** `normalize` lower-cases and unifies odd characters (micro sign, curly quotes); `stem` shortens words (`approves`,
  `approved`, `approve` -> `approv`; `qualification`, `qualified` -> `qualif`); `tokens` splits text into stemmed words without common
  ones ("the", "of", ... and the question-framing words "difference", "between"); `search_text` = heading + text (what retrievers see).
* **`keyword_search` (the baseline).** Count how many *distinct* words the question and the chunk share. Highest wins; ties go to the earlier
  chunk in the document, so results never vary between runs.
* **`make_embedding_search`.** An embedding model turns a text into a list of numbers (a vector) so that texts with similar meaning get
  similar vectors. Each chunk is encoded once and kept in memory; the question is encoded; chunks are ranked by **cosine similarity**
  (how closely two vectors point the same way; with unit-length vectors it is just a dot product). `load_encoder` loads
  `sentence-transformers/all-MiniLM-L6-v2` and raises a clear error if it cannot - it never silently switches model. The encoder is passed in as a
  function, which is why the tests can use a tiny fake one.
* `make_search(method, chunks)` returns whichever search you ask for, with the same call signature.

### `answer.py` - Step 3: answer or abstain
Input: the question and the 3 retrieved chunks. Output: an answer record.

1. **Weights.** `build_idf` gives every word a weight from how many chunks contain it: rare words weigh more. A word found in no chunk
   (such as "price" or "telephone") gets the largest weight (4.14 with 22 chunks).
2. **Coverage.** For a chunk: (weight of the question's words found in the chunk) / (weight of all the question's words).
3. **Decision.** Take the best coverage among the three chunks. Below `COVERAGE_THRESHOLD` (0.5) -> abstain: the fixed sentence, no pages.
   A high rank alone never counts as evidence.
4. **Copy.** From the best-covering chunk, score each sentence the same way and copy the best one(s): up to 5, keeping those at least 60 % as good
   as the best, in document order (`split_sentences` does not split after "ref.", "e.g." or clause numbers). For a table chunk, `_table_answer`
   keeps the label, the `Standard Limit` header and only the row(s) that match the question.
5. **Cite.** Return the chunk's page(s), id and section (and the wiki file for wiki chunks).

`answer_question` never knows which retriever produced the chunks. `use_threshold=False` skips step 3 so an outside judge (Laya) can make that
call; steps 4-5 are unchanged.

### `run_eval.py` - Step 4: run, score, report, enforce the freeze
The command-line entry point. For one split (dev/test) and one system (pdf/wiki/laya) it:

1. checks the **freeze guard** (test runs only: needs a dev result with the same fingerprint, and never overwrites a finished result);
2. loads the questions (`data/questions.json`) and chunks (PDF chunks, or wiki chunks);
3. for every question: retrieve -> answer (or retrieve -> Laya -> answer) -> `judge`;
4. checks the output contract on every record and writes `outputs/<split>_<system>_<method>.json` (+ a readable `.md`, + the Laya log);
5. prints a short table. If a model cannot be loaded it writes `"status": "not_run"` with the error instead of pretending.

`fingerprint` is a sha256 over the files that must not change between the last dev run and the test run. `report` builds the tables from the
saved JSON and writes them to `outputs/results.md`, `README.md` and `REPORT.md` (between the `RESULTS` markers). `hf_revision` records the
Hugging Face commit of the embedding model and the Laya checkpoint (the "version" of the downloaded weights).

### `wiki_tools.py` - Extension 2 helpers
`load_wiki_chunks` reads `wiki/*.md` (not `index.md`, which is navigation) and produces chunks in the shape of section 3: one per cited bullet,
one per cited table. It strips the `(Source: ... PDF page N.)` tag from the text but remembers the pages and the file name. `table_text`
rewrites a markdown table into the same `Label: / header / rows` text as the PDF table chunks, so the same answer code works. `validate_wiki`
checks files, links, citations on every fact, page range 4-8, and that every number in a claim appears on the cited PDF page.

### `laya_gate.py` - Extension 1
`build_state` writes the question and the three chunks (with page numbers) as one text. `LayaGate.decide` sends it to Laya together with the
question in `prompts/laya_question.json` and returns the label, the exact input, the raw output and any error. Any failure or unexpected label
becomes `abstain` + a recorded error.

### `run_all.py`
Runs the protocol in order: chunks -> dev (both retrievers) -> pause to look at dev -> dev for wiki and Laya -> all test runs -> report ->
pytest -> `pip freeze > requirements-lock.txt`.

### The folders
`data/` questions as supplied and scoring keys · `prompts/` the saved wiki prompt, wiki provenance note, frozen Laya question ·
`wiki/` the wiki · `outputs/` everything produced · `tests/test_all.py` 24 tests · `materials/` where the PDF goes (git-ignored).

## 5. Two questions traced through the code

### Dev q3 - an answerable question
*"Which qualification standard is named for welders?"* (expected page 5)

1. **Chunk.** Clause 3.1.1 became chunk `p5-3.1.1-01`; its `heading` is "3. Qualification of personnel and procedures > 3.1 Qualification of
   welders" and its text begins "3.1.1 Welders are to be qualified in accordance with UR W32 (ref.A18)...".
2. **Tokens.** The question becomes four words: `named`, `qualif`, `standard`, `welder`.
3. **Retrieve (keyword).** The chunk shares `qualif`, `standard` and `welder` (3 words, helped by its heading); the next two chunks share 2.
   Top 3: `p5-3.1.1-01` (3.0), `p4-1.4-01` (2.0), `p5-2.3-01` (2.0).
4. **Decide.** Weights: named 4.14 (appears nowhere), qualif 2.34, standard 1.65, welder 3.04; total 11.17. Found: 2.34 + 1.65 + 3.04 = 7.03.
   Coverage 7.03 / 11.17 = **0.63 >= 0.5** -> answer.
5. **Copy and cite.** The best two sentences are copied ("3.1.1 Welders are to be qualified in accordance with UR W32 ... Classification
   Society. Subcontractors are to keep records of welders qualification ..."), citing PDF page 5.
6. **Score.** The chunk is on page 5 and contains "UR W32" -> retrieval hit; the answer contains "UR W32" and cites page 5 -> correct.

### Test q9 - a question that must be refused
*"What is the purchase price of a steel plate from EXTAA supplier A?"*

1. Retrieval still returns three chunks (it always does): `p4-1.1-01`, `p7-4.2.2-01`, `p8-4.2.4.1-01`. Ranking cannot say "nothing is relevant".
2. The question's words are `extaa`, `plat`, `pric`, `purchas`, `steel`, `supplier`. Four of them (`extaa`, `pric`, `purchas`, `supplier`) appear nowhere in the
   manual and carry the maximum weight 4.14 each.
3. The best chunk contains only `steel` (3.44 of a total 22.75) -> coverage **0.15 < 0.5** -> the fixed refusal sentence, no pages.

This is why *ranking* and *evidence checking* are separate steps. Test q10 ("emergency telephone number for the shipyard") behaves the same: only
"shipyard" is found, coverage 0.22.

## 6. Dev and test: why they are kept apart

Tuning on the questions you are graded on would inflate the score. So the five **dev** questions (q1-q5) were used to improve the system, and
the five **test** questions (q6-q10) are run once the settings are frozen. The code enforces this: `run_eval.py test` refuses to run unless a dev
result exists whose fingerprint equals the current one (so *any* change to config, retrieval, answering, questions, keys, wiki files, wiki prompt
or Laya prompt forces a fresh dev run first), and it never overwrites a finished test result without `--force`. Three changes came from dev: the
stop-words `difference`/`between`, `MAX_SENTENCES` 3 -> 5, and a stemmer rule for `qualification`/`qualified`. README section 13 states the honest
limits of this (several iterations of the project; the AI author had seen the test questions).

## 7. How scoring works

`judge` in `run_eval.py` compares each result with `data/questions.json` (expected pages) and `data/answer_keys.json` (key strings; written by the
AI assistant, to be reviewed by a human). The key strings are used **only** for scoring, never by retrieval or answering.

* **Retrieval hit** (answerable questions): a top-3 chunk is on an expected page **and** contains the evidence strings. Page alone is not enough.
* **Correct supported answer:** answered, the answer contains the key strings, cites an expected page. An abstention on an answerable question
  earns no credit.
* **Correct abstention** (unanswerable questions): abstained, no pages. If Laya crashed and the system abstained because of the crash, that is
  counted as a Laya runtime error, **not** a correct abstention.
* **Extra statistic:** how many top-3 chunks are on an expected page (used only to describe retrievers; not part of the required metrics).

## 8. Extension 2: the wiki

**What it is.** A few short, linked Markdown notes written from the same five pages: `welder-qualifications.md`, `materials-and-standards.md`,
`surface-quality-and-cutting.md`, plus `index.md` (links only). 44 cited claims.

**Rules it follows** (`prompts/wiki_prompt.md`): use only the source; one fact per bullet; every bullet ends with
`(Source: IACS Rec. 47 Part A, Sec. X, PDF page N.)`; keep numbers, units and table headings; link related pages.

**How it is searched.** `wiki_tools.load_wiki_chunks` makes one chunk per bullet and one per table. `run_eval.py --system wiki` then applies the
**same retriever and same answer code**. Answers carry the wiki file and the PDF page; the retrieved wiki text is saved with each answer.

**How it is checked.** `python wiki_tools.py`: all files exist, links resolve, every fact has a citation, cited pages are 4-8, and every number
appears on the cited PDF page. A real catch while building it: a bullet said "the two tables below", which is not in the source, so it was reworded.

**Provenance and honesty.** Written by Claude Sonnet 5.5 in a chat session (`prompts/wiki_provenance.md`), after the author had seen the evaluation
questions, so the comparison is not blind. The wiki mostly copies the manual's wording, so its main effect is **chunk size** (one claim vs one clause).
Five pages say nothing about larger collections.

## 9. Extension 1: Laya

**What Laya is.** An open model (PyPI `laya`, author Convai Innovations; checkpoints on Hugging Face) that reads a text and answers *typed* questions
in a single pass without writing text. A `choice` question returns one of the options you define.

```python
import laya
agent  = laya.load("convaiinnovations/laya", subfolder="typed-decisions")
result = agent.predict(state, {"enough_evidence": {"type": "choice", "instructions": "...",
                               "criteria": {"answer": "...", "abstain": "..."}}})
label  = result["answers"]["enough_evidence"]["choice"]        # "answer" or "abstain"
```

**How it is wired** (`laya_gate.py` + `run_eval.py --system laya`):

1. Retrieval is the selected Task 1 method, unchanged. Its outputs are the **baseline**; the same questions and the same retrieved chunks go to Laya.
2. The state is `Question: ...` followed by `Evidence 1 (PDF page N): ...` for each of the three chunks.
3. Label `answer` -> `answer_question(..., use_threshold=False)`: the normal sentence-copying code runs, but the built-in coverage check is switched
   off because Laya is now the judge. Label `abstain` -> the fixed sentence, no pages.
4. Crash, invalid output or unexpected label -> abstain, error recorded, counted as a **Laya runtime error** (never as a correct abstention).
5. Each question's actual Laya **input, raw output, label and error** are saved in `outputs/<split>_laya_<method>_laya_log.json`, with the Laya
   version, checkpoint, its Hugging Face revision and a hash of the question prompt. The question text itself is frozen in
   `prompts/laya_question.json`.

**Practical notes.** The checkpoint is 847 MB and needs `torch`. On load Laya warns that this checkpoint ships invalid temperatures for some heads and
that its confidence values are uncalibrated; this project uses only the label, never the confidence.

**What Laya can and cannot change.** It only moves the answer/abstain line. It cannot fix a case where the right chunk was found but the wrong sentence
was copied (test q6), and a typed label "makes the output predictable; it does not guarantee a correct decision".

## 10. Choosing the retriever

The extensions need one retriever. The rule (fixed on the required dev metrics): highest dev retrieval hits; tie -> higher dev correct answers;
still tied -> keyword. Dev: both retrievers scored 5/5 hits and 5/5 answers, so **keyword** is selected. An extra statistic favours embedding (11 vs 9 of
15 top-3 chunks on an expected page) but was not used to decide. The tie-break was written after seeing the tie; keyword was already the working
default. To switch: set `SELECTED_METHOD = "embedding"` in `config.py` and run `python run_all.py --force`.

## 11. Why the failures happened

* **q6 (PDF and wiki): right chunk, wrong sentence.** The retrieved chunk `p5-3.1.2-01` has coverage 0.89 and is correct. Inside it, the first sentence
  mentions "fully mechanized or fully automatic processes", which the question repeats, so it scores highest. The sentence that answers ("Records of
  training and operation experience shall be maintained...") only shares "records" and "training". Counting words cannot tell what is being asked.
  This is a weakness of the *answer* step, not of retrieval.
* **Wiki q8.** In the wiki the gas-cutting numbers live in table chunks that contain only the label, header and rows. The cited intro bullet ("The
  roughness of the cut edges is to meet the following requirements...") contains more of the question's words (coverage 0.80), so it was copied instead.
  In the PDF the table chunk includes the clause heading and intro, and the answer (`Free Edges: Standard Limit Strength Members 150 µm 300 µm`) was correct.
* **Wiki dev q2.** The wiki splits text into many small bullets; three of them tied on keyword score and document order picked the wrong ones.
* **q9, q10 passed** for the reasons in section 5. **q7 passed narrowly** (coverage 0.515), which shows the threshold is not stress-tested.

## 12. How to change things safely

* *Change a setting:* edit `config.py`, run `python run_eval.py dev --system pdf --method all`, read `outputs/dev_*.md` (it lists the retrieved text and
  coverage). Do this **before** any test run; the freeze guard will force it anyway.
* *Add a question:* add it to `data/questions.json` with matching `evidence`/`answer` strings in `data/answer_keys.json`.
* *Change the wiki:* edit `wiki/*.md`, run `python wiki_tools.py`, then re-run the wiki dev and test.
* *Run everything again:* `python run_all.py --force`.

## 13. Glossary

**Chunk** - a small piece of the document used as the unit of search · **Retrieval** - choosing chunks for a question · **Top-3** - the three best-ranked
chunks · **Keyword baseline** - ranking by shared words · **Embedding** - numbers that represent the meaning of a text · **Cosine similarity** - how alike
two embeddings are · **IDF weight** - a word's rarity score · **Coverage** - the share of the question's weight found in a chunk · **Abstain** - refuse
because the evidence is missing · **Retrieval hit** - a top-3 chunk with the key evidence on an expected page · **Dev / test** - the tuning questions / the
final questions · **Freeze guard / fingerprint** - the check that nothing changed between the last dev run and the test run · **Wiki** - short linked,
cited notes written from the source · **Laya** - a model that returns typed decisions (choice / score / yes-no) in one pass.
