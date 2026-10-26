# Manual RAG over IACS Rec. 47 (PDF pages 4-8)

A small question-answering tool for one standard (IACS Recommendation 47, "Shipbuilding and Repair Quality Standard").
It finds the relevant manual text *before* answering, copies the answer from it with PDF page citations, and says
**"The supplied pages do not specify an answer."** when the manual does not contain the fact.

* **Task 1** - compare two ways of finding text: keyword matching vs pretrained embeddings.
* **Extension 1 (Laya)** - an AI model decides whether the retrieved text is enough to answer.
* **Extension 2 (LLM wiki)** - rebuild the same pages as a small cited wiki and search that instead of the PDF.

> Read **EXPLAINER.md** for how every file connects and how each part works, step by step, in plain language.

## Honest status (what was actually run)

| Part | Status |
|---|---|
| Task 1, keyword retrieval (dev + test) | **run**, results below |
| Task 1, embedding retrieval | code + tests written, **not run**: the build sandbox could not reach Hugging Face to download the model |
| Extension 2, wiki | **run**, results below. The wiki was written by Claude in a chat session, *not* through an API |
| Extension 1, Laya | code + tests written against Laya's documented API, **not run**: needs `torch` and Hugging Face weights. The "with Laya" cells are empty on purpose; nothing is mocked |

Every number below is produced by `python run_eval.py report` from the JSON files in `outputs/`.

| Task 1 | Keyword | Embedding |
|---|---|---|
| Dev retrieval hits | 5 / 5 | not run |
| Test retrieval hits | 3 / 3 | not run |
| Test correct supported answers | 2 / 3 | not run |
| Test correct missing-evidence responses | 2 / 2 | not run |

| Extension 2 (test, keyword retrieval) | PDF RAG | Wiki RAG |
|---|---|---|
| Correct supported answers | 2 / 3 | 1 / 3 |
| Correct abstentions | 2 / 2 | 2 / 2 |
| (dev) retrieval hits | 5 / 5 | 4 / 5 |

| Extension 1 (test, keyword retrieval) | Task 1 baseline | With Laya |
|---|---|---|
| Correct supported answers | 2 / 3 | not run |
| Correct abstentions | 2 / 2 | not run |
| Laya runtime errors | n/a | not run |

Keyword retrieval is the "selected" method because it is the only one with measured dev results (not because it is better).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                     # pymupdf, numpy, pytest  (enough for Task 1 keyword + wiki)
cp /path/to/IACS_47_Steel_Shipbuilding_Quality_2025.pdf materials/     # not committed to git
pip install -r requirements-extras.txt              # OPTIONAL: embeddings, Laya, wiki generation (large downloads)
```

## Run

```bash
python pdf_to_chunks.py                              # 1. PDF pages 4-8 -> outputs/chunks.json
python run_eval.py dev  --system pdf --method all    # 2. dev questions: keyword (and embedding if installed)
python run_eval.py test --system pdf --method all    # 3. FINAL test run (refuses to overwrite; --force to redo)

python wiki_tools.py                                 # Extension 2: check the wiki's links, citations, numbers
python run_eval.py dev  --system wiki
python run_eval.py test --system wiki

python run_eval.py dev  --system laya                # Extension 1 (needs `laya` + weights; else writes "not run")
python run_eval.py test --system laya

python run_eval.py report                            # tables -> outputs/results.md
python -m pytest -q                                  # 18 tests
```
Optional: `python make_wiki.py` regenerates the wiki with an API (`ANTHROPIC_API_KEY`, `LLM_MODEL`) into `wiki_generated/`.

## What is in the folder

```
config.py          every setting (top-3, abstain threshold, models ...) - the only place to tune
pdf_to_chunks.py   STEP 1  PDF pages 4-8 -> clean rows -> chunks
retrieve.py        STEP 2  keyword search + embedding search (same output shape)
answer.py          STEP 3  top-3 chunks -> copied answer + pages, or abstain
run_eval.py        STEP 4  run dev/test, score, save outputs/, print tables
wiki_tools.py      Extension 2: wiki files -> chunks, and wiki validation
make_wiki.py       Extension 2 (optional): generate the wiki with an LLM API
laya_gate.py       Extension 1: ask Laya "answer or abstain?"
wiki/              the wiki (index + 3 topic pages)        prompts/wiki_prompt.md   the wiki-writing prompt
data/              questions.json (as supplied), answer_keys.json (scoring keys)
outputs/           chunks, per-question results (retrieved text, scores, answers), results.md
tests/test_all.py  18 tests
```

## How an answer is produced (one paragraph)

The question goes to a retriever, which returns the 3 best chunks (each tagged with its PDF page). `answer.py` weighs
the question's content words by rarity and checks how much of that weight the best chunk contains; below 50 % it
abstains, otherwise it copies the best-matching sentence(s) (or the matching table row with its column header) and cites
the page. Scoring never trusts a page number alone: a "hit" needs the key evidence text inside a top-3 chunk on an
expected page.

## Dev vs test discipline

Settings in `config.py` and the helpers were tuned using the five **dev** questions only; the five **test** questions
were run once per system (`run_eval.py test` refuses to overwrite a result unless `--force`). Changes made during dev:
stop-words `difference`/`between` (dev q2 abstained), `MAX_SENTENCES` 3 -> 5 (dev q2 lost a sentence), stemmer rule for
`qualification ~ qualified` (found while testing dev q3). This simplified project is a rebuild of an earlier, larger
build; the rebuild reproduced the earlier dev and test numbers exactly, and its test run was therefore not "blind":
the settings were already fixed by the earlier build's dev tuning.

## Known mistakes and limitations

1. **Test q6 (PDF and wiki):** retrieval found the right chunk but the answer step copied the wrong sentence. The
   question mentions "fully mechanized or fully automatic ... processes", which matches the first sentence of clause 3.1.2;
   the sentence that actually answers ("Records of training and operation experience ...") shares only "records/training".
   Word overlap cannot tell what is being asked.
2. **Abstention rule is lightly validated:** test q7 passed with coverage 0.515 against a 0.5 threshold; the two
   unanswerable questions scored 0.15 and 0.22. Only two unanswerable examples exist, neither topically close.
3. **Wiki q8:** the wiki's table chunk lacks the words "gas cutting", so the cited intro bullet out-scored it and was returned
   instead of the table row. Wiki dev q2 also lost a three-way keyword tie.
4. Ten questions over five pages cannot establish that one method beats another.
5. Laya's own model card says the base checkpoint is near chance zero-shot and the strong numbers belong to the
   fine-tuned "typed-decisions" checkpoint, trained on other workflows. Whether it judges manual-evidence sufficiency
   well is untested here. Also, Laya only gates answer/abstain: it cannot repair a wrong-sentence choice like q6.

## Wiki provenance

`wiki/` was written by Claude Sonnet 5.5 in a chat session from the extracted page text, following
`prompts/wiki_prompt.md`. It was **not** produced by `make_wiki.py` (no API key), so there is no API model snapshot.
The author had already seen the evaluation questions, the wiki stays close to the manual's wording, and it was not edited
after any evaluation. `python wiki_tools.py` passes (44 claims). A human should spot-check it against the PDF.

## Other notes

* Scoring keys (`data/answer_keys.json`) were also written by the AI assistant; please review them.
* Laya: `requirements-extras.txt` pins `laya==0.4.15` (the release whose README the code follows). Checkpoint:
  `convaiinnovations/laya`, subfolder `typed-decisions`. `run_eval.py` records the installed Laya version in every
  Laya run; record the Hugging Face revision yourself when you run it.
* Time spent: not measured by the assistant. Enter your own figure here: ______
* AI help: the whole project (code, tests, wiki, scoring keys, docs) was written by Claude in chat; the tests and evaluations
  were run in a sandbox (18 tests pass).
