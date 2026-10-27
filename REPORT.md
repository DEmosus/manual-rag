# Short report

Project: manual RAG over IACS Rec. 47 (PDF pages 4-8) with a keyword/embedding comparison, a Laya answer/abstain gate (Extension 1) and an LLM wiki
(Extension 2). Setup and commands: [README.md](README.md). How everything works: [EXPLAINER.md](EXPLAINER.md).

## Approach

- **Chunks:** the pages are cut into 22 clause-level chunks that keep their original PDF page; the gas-cutting tables keep their `Standard Limit` header with
  their values.
- **Retrieval (identical chunks, top 3):** keyword = count of distinct shared words; embedding = cosine similarity of
  `sentence-transformers/all-MiniLM-L6-v2` vectors held in memory.
- **Answer and abstention (identical for every retriever):** question words are weighted by rarity in the manual; if the best top-3 chunk covers less than 50 %
  of that weight the system returns "The supplied pages do not specify an answer."; otherwise it copies the best sentence(s) (or table row with its header) and cites the
  PDF page. A high rank alone is never treated as evidence.
- **Extension 1:** Laya (`convaiinnovations/laya`, `typed-decisions`) receives the question and the three retrieved chunks and chooses `answer` or `abstain`; errors abstain and are
  counted separately. **Extension 2:** the same retriever and answer code run over a cited wiki (index + 3 topic pages), answers cite wiki file and PDF page.
- **Method:** tune on the 5 dev questions, choose the retriever from dev results (rule in README section 6), freeze (fingerprint check in `run_eval.py`), run the 5 test
  questions once per system.

## Results

<!-- RESULTS:START -->

Selected retrieval method for the extensions: **keyword** (rule: README, "Choosing the retrieval method").

**Task 1 - keyword vs embedding**

| Result                                          | Keyword baseline | Embedding retrieval |
| ----------------------------------------------- | ---------------- | ------------------- |
| Dev retrieval hits / 5                          | 5 / 5            | 5 / 5               |
| Dev correct supported answers / 5               | 5 / 5            | 5 / 5               |
| Dev top-3 chunks on an expected page (extra)    | 9 / 15           | 11 / 15             |
| **Test retrieval hits / 3**                     | 3 / 3            | 3 / 3               |
| Test correct supported answers / 3 (extra)      | 2 / 3            | 2 / 3               |
| **Test correct missing-evidence responses / 2** | 2 / 2            | 2 / 2               |

**Extension 1 - Laya gate** (retrieval: keyword)

| Result                                 | Task 1 baseline | With Laya |
| -------------------------------------- | --------------- | --------- |
| Dev correct supported answers / 5      | 5 / 5           | 5 / 5     |
| **Test correct supported answers / 3** | 2 / 3           | 1 / 3     |
| **Test correct abstentions / 2**       | 2 / 2           | 2 / 2     |
| **Laya runtime errors (test)**         | N/A             | 0         |
| Laya runtime errors (dev)              | N/A             | 0         |

**Extension 2 - wiki** (retrieval: keyword)

| Result                                 | Task 1: PDF RAG | Wiki RAG |
| -------------------------------------- | --------------- | -------- |
| Dev retrieval hits / 5                 | 5 / 5           | 4 / 5    |
| Dev correct supported answers / 5      | 5 / 5           | 3 / 5    |
| **Test correct supported answers / 3** | 2 / 3           | 1 / 3    |
| **Test correct abstentions / 2**       | 2 / 2           | 2 / 2    |

**Environment of the last runs** (recorded in every `outputs/*.json`)

- Python 3.13.3; pymupdf 2.0, numpy 2.4.4, sentence-transformers 6.2.0, transformers 5.21.0, huggingface-hub 2.3.0, torch 2.14.1, laya 0.4.15
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- Laya checkpoint: `convaiinnovations/laya` / `typed-decisions` @ `7b928d828b7b0e022f929d9bd2e44165aa270148`
<!-- RESULTS:END -->

## Mistakes and limitations (actual)

1. **Test q6 (PDF and wiki): right chunk, wrong sentence.** Retrieval found clause 3.1.2 but the copied sentence was the first one (about operators not needing approval
   testing), because it repeats the question's "fully mechanized or fully automatic processes"; the sentence that answers ("Records of training and operation
   experience ...") shares only "records" and "training". Word overlap cannot tell what is asked.
2. **The abstention rule is lightly validated.** Test q7 passed at coverage 0.515 against the 0.5 threshold; the unanswerable q9 and q10 scored 0.15 and 0.22. Only two unanswerable
   examples exist and neither is topically close.
3. **Wiki test q8:** the wiki's table chunk lacks the words "gas cutting", so the cited intro bullet (coverage 0.80) was copied instead of the values; the PDF chunk keeps heading and
   intro with the table and was correct. Wiki dev q2 also lost a three-way keyword tie because the wiki splits text into many small bullets.
4. **Laya (remaining limitation):** it only decides answer vs abstain, so with the same retrieved chunks and the same sentence-copying code it cannot make q6 correct. Its
   confidence output is uncalibrated for this checkpoint (Laya's own warning); only the label is used.
5. Ten questions over five pages cannot show that one method is better than another.

## Manual corrections

Only the font glyph U+F0A3 -> "≤" in Table 1 (page 6). Wiki: one bullet was reworded because it contained wording not in the source ("the two tables below").

## Models, versions, environment variables

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (revision recorded in `outputs/*.json` and in the results block above).
- Laya: package `laya==0.4.15`; checkpoint `convaiinnovations/laya`, subfolder `typed-decisions` (847 MB; revision recorded in the Laya log).
- Wiki author: Claude Sonnet 5.5 (chat session; no API call, so no snapshot id or sampling parameters). Prompt: `prompts/wiki_prompt.md`.
- Pinned core: `pymupdf==2.0`, `numpy==2.4.4`, `pytest==9.1.1`; extras `sentence-transformers>=3.0`, `laya==0.4.15`; exact versions of a finished run in `requirements-lock.txt`.
- Environment variables: **none required**; `HF_TOKEN` optional (Hugging Face rate limits).

## AI and other help

Claude (Anthropic) wrote the code, tests, wiki, scoring keys and documentation in a chat session; the embedding and Laya runs were executed on the author's machine. The scoring
keys were written by the assistant and should be reviewed by a human. The wiki author had seen the evaluation questions beforehand, so the wiki comparison is not blind. This
repository is the final simplified version of an earlier larger build that gave the same dev and test numbers; test results were regenerated after format changes, so the test
questions were seen across iterations.
