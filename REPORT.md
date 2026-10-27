# Short report

**Approach.** PDF pages 4-8 are cut into 22 clause-level chunks (original page numbers, gas-cutting tables kept with their
`Standard Limit` header). Two retrievers return the top 3: keyword (shared-word count) and embedding
(`sentence-transformers/all-MiniLM-L6-v2`, cosine). One shared answer step weighs question words by rarity; if the best chunk
covers < 50 % of that weight it returns "The supplied pages do not specify an answer.", otherwise it copies the best sentence(s)
with PDF-page citations. Extension 2 searches a cited wiki instead of the PDF; Extension 1 lets Laya make the answer/abstain call.
Dev questions were used for tuning, test questions once per system (freeze guard in `run_eval.py`). Details: `EXPLAINER.md`.

## Results (from `outputs/results.md`; sandbox run, keyword only)
| Task 1 | Keyword | Embedding |
|---|---|---|
| Dev retrieval hits / 5 | 5 | **not run** |
| Test retrieval hits / 3 | 3 | **not run** |
| Test correct missing-evidence responses / 2 | 2 | **not run** |
| (extra) Test correct supported answers / 3 | 2 | **not run** |

| Extension 2 (test) | PDF RAG | Wiki RAG |
|---|---|---|
| Correct supported answers / 3 | 2 | 1 |
| Correct abstentions / 2 | 2 | 2 |

| Extension 1 (test) | Task 1 baseline | With Laya |
|---|---|---|
| Correct supported answers / 3 | 2 | **not run** |
| Correct abstentions / 2 | 2 | **not run** |
| Laya runtime errors | N/A | **not run** |

**Not run, and why:** embedding retrieval and Laya need `sentence-transformers` / `torch` and weights from Hugging Face, which the
build sandbox could not reach. Code, tests (with a fake encoder / fake agent, labelled as such) and the run commands exist; the
result files say `not_run` with the error. Keyword was therefore selected by default rather than by a dev comparison.

## Mistakes and limitations (actual)
1. **Test q6, PDF and wiki:** right chunk retrieved, wrong sentence copied. The question repeats "fully mechanized or fully automatic
   processes", so the first sentence of 3.1.2 outscored "Records of training and operation experience ...". Overlap ignores intent.
2. **Abstention rule is lightly validated:** test q7 passed with coverage 0.515 vs threshold 0.5; the two unanswerable questions
   scored 0.15 and 0.22; only two unanswerable examples, none topically close.
3. **Wiki q8:** the wiki table chunk lacks the words "gas cutting", so the cited intro bullet outscored it and the numbers were not
   returned. Wiki dev q2 also lost a three-way keyword tie.
4. **Laya (remaining limitation):** Laya can only change answer vs abstain; it cannot fix a wrong-sentence choice like q6, and its
   own model card says the base checkpoint is near chance zero-shot (the strong accuracy is for the fine-tuned typed-decisions one).

## Manual corrections
Only the font glyph U+F0A3 -> "≤" in Table 1 (page 6), logged in the chunking code (`GLYPHS`). Wiki: one bullet reworded to match the source.

## Versions, models, environment variables
Python 3.12.3 · pymupdf 2.0 · numpy 2.4.4 · pytest 9.1.1 (pinned in `requirements.txt`). Not installed here: sentence-transformers
(>=3.0), torch, laya (==0.4.15 pinned in `requirements-extras.txt`, unverified), anthropic. Models: embedding
`sentence-transformers/all-MiniLM-L6-v2`; Laya `convaiinnovations/laya`, subfolder `typed-decisions`; wiki author Claude Sonnet 5.5
(chat session). Env vars (only for `make_wiki.py`): `ANTHROPIC_API_KEY`, `LLM_MODEL`.

## Time spent
Not measured by the assistant: ______ hours (please fill in).

## AI and other help
Claude (Anthropic) wrote the code, tests, wiki, scoring keys and docs in chat and ran tests/evaluations in a sandbox. The wiki author had
seen the evaluation questions beforehand (so the wiki comparison is not blind) and the wiki mostly copies the manual's wording. The scoring
keys were also written by the assistant and should be reviewed. The simplified project is a rebuild of an earlier larger one; the rebuild
reproduced that build's dev and test numbers exactly. Laya's API was taken from its PyPI README (0.4.15), not from a run.

## Development process
Atomic commits are in `git log`; no remote existed in the sandbox, so nothing was pushed and no PR was opened - `PR_DESCRIPTION.md` is ready to paste.
