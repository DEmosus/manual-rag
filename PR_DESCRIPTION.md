## Summary
Manual RAG over IACS Rec. 47 PDF pages 4-8. Task 1 compares a keyword baseline with embedding retrieval (`sentence-transformers/all-MiniLM-L6-v2`) over identical chunks with one
shared answer/abstain step. Extension 1 adds a Laya answer/abstain gate; Extension 2 searches a cited wiki instead of the PDF. Answers are copied from retrieved text with PDF-page
citations, or the fixed sentence "The supplied pages do not specify an answer."

## Changes
1. Task 1: page-limited extraction, clause/table-aware chunks, keyword and embedding retrievers, shared answer/abstain, scoring, tests.
2. Extension 2: wiki (index + 3 topic pages), wiki loader and validator, saved prompt and provenance note.
3. Extension 1: Laya gate using the documented typed-choice API; frozen question prompt; separate Laya input/output/error log.
4. Output contract alignment, freeze guard (dev fingerprint must match before test), model-revision recording, auto-filled results, `run_all.py`.
5. Final docs: README, EXPLAINER, REPORT.

## Tests
`python -m pytest -q` -> 24 tests (chunking and table headers, both retrievers with a fake encoder, answer/abstain, scoring, wiki loader/validator, Laya gate with a fake agent,
freeze guard, output contract). The fake encoder/agent tests check code paths only; real embedding and Laya runs are in `outputs/`.

## Evaluation
See the results tables in README.md / REPORT.md / `outputs/results.md` (generated from `outputs/*.json` by `python run_eval.py report`). Failures are listed and explained in
README section 11: q6 (right chunk, wrong sentence), wiki q8 (table chunk lost its context).

## Limitations
Lightly validated abstention threshold; wiki authored in chat after the author saw the questions (not blind); scoring keys written by the AI assistant; 10 questions only; Laya
confidence is uncalibrated for this checkpoint (unused).

## Reproduce
`pip install -r requirements.txt -r requirements-extras.txt && python run_all.py` (first run downloads ~0.9 GB for Laya and ~90 MB for the embedding model).
