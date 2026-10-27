## Summary
Manual RAG over IACS Rec. 47 PDF pages 4-8: keyword vs embedding retrieval (Task 1), a cited-wiki variant (Extension 2) and a Laya
answer/abstain gate (Extension 1). Answers are copied from retrieved text with PDF-page citations or the fixed abstention sentence.

## Changes (one commit each)
1. Task 1: extraction/chunking, keyword + embedding retrievers, shared answer/abstain step, scoring, freeze guard, tests.
2. Extension 2: wiki (index + 3 topics), wiki loader/validator, optional API generator and prompt.
3. Extension 1: Laya gate wired to the documented typed-choice API; frozen question prompt; separate Laya log.
4. Docs, outputs, contract-aligned results, short report.

## Tests
`python -m pytest -q` -> 20 passed (sandbox). Embedding/Laya logic is tested with a fake encoder / fake agent only (not model runs).

## Evaluation (outputs/results.md)
Keyword: dev 5/5 hits; test 3/3 hits, 2/3 supported answers, 2/2 abstentions. Wiki: test 1/3 answers, 2/2 abstentions.
**Embedding and Laya: not run** (no Hugging Face / torch in the sandbox). Failures: q6 (wrong sentence), wiki q8 (table chunk lost context).

## Limitations
See REPORT.md: lightly validated abstention threshold; wiki authored in chat after seeing the questions; scoring keys AI-written; 10 questions only.

## Reproduce
`pip install -r requirements.txt [-r requirements-extras.txt] && ./run_all.sh`
