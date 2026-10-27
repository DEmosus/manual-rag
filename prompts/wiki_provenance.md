# How the wiki in `wiki/` was produced

* **Model:** Claude Sonnet 5.5 (Anthropic), used in a chat session (no API call, so no model snapshot id or sampling parameters exist).
* **Input:** the text of PDF pages 4-8 as extracted by `pdf_to_chunks.py` (`outputs/pages.json`), under the rules in `prompts/wiki_prompt.md`
  (the saved prompt; `{{SOURCE_TEXT}}` stands for that page text).
* **Output:** `wiki/index.md` and three topic pages, written once and not edited after any evaluation run.
* **Checks:** `python wiki_tools.py` (links, citations, page range, numbers traceable to the cited page). One bullet was reworded
  during writing because it contained wording that is not in the source ("the two tables below").
* **Limitations:** the author had already seen the evaluation questions (the task says not to use them to generate the wiki), so the
  wiki-vs-PDF comparison is not blind; the wiki stays close to the manual's wording; a human has not independently checked every claim.
