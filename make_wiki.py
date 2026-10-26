"""EXTENSION 2 (optional) - ask an LLM to write the wiki. Needs:  pip install anthropic ; ANTHROPIC_API_KEY ; LLM_MODEL.
Writes to wiki_generated/ (never over wiki/). The committed wiki/ was written by Claude in a chat session instead -
see README 'Wiki provenance'. Prompt: prompts/wiki_prompt.md."""
import json
import os
import re
import sys
from pathlib import Path

pages = json.loads(Path("outputs/pages.json").read_text(encoding="utf-8"))
source = "\n\n".join(f"### PDF page {n}\n" + "\n".join(rows) for n, rows in pages.items())
prompt = Path("prompts/wiki_prompt.md").read_text(encoding="utf-8").replace("{{SOURCE_TEXT}}", source)
if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("LLM_MODEL"):
    sys.exit("set ANTHROPIC_API_KEY and LLM_MODEL first")
import anthropic  # noqa: E402

reply = anthropic.Anthropic().messages.create(model=os.environ["LLM_MODEL"], max_tokens=8000, temperature=0,
                                              messages=[{"role": "user", "content": prompt}])
text = "".join(b.text for b in reply.content if b.type == "text")
marks = list(re.finditer(r"^=== FILE: ([a-z0-9-]+\.md) ===\s*$", text, re.M))
out = Path("wiki_generated")
out.mkdir(exist_ok=True)
for i, m in enumerate(marks):
    end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
    (out / m.group(1)).write_text(text[m.end():end].strip() + "\n", encoding="utf-8")
(out / "generation_meta.json").write_text(json.dumps({"model": os.environ["LLM_MODEL"], "sdk": anthropic.__version__,
                                                       "temperature": 0, "files": [m.group(1) for m in marks]}, indent=1))
print("wrote", [m.group(1) for m in marks], "to wiki_generated/ - review, validate, then copy into wiki/")
