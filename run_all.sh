#!/usr/bin/env bash
# The protocol in order. Needs: pip install -r requirements.txt  (+ requirements-extras.txt for embedding / Laya).
set -e
python pdf_to_chunks.py
python run_eval.py dev --system pdf --method all          # keyword AND embedding on the dev questions
python run_eval.py report | head -12
if [ -t 0 ]; then
  echo ">>> Compare the DEV rows above. If embedding is better on dev, set SELECTED_METHOD = \"embedding\" in config.py."
  read -r -p ">>> Press Enter to continue (settings are about to be frozen) " _
fi
python run_eval.py dev --system wiki                       # extensions use the selected method
python run_eval.py dev --system laya
python run_eval.py test --system pdf --method all          # test runs: refused if code/settings changed since dev
python run_eval.py test --system wiki
python run_eval.py test --system laya
python run_eval.py report
python -m pytest -q
