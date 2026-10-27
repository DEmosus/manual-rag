"""Run the whole protocol in the right order (cross-platform).

    python run_all.py            # dev for both retrievers -> (pause: look at dev) -> extensions dev -> all test runs -> report
    python run_all.py --yes      # no pause
    python run_all.py --force    # re-do test runs that already exist (deterministic, same code => same results)

Needs: pip install -r requirements.txt -r requirements-extras.txt   (the first run downloads ~0.9 GB of Laya weights)
"""
import argparse
import subprocess
import sys


def run(*args, stdout=None):
    print("\n$ python", *args, flush=True)
    subprocess.run([sys.executable, *args], check=True, stdout=stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="do not pause after the dev comparison")
    ap.add_argument("--force", action="store_true", help="overwrite existing test results")
    a = ap.parse_args()
    force = ["--force"] if a.force else []
    run("pdf_to_chunks.py")
    run("run_eval.py", "dev", "--system", "pdf", "--method", "all")
    run("run_eval.py", "report")
    if not a.yes and sys.stdin.isatty():
        print("\n>>> Look at the DEV rows above. The selected retriever is SELECTED_METHOD in config.py (rule: README).")
        print(">>> If you change it, do it NOW, before the extensions and before any test run.")
        input(">>> Press Enter to freeze and continue ")
    run("run_eval.py", "dev", "--system", "wiki")
    run("run_eval.py", "dev", "--system", "laya")
    run("run_eval.py", "test", "--system", "pdf", "--method", "all", *force)
    run("run_eval.py", "test", "--system", "wiki", *force)
    run("run_eval.py", "test", "--system", "laya", *force)
    run("run_eval.py", "report")
    run("-m", "pytest", "-q")
    with open("requirements-lock.txt", "w", encoding="utf-8") as fh:        # exact versions used for these results
        run("-m", "pip", "freeze", stdout=fh)
    print("\nDone. Results: outputs/results.md (also written into README.md and REPORT.md). Versions: requirements-lock.txt")


if __name__ == "__main__":
    main()
