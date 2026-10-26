"""Every setting in one place. Edit these ONLY while tuning on the dev questions.
Once you run `python run_eval.py test ...`, treat this file as frozen."""

PDF_PATH = "materials/IACS_47_Steel_Shipbuilding_Quality_2025.pdf"
FIRST_PAGE, LAST_PAGE = 4, 8          # 1-based PDF pages, inclusive

TOP_K = 3                              # chunks returned by every retriever
ABSTAIN_MESSAGE = "The supplied pages do not specify an answer."
COVERAGE_THRESHOLD = 0.5               # abstain if the best chunk covers < 50% of the question's weight
MAX_SENTENCES = 5                      # an answer copies at most this many sentences
SENTENCE_CUTOFF = 0.6                  # ...that score at least 60% of the best sentence

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SELECTED_METHOD = "keyword"            # chosen from DEV results only (see README)

LAYA_REPO = "convaiinnovations/laya"   # Extension 1
LAYA_SUBFOLDER = "typed-decisions"     # 1024-token context (the base English one has 512)
