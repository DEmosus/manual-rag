"""STEP 2 - find the 3 most relevant chunks for a question. Two interchangeable methods:

    keyword    count the distinct words the question and the chunk share
    embedding  turn both into vectors with a pretrained model, rank by cosine similarity

Both return the same thing: a list of {"chunk", "score", "rank", "method"} (best first)."""
import re
import unicodedata

import numpy as np

STOPWORDS = set("""a an and are as at be by for from has have how in is it its of on or that the their this to was were
what when where which who whom why will with does do did can could should would may might must shall there these those
than then them they he she we you i not no difference between""".split())      # last two: question-framing words
SUFFIXES = ("ication", "ications", "ied", "ations", "ation", "ings", "ing", "edly", "ed", "es", "s", "e")


def normalize(text):
    """lower-case, unify unicode (micro sign, curly quotes) and whitespace - used for ALL text matching."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def stem(word):
    """tiny stemmer: approves/approved/approve -> approv; qualification/qualified -> qualif"""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def tokens(text):
    words = re.findall(r"[a-z0-9]+", normalize(text))
    return [stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


def search_text(chunk):
    """what retrievers look at: parent headings + text (answers copy from text only)"""
    return f"{chunk['heading']} {chunk['text']}".strip()


def keyword_search(chunks, question, k):
    q = set(tokens(question))
    if not q:
        return []
    scored = [(len(q & set(tokens(search_text(c)))), i) for i, c in enumerate(chunks)]
    scored.sort(key=lambda t: (-t[0], t[1]))                     # best score first; ties -> document order
    return [{"chunk": chunks[i], "score": float(s), "rank": r, "method": "keyword"}
            for r, (s, i) in enumerate(scored[:k], start=1)]


def load_encoder(model_name):
    """Load the pretrained embedding model. Fails loudly (never silently uses another model)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("sentence-transformers is not installed (pip install -r requirements-extras.txt)") from e
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(f"could not load embedding model {model_name!r}: {e}") from e
    return lambda texts: model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)


def make_embedding_search(chunks, encode):
    """`encode` maps a list of strings to a 2-D array; injected so tests can use a fake one."""
    def unit(m):
        m = np.asarray(m, dtype=float)
        if m.ndim != 2 or not np.all(np.isfinite(m)):
            raise ValueError("encoder must return a finite 2-D array")
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(n == 0, 1, n)

    vectors = unit(encode([search_text(c) for c in chunks]))     # chunk vectors computed once, kept in memory

    def search(question, k):
        if not question.strip():
            return []
        sims = vectors @ unit(encode([question]))[0]             # unit vectors: dot product == cosine similarity
        order = sorted(range(len(chunks)), key=lambda i: (-sims[i], i))
        return [{"chunk": chunks[i], "score": float(sims[i]), "rank": r, "method": "embedding"}
                for r, i in enumerate(order[:k], start=1)]
    return search


def make_search(method, chunks, encode=None):
    """Return search(question, k) for 'keyword' or 'embedding'."""
    if method == "keyword":
        return lambda question, k: keyword_search(chunks, question, k)
    if method == "embedding":
        if encode is None:
            import config
            encode = load_encoder(config.EMBEDDING_MODEL)
        return make_embedding_search(chunks, encode)
    raise ValueError(f"unknown method {method!r}")
