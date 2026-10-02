"""STAGE 4 - VECTORIZATION (BoW, TF-IDF) and WORD EMBEDDING (Word2Vec CBOW / Skip-Gram)."""
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .utils import SEED


def make_bow():
    return CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)


def make_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=5000)


def train_word2vec(token_lists, sg: int, vector_size=100, window=5, min_count=2, epochs=40):
    """sg=0 -> CBOW, sg=1 -> Skip-Gram. Trained on TRAIN tokens only."""
    return Word2Vec(sentences=token_lists, vector_size=vector_size, window=window,
                    min_count=min_count, sg=sg, epochs=epochs, workers=1, seed=SEED)


def doc_vector(tokens, wv, weights=None):
    """Mean (weights=None) or weighted mean of word vectors. OOV words are skipped;
    an all-OOV query gets the zero vector."""
    vecs, ws = [], []
    for t in tokens:
        if t in wv.key_to_index:
            vecs.append(wv[t]); ws.append(1.0 if weights is None else weights.get(t, 1.0))
    if not vecs:
        return np.zeros(wv.vector_size, dtype=np.float32)
    return np.average(np.vstack(vecs), axis=0, weights=ws)


def doc_matrix(token_lists, wv, weights=None):
    return np.vstack([doc_vector(t, wv, weights) for t in token_lists])


def similarity_report(wv, pairs):
    rows = []
    for a, b in pairs:
        if a in wv.key_to_index and b in wv.key_to_index:
            rows.append({"word_a": a, "word_b": b, "cosine": round(float(wv.similarity(a, b)), 3), "note": ""})
        else:
            miss = [w for w in (a, b) if w not in wv.key_to_index]
            rows.append({"word_a": a, "word_b": b, "cosine": None, "note": f"OOV: {', '.join(miss)}"})
    return rows
