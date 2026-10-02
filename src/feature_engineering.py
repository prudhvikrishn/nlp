"""STAGE 6 - FEATURE ENGINEERING: five feature sets, everything fitted on TRAIN only."""
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from . import embeddings as emb

FEATURE_KINDS = ["bow", "tfidf", "w2v_cbow", "w2v_skipgram", "hybrid"]
LING_NAMES = ["n_tokens", "n_chars", "n_exclaim", "n_question", "upper_ratio", "has_money",
              "has_card_kw", "has_negation", "has_urgency", "verb_ratio", "noun_ratio"]
_CARD = re.compile(r"\b(card|atm|pin|chip|debit|credit|swipe)\b")
_URG = re.compile(r"\b(urgent|asap|immediately|now|hurry|fast|emergency)\b")
_NEG = re.compile(r"\b(not|no|never|cannot|without)\b")


def linguistic_features(proc: dict) -> list:
    raw, cl, pos = proc["raw"], proc["cleaned"], proc["pos"]
    n = max(len(pos), 1)
    letters = [c for c in raw if c.isalpha()]
    return [
        len(proc["tokens"]), len(raw), raw.count("!"), raw.count("?"),
        sum(c.isupper() for c in letters) / max(len(letters), 1),
        float(bool(re.search(r"[$\u20b9]|\b\d+\b|dollar|rupee", raw.lower()))),
        float(bool(_CARD.search(cl))), float(bool(_NEG.search(cl))), float(bool(_URG.search(cl))),
        sum(p.startswith("VB") for p in pos) / n, sum(p.startswith("NN") for p in pos) / n,
    ]


class FeatureBuilder:
    """fit(train_proc) learns vectorizers / Word2Vec / scaler; transform(proc, kind) applies them."""

    def fit(self, train_proc):
        docs = [" ".join(p["lemmas"]) for p in train_proc]
        toks = [p["lemmas"] for p in train_proc]
        self.bow = emb.make_bow().fit(docs)
        self.tfidf = emb.make_tfidf().fit(docs)
        self.train_tfidf = self.tfidf.transform(docs)      # reference set for OOS detection
        uni = TfidfVectorizer(token_pattern=r"\S+").fit(docs)          # idf weights for weighted pooling
        self.idf = dict(zip(uni.get_feature_names_out(), uni.idf_))
        self.w2v_cbow_model = emb.train_word2vec(toks, sg=0)
        self.w2v_sg_model = emb.train_word2vec(toks, sg=1)
        self.wv_cbow, self.wv_sg = self.w2v_cbow_model.wv, self.w2v_sg_model.wv
        self.scaler = MinMaxScaler().fit(np.array([linguistic_features(p) for p in train_proc]))
        return self

    def nearest_similarity(self, proc):
        """Cosine similarity of each query to its closest TRAIN query (TF-IDF rows are L2-normalised).
        Low values mean the query shares almost no vocabulary with anything seen in training."""
        q = self.tfidf.transform([" ".join(p["lemmas"]) for p in proc])
        return np.asarray((q @ self.train_tfidf.T).max(axis=1).todense()).ravel()

    def drop_training_state(self):
        """Keep only KeyedVectors in the pickled artifact (smaller, inference-only)."""
        self.w2v_cbow_model = self.w2v_sg_model = None
        return self

    def transform(self, proc, kind: str):
        docs = [" ".join(p["lemmas"]) for p in proc]
        toks = [p["lemmas"] for p in proc]
        if kind == "bow":
            return self.bow.transform(docs)
        if kind == "tfidf":
            return self.tfidf.transform(docs)
        if kind == "w2v_cbow":
            return emb.doc_matrix(toks, self.wv_cbow)
        if kind == "w2v_skipgram":
            return emb.doc_matrix(toks, self.wv_sg)
        if kind == "hybrid":
            w = emb.doc_matrix(toks, self.wv_cbow, self.idf)
            ling = np.clip(self.scaler.transform(np.array([linguistic_features(p) for p in proc])), 0, 1)
            return sparse.hstack([self.tfidf.transform(docs), sparse.csr_matrix(w), sparse.csr_matrix(ling)]).tocsr()
        raise ValueError(kind)
