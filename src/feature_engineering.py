"""STAGE 6 - FEATURE ENGINEERING: six feature sets, everything fitted on TRAIN only.

`hybrid_nlp` is where Stage 5 (NLP ANALYSIS) feeds this stage: entity labels, the action->target
semantic labels and dependency-parse relations become sparse indicator features next to the hybrid set.
"""
import re
import warnings
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize

from . import embeddings as emb
from .nlp_analysis import BankingNLPAnalyzer

FEATURE_KINDS = ["bow", "tfidf", "w2v_cbow", "w2v_skipgram", "hybrid", "hybrid_nlp"]
NLP_MIN_COUNT = 2         # drop NLP indicators seen fewer than twice in TRAIN
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


def nlp_features(analysis: dict) -> dict:
    """Stage-5 output -> indicator dict: NER labels, semantic action/target, dependency relations."""
    f = {f"ent={e['label']}": 1.0 for e in analysis["entities"]}
    action, target = analysis.get("action"), analysis.get("target")
    if target:
        f[f"target={target}"] = 1.0
    if action:
        neg = action.startswith("not ")
        f[f"action={action[4:] if neg else action}"] = 1.0
        if neg:
            f["action_negated"] = 1.0
        if target:
            f[f"frame={action}->{target}"] = 1.0
    for d in analysis["deps"]:
        tok, head = d["token"].lower(), d["head"].lower()
        f[f"dep={d['dep']}"] = 1.0
        if d["dep"] == "ROOT":
            f[f"root={tok}"] = 1.0
        elif d["dep"] in ("dobj", "nsubj", "nsubjpass", "pobj", "acomp"):
            f[f"{d['dep']}={head}_{tok}"] = 1.0
    return f


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
        feats = self._nlp_dicts(train_proc)
        self.nlp_backend = self._analyzer().backend
        counts = Counter(k for f in feats for k in f)
        self.nlp_vocab = {k for k, c in counts.items() if c >= NLP_MIN_COUNT}
        self.nlp_vec = DictVectorizer().fit([{k: v for k, v in f.items() if k in self.nlp_vocab} for f in feats])
        return self

    # ---- Stage 5 -> Stage 6 bridge ---------------------------------------------------------
    def _analyzer(self):
        if getattr(self, "_nlp", None) is None:     # spaCy is loaded by name, never pickled
            self._nlp = BankingNLPAnalyzer()
            fitted = getattr(self, "nlp_backend", None)
            if fitted and self._nlp.backend != fitted:
                warnings.warn(f"NLP features were fitted with {fitted} but {self._nlp.backend} is active; "
                              "hybrid_nlp predictions may degrade. Install spaCy + en_core_web_sm.")
        return self._nlp

    def _nlp_dicts(self, proc):
        an = self._analyzer()
        return [nlp_features(an.analyze(p["raw"])) for p in proc]

    def nlp_matrix(self, proc):
        """Rows are L2-normalised like the TF-IDF block, so ~100 binary indicators cannot swamp it."""
        return normalize(self.nlp_vec.transform([{k: v for k, v in f.items() if k in self.nlp_vocab} for f in self._nlp_dicts(proc)]))

    def __getstate__(self):
        state = self.__dict__.copy(); state.pop("_nlp", None); return state

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
        if kind == "hybrid_nlp":
            return sparse.hstack([self.transform(proc, "hybrid"), self.nlp_matrix(proc)]).tocsr()
        raise ValueError(kind)
