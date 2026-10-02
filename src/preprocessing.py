"""STAGE 3 - TEXT PREPROCESSING: one reusable engine for training AND inference."""
import re

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

from .utils import ensure_nltk

ensure_nltk()

CONTRACTIONS = [
    (r"\bcan'?t\b", "cannot"), (r"\bwon'?t\b", "will not"), (r"\bdon'?t\b", "do not"),
    (r"\bdoesn'?t\b", "does not"), (r"\bdidn'?t\b", "did not"), (r"\bisn'?t\b", "is not"),
    (r"\baren'?t\b", "are not"), (r"\bwasn'?t\b", "was not"), (r"\bcouldn'?t\b", "could not"),
    (r"\bshouldn'?t\b", "should not"), (r"\bhaven'?t\b", "have not"), (r"\bhasn'?t\b", "has not"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'m\b", " am"), (r"'ll\b", " will"),
    (r"'ve\b", " have"), (r"'d\b", " would"), (r"\bwhat's\b", "what is"), (r"\bthat's\b", "that is"),
    (r"\bit's\b", "it is"), (r"\bi'm\b", "i am"), (r"'s\b", ""),
]

# Slang / shorthand actually observed in the dataset + common banking SMS speak
SLANG = {
    "pls": "please", "plz": "please", "u": "you", "ur": "your", "r": "are", "rn": "right now",
    "wut": "what", "deets": "details", "cc": "credit card", "acc": "account", "acct": "account",
    "pwd": "password", "pw": "password", "pass": "password", "pswd": "password",
    "txn": "transaction", "trans": "transaction", "transac": "transaction", "transction": "transaction",
    "purchse": "purchase", "appl": "application", "app": "application", "lmk": "let me know",
    "ngl": "", "smh": "", "tho": "though", "bc": "because", "cuz": "because", "thx": "thanks",
    "asap": "urgent", "info": "information", "bal": "balance", "dept": "department",
    "atm": "atm", "wanna": "want to", "gonna": "going to", "gotta": "have to", "ya": "you",
    "b4": "before", "w": "with", "w/": "with", "n": "and", "dunno": "do not know",
}

KEEP_WORDS = {"not", "no", "nor", "never", "cannot", "out", "to", "from", "why", "when", "how",
              "what", "over", "after", "before", "off", "up", "down", "against", "more", "most"}
TO_VERBS = r"(transfer|check|see|know|pay|reset|block|apply|get|send|change|open|close|stop|report|find|view)"


class BankingTextPreprocessor:
    """clean -> tokenize -> banking-aware stopword removal -> POS-guided lemmatization."""

    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords
        base = set(stopwords.words("english"))
        self.stop = base - KEEP_WORDS - {w for w in base if w.endswith("n't")}
        self._lem = WordNetLemmatizer()
        self._stem = PorterStemmer()

    # -- step 1: cleaning ------------------------------------------------------
    def clean(self, text: str) -> str:
        t = str(text).lower().replace("\u2019", "'").replace("\u2018", "'")
        t = re.sub(r"\b4(?= (a|the|my|me|u|you|your|ur)\b)", "for", t)
        t = re.sub(rf"\b2(?= {TO_VERBS}\b)", "to", t)
        t = t.replace("$", " dollar ").replace("\u20b9", " rupee ")
        for pat, rep in CONTRACTIONS:
            t = re.sub(pat, rep, t)
        t = re.sub(r"(.)\1{2,}", r"\1\1", t)                 # pleeease -> pleease
        t = re.sub(r"[^a-z0-9%/ ]", " ", t)                   # keep % and w/
        words = []
        for w in t.split():
            words.append(SLANG.get(w, w))
        t = " ".join(words)
        return re.sub(r"\s+", " ", t).strip()

    # -- step 2/3/4 ------------------------------------------------------------
    def tokenize(self, cleaned: str):
        return cleaned.split()

    def drop_stopwords(self, tokens):
        kept = [w for w in tokens if w not in self.stop]
        return kept if kept else tokens

    @staticmethod
    def _wn(tag: str):
        return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)

    def lemmatize(self, tokens):
        if not tokens:
            return [], []
        tagged = nltk.pos_tag(tokens)
        return [self._lem.lemmatize(w, self._wn(t)) for w, t in tagged], [t for _, t in tagged]

    def stem(self, tokens):
        return [self._stem.stem(w) for w in tokens]

    # -- public API ------------------------------------------------------------
    def process(self, text: str) -> dict:
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        if self.remove_stopwords:
            tokens = self.drop_stopwords(tokens)
        lemmas, pos = self.lemmatize(tokens)
        return {"raw": text, "cleaned": cleaned, "tokens": tokens,
                "lemmas": lemmas, "stems": self.stem(tokens), "pos": pos}

    def process_many(self, texts):
        return [self.process(t) for t in texts]

    def transform(self, text: str) -> str:
        return " ".join(self.process(text)["lemmas"])
