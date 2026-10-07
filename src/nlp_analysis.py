"""STAGE 5 - NLP ANALYSIS: POS tagging, dependency parsing, NER and Action->Target labeling.

spaCy (en_core_web_sm) is loaded by name at runtime and is NEVER pickled.
If the model is missing we try a one-time download, then fall back to NLTK + regex.
"""
import re

import nltk

from .utils import ensure_nltk

ensure_nltk()

BANK_TARGETS = [
    "atm pin number", "atm pin", "debit card pin", "credit card pin", "card pin", "debit card", "credit card",
    "atm card", "bank card", "visa card", "master card", "savings account", "checking account", "current account", "online banking",
    "net banking", "netbanking", "mobile banking", "interest rate", "personal loan", "home loan", "car loan",
    "line of credit", "credit limit", "password", "passcode", "pin", "otp", "card", "loan", "balance",
    "transaction", "charge", "payment", "transfer", "statement", "account", "fraud", "cheque", "check",
    "deposit", "mortgage", "emi",
]
_TARGET_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, BANK_TARGETS), key=len, reverse=True)) + r")\b")

_AMOUNT_RE = re.compile(r"(?:\$|\u20b9|rs\.?\s?)\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars?|rupees?|usd|inr|bucks)\b", re.I)
_MASKED_CARD_RE = re.compile(r"(?:\*{2,}|x{2,})[\s-]?\d{4}", re.I)
_CARD_NUM_RE = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
_ACCOUNT_RE = re.compile(r"\b\d{9,18}\b")
_CARD_TYPE_RE = re.compile(r"\b(debit card|credit card|atm card|bank card|visa|mastercard|rupay|amex)\b", re.I)
_CURRENCY_RE = re.compile(r"[$\u20b9\u00a3\u20ac]|\b(?:usd|inr|eur|gbp|dollars?|rupees?)\b", re.I)
_SPACY_KEEP = {"MONEY", "DATE", "TIME", "GPE", "LOC", "ORG", "PERSON", "CARDINAL"}


class BankingNLPAnalyzer:
    def __init__(self, allow_download: bool = False):
        self.nlp, self.backend = None, "nltk-fallback"
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                if allow_download:
                    from spacy.cli import download
                    download("en_core_web_sm"); self.nlp = spacy.load("en_core_web_sm")
            if self.nlp is not None:
                self.backend = "spacy:en_core_web_sm"
        except Exception:
            self.nlp = None

    # ---- banking regex entities --------------------------------------------------
    @staticmethod
    def banking_entities(text: str):
        ents = []
        for lab, rx in (("AMOUNT", _AMOUNT_RE), ("MASKED_CARD", _MASKED_CARD_RE), ("CARD_NUMBER", _CARD_NUM_RE),
                        ("ACCOUNT_NO", _ACCOUNT_RE), ("CARD_TYPE", _CARD_TYPE_RE), ("CURRENCY", _CURRENCY_RE)):
            for m in rx.finditer(text):
                ents.append({"text": m.group(0).strip(), "label": lab, "source": "regex"})
        return ents

    @staticmethod
    def find_target(text: str):
        m = _TARGET_RE.search(text.lower())
        return m.group(1) if m else None

    # ---- main entry ---------------------------------------------------------------
    def analyze(self, text: str) -> dict:
        return self._spacy(text) if self.nlp is not None else self._fallback(text)

    def _spacy(self, text):
        doc = self.nlp(text)
        pos = [{"token": t.text, "pos": t.pos_, "tag": t.tag_} for t in doc if not t.is_space]
        deps = [{"token": t.text, "dep": t.dep_, "head": t.head.text} for t in doc if not t.is_space]
        ents = [{"text": e.text, "label": e.label_, "source": "spacy"} for e in doc.ents if e.label_ in _SPACY_KEEP]
        ents = [e for e in ents if not (e["source"] == "spacy" and e["text"].lower() in {"atm", "pin", "otp", "atm pin"})]
        ents += self.banking_entities(text)

        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        action = None
        if root is not None:
            verb = root
            if root.pos_ in ("AUX", "PRON", "NOUN", "PROPN"):          # "my card is damaged" -> damaged
                comp = [c for c in root.children if c.dep_ in ("acomp", "xcomp", "attr", "ccomp") and c.pos_ in ("ADJ", "VERB")]
                if comp:
                    verb = comp[0]
                else:
                    v = next((t for t in doc if t.pos_ == "VERB"), None)
                    verb = v if v is not None else root
            neg = any(c.dep_ == "neg" for c in verb.children) or any(c.dep_ == "neg" for c in root.children)
            action = ("not " if neg else "") + verb.text.lower()
        if action and not re.fullmatch(r"(not )?[a-z]{2,}", action):
            action = None
        target = self.find_target(text)
        if target is None and root is not None:
            for t in doc:
                if t.dep_ in ("dobj", "nsubj", "pobj", "attr") and t.pos_ in ("NOUN", "PROPN"):
                    toks = [w.text for w in t.subtree if w.dep_ in ("compound", "amod") or w is t]
                    target = " ".join(toks).lower(); break
        return {"backend": self.backend, "pos": pos, "deps": deps, "entities": ents,
                "action": action, "target": target}

    def _fallback(self, text):
        toks = nltk.word_tokenize(text.replace("\u2019", "'"))   # "didn’t" -> did + n't, so negation is seen
        tagged = nltk.pos_tag(toks)
        umap = lambda t: "VERB" if t.startswith("VB") else "NOUN" if t.startswith("NN") else "ADJ" if t.startswith("JJ") \
            else "ADV" if t.startswith("RB") else "PRON" if t.startswith("PRP") else "OTHER"
        pos = [{"token": w, "pos": umap(t), "tag": t} for w, t in tagged]
        verbs = [w for w, t in tagged if t.startswith("VB")]
        neg = any(w.lower() in ("not", "n't", "no", "never", "cannot") for w in toks)
        action = (("not " if neg else "") + verbs[-1].lower()) if verbs else None
        return {"backend": self.backend, "pos": pos, "deps": [], "entities": self.banking_entities(text),
                "action": action, "target": self.find_target(text)}
