"""Inference wrapper shared by predict.py and the Streamlit app."""
import joblib

from .nlp_analysis import BankingNLPAnalyzer
from .utils import INTENT_DISPLAY, MODELS_DIR, RECOMMENDED_ACTIONS


class IntentPredictor:
    def __init__(self, path=None):
        art = joblib.load(path or MODELS_DIR / "best_model.pkl")
        self.pre, self.fb, self.model = art["preprocessor"], art["feature_builder"], art["model"]
        self.le, self.kind, self.threshold = art["label_encoder"], art["feature_kind"], art["reject_threshold"]
        self.sim_threshold = art.get("similarity_threshold", 0.0)
        self.model_name = art["model_name"]
        self.analyzer = BankingNLPAnalyzer()

    def predict(self, query: str, with_analysis: bool = True) -> dict:
        proc = self.pre.process(query)
        proba = self.model.predict_proba(self.fb.transform([proc], self.kind))[0]
        classes = list(self.le.classes_); i = int(proba.argmax()); conf = float(proba[i])
        sim = float(self.fb.nearest_similarity([proc])[0])
        second = float(sorted(proba)[-2])
        needs_review = conf < self.threshold or sim < self.sim_threshold
        label = classes[i]
        res = {"query": query, "intent": label, "display": INTENT_DISPLAY.get(label, label), "confidence": conf,
               "needs_review": needs_review, "margin": conf - second,
               "ambiguous": (not needs_review) and (conf - second) < 0.10, "runner_up": classes[int(proba.argsort()[-2])], "threshold": self.threshold, "similarity": sim, "sim_threshold": self.sim_threshold,
               "probabilities": {c: float(p) for c, p in zip(classes, proba)},
               "action": (f"Low confidence / unfamiliar wording - route to a human agent (best guess: {INTENT_DISPLAY.get(label, label)})."
                  if needs_review else RECOMMENDED_ACTIONS.get(label, "")),
               "preprocessing": {k: proc[k] for k in ("cleaned", "tokens", "lemmas")}}
        if with_analysis:
            res["analysis"] = self.analyzer.analyze(query)
        return res
