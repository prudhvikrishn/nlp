"""BSD Bank customer support web app: classifies a banking query and explains the result.

Entry point for Vercel's Python runtime (`app` is the WSGI application). The app is stateless:
queries are classified and shown back to the customer, and nothing is stored.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predictor import IntentPredictor
from src.utils import INTENT_DISPLAY, RECOMMENDED_ACTIONS

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
app.config.update(MAX_CONTENT_LENGTH=8 * 1024)

_predictor: IntentPredictor | None = None


def get_predictor() -> IntentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = IntentPredictor()
    return _predictor


EXAMPLE_QUERIES = [
    "I forgot the PIN for my debit card.",
    "There is a charge on my card I don't recognize.",
    "What is the balance in my savings account?",
    "How do I apply for a home loan?",
]


def _render_home(result=None, form=None, error=None, status=200):
    return render_template("customer.html", result=result, form=form or {}, error=error,
                           examples=EXAMPLE_QUERIES), status


_LIGHT_VERBS = {"is", "are", "was", "were", "be", "am", "'s", "do", "does", "did", "have", "has", "had"}


def _customer_result(query: str, prediction: dict) -> dict:
    """Shape a prediction for the customer page: the query itself, the category,
    the closest alternatives, and what the NLP analysis picked out of the text."""
    ranked = sorted(prediction["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
    analysis = prediction.get("analysis") or {}
    entities, seen = [], set()
    raw = analysis.get("entities", [])
    texts = [e["text"].lower() for e in raw]
    for ent in raw:
        text = ent["text"].lower()
        # skip duplicates and fragments of a longer detection ("450" and "$" inside "$450")
        if text in seen or any(text != other and text in other for other in texts):
            continue
        seen.add(text); entities.append(ent)
    action = analysis.get("action")
    if action and action.removeprefix("not ") in _LIGHT_VERBS:   # "There is a charge" -> no useful action
        action = None
    confidence = prediction["confidence"]
    return {
        "query": query,
        "intent": prediction["intent"],
        "display": prediction["display"],
        "confidence": confidence,
        "confidence_level": "High" if confidence >= 0.85 else "Medium" if confidence >= 0.6 else "Low",
        "needs_review": prediction["needs_review"],
        "next_step": RECOMMENDED_ACTIONS.get(prediction["intent"], ""),
        "alternatives": [{"display": INTENT_DISPLAY.get(k, k), "probability": p} for k, p in ranked[:3]],
        "action": action,
        "target": analysis.get("target"),
        "entities": entities[:6],
        "keywords": prediction.get("preprocessing", {}).get("lemmas", [])[:10],
    }


@app.get("/")
def customer_home():
    return _render_home()


@app.post("/submit")
def submit_query():
    customer_name = request.form.get("customer_name", "").strip()
    query = request.form.get("query", "").strip()
    form = {"customer_name": customer_name, "query": query}
    if not customer_name or not query:
        return _render_home(form=form, error="Please enter your name and describe your query.", status=400)
    if len(customer_name) > 100 or len(query) > 2000:
        return _render_home(form=form, error="Name or query is longer than the allowed limit.", status=400)

    try:
        prediction = get_predictor().predict(query, with_analysis=True)
    except Exception:
        app.logger.exception("Customer query could not be classified")
        return _render_home(form=form, error="We could not process your query right now. Please try again later.",
                            status=503)

    result = _customer_result(query, prediction)
    result["customer_name"] = customer_name
    return _render_home(result=result)


@app.get("/health")
def health():
    return {"status": "ok", "service": "BSD Bank query classifier"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8501")), debug=False)
