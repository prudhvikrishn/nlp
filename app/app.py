"""BSD Bank web application entrypoint for Vercel's Python WSGI runtime."""
from __future__ import annotations

import hmac
import os
import secrets
import sys
from datetime import timedelta
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, session, url_for
from flask_wtf.csrf import CSRFError, CSRFProtect

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
CONFIGURED_SECRET_KEY = os.environ.get("SECRET_KEY", "")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from submissions import get_submissions, initialize_store, save_submission
from src.predictor import IntentPredictor
from src.utils import INTENT_DISPLAY, RECOMMENDED_ACTIONS

app = Flask(__name__, template_folder=str(APP_DIR / "templates"))
app.config.update(
    SECRET_KEY=CONFIGURED_SECRET_KEY or secrets.token_urlsafe(48),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("VERCEL")),
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
    MAX_CONTENT_LENGTH=8 * 1024,
)
CSRFProtect(app)

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


def _render_home(result=None, form=None, status=200):
    return render_template("customer.html", result=result, form=form or {},
                           examples=EXAMPLE_QUERIES), status


_LIGHT_VERBS = {"is", "are", "was", "were", "be", "am", "'s", "do", "does", "did", "have", "has", "had"}


def _customer_result(query: str, prediction: dict) -> dict:
    """Shape a prediction for the customer page: the query itself, the category,
    the closest alternatives, and what the NLP analysis picked out of the text."""
    ranked = sorted(prediction["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
    analysis = prediction.get("analysis") or {}
    entities, seen = [], set()
    for ent in analysis.get("entities", []):
        key = (ent["text"].lower(), ent["label"])
        if key not in seen:
            seen.add(key); entities.append(ent)
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
        flash("Please enter your name and describe your query.", "error")
        return _render_home(form=form, status=400)
    if len(customer_name) > 100 or len(query) > 2000:
        flash("Name or query is longer than the allowed limit.", "error")
        return _render_home(form=form, status=400)

    try:
        prediction = get_predictor().predict(query, with_analysis=True)
    except Exception:
        app.logger.exception("Customer query could not be classified")
        flash("We could not process your query right now. Please try again later.", "error")
        return _render_home(form=form, status=503)
    try:
        initialize_store()
        save_submission(customer_name, query, prediction["intent"], prediction["confidence"])
        recorded = True
    except Exception:
        app.logger.exception("Customer query could not be recorded")
        recorded = False

    result = _customer_result(query, prediction)
    result["customer_name"] = customer_name
    result["recorded"] = recorded
    return _render_home(result=result)


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("admin_authenticated"):
        return redirect(url_for("admin_inbox"))
    configured_password = os.environ.get("ADMIN_PASSWORD", "")
    admin_configured = bool(configured_password and CONFIGURED_SECRET_KEY)
    if request.method == "POST":
        provided = request.form.get("password", "")
        if admin_configured and hmac.compare_digest(provided, configured_password):
            session.clear()
            session["admin_authenticated"] = True
            session.permanent = True
            return redirect(url_for("admin_inbox"))
        flash("Admin access is not configured or the password is incorrect.", "error")
    return render_template("admin_login.html", admin_configured=admin_configured)


@app.get("/admin")
def admin_inbox():
    if not session.get("admin_authenticated"):
        return redirect(url_for("admin_login"))
    try:
        initialize_store()
        records = get_submissions()
    except Exception:
        app.logger.exception("Admin inbox could not read the submission store")
        return render_template("admin.html", records=[], storage_error=True)
    return render_template("admin.html", records=records, storage_error=False)


@app.post("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


@app.get("/health")
def health():
    return {"status": "ok", "service": "BSD Bank query classifier"}


@app.errorhandler(CSRFError)
def csrf_error(_error):
    return "The form expired. Reload the page and try again.", 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8501")), debug=False)
