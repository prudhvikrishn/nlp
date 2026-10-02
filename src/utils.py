"""Paths, seeds, label maps and shared helpers (cross-platform via pathlib)."""
import random
import re
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "bank_customer_service_intent_classification_dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
BENCH_PATH = DATA_DIR / "benchmarks" / "behavioral_test_suite.json"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
SEED = 42

INTENT_DISPLAY = {
    "card_issue": "Card Issue",
    "forgot_pin": "Forgot PIN",
    "fraud_report": "Fraud Report",
    "loan_inquiry": "Loan Inquiry",
    "transaction_query": "Transaction Query",
    "balance_inquiry": "Balance Inquiry",
    "credit_card_application": "Credit Card Application",
    "password_reset": "Password Reset",
}

RECOMMENDED_ACTIONS = {
    "card_issue": "Run card diagnostics (chip/PIN test), then offer instant replacement card.",
    "forgot_pin": "Start the secure 2FA PIN reset workflow (OTP + identity check).",
    "fraud_report": "Freeze the card immediately and connect to the 24/7 fraud hotline.",
    "loan_inquiry": "Show loan calculator, rate table and the document upload checklist.",
    "transaction_query": "Show recent statement breakdown with a 'dispute this transaction' option.",
    "balance_inquiry": "Display the available checking/savings balance summary.",
    "credit_card_application": "Open card comparison and the pre-approval application form.",
    "password_reset": "Send a secure password-reset link after identity verification.",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, BENCH_PATH.parent, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def normalize_key(text: str) -> str:
    """Aggressive normalization used ONLY for duplicate / leakage detection."""
    t = str(text).lower().replace("\u2019", "'")
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def ensure_nltk() -> None:
    """Download the small NLTK resources on first run."""
    import nltk
    bundled_data = BASE_DIR / "nltk_data"
    if bundled_data.is_dir() and str(bundled_data) not in nltk.data.path:
        nltk.data.path.insert(0, str(bundled_data))
    needs = {"wordnet": "corpora/wordnet", "stopwords": "corpora/stopwords",
             "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
             "punkt_tab": "tokenizers/punkt_tab"}
    for pkg, path in needs.items():
        try:
            nltk.data.find(path)
        except LookupError:
            # NLTK corpus readers can consume these bundled zip archives directly.
            if (bundled_data / f"{path}.zip").is_file():
                continue
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
