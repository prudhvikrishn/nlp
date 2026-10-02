"""Held-out behavioral checks. Requires `python -m src.pipeline` to have been run once."""
import json
import unittest

from src.predictor import IntentPredictor
from src.utils import BENCH_PATH
from src.utils import normalize_key
import pandas as pd
from src.utils import PROCESSED_DIR


class TestBehavioral(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pr = IntentPredictor()

    def _check(self, q, intent, min_conf=0.60, min_margin=0.30):
        r = self.pr.predict(q, with_analysis=False)
        probs = sorted(r["probabilities"].values())
        self.assertEqual(r["intent"], intent)
        self.assertGreaterEqual(r["confidence"], min_conf)
        self.assertGreaterEqual(r["confidence"] - probs[-2], min_margin)

    def test_required_card_issue(self):
        self._check("My debit card is not working.", "card_issue", 0.90)

    def test_required_forgot_pin(self):
        self._check("I forgot my ATM PIN.", "forgot_pin", 0.90)

    def test_fraud_and_balance_and_loan(self):
        self._check("Someone charged $450 in Paris on my visa card!", "fraud_report")
        self._check("How much money is left in my savings account?", "balance_inquiry")
        self._check("What are the interest rates for a personal loan?", "loan_inquiry")

    def test_no_benchmark_leakage(self):
        bench = {normalize_key(b["query"]) for b in json.loads(BENCH_PATH.read_text())}
        for n in ("train", "val", "test"):
            keys = set(pd.read_csv(PROCESSED_DIR / f"{n}.csv")["query"].map(normalize_key))
            self.assertFalse(bench & keys, f"benchmark leaked into {n}")

    def test_oos_flagged(self):
        self.assertTrue(self.pr.predict("what's the weather like today", with_analysis=False)["needs_review"])


if __name__ == "__main__":
    unittest.main()
