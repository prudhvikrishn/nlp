import pickle
import unittest

from src.feature_engineering import FeatureBuilder, nlp_features
from src.nlp_analysis import BankingNLPAnalyzer
from src.preprocessing import BankingTextPreprocessor


class TestNLPFeatures(unittest.TestCase):
    """Stage 5 (NLP analysis) must feed Stage 6 (feature engineering)."""

    @classmethod
    def setUpClass(cls):
        cls.a = BankingNLPAnalyzer()

    def test_semantic_and_entity_indicators(self):
        f = nlp_features(self.a.analyze("My debit card is not working."))
        self.assertIn("target=debit card", f)
        self.assertIn("action=working", f)
        self.assertIn("action_negated", f)
        self.assertIn("ent=CARD_TYPE", f)

    def test_hybrid_nlp_matrix_and_pickle(self):
        pre = BankingTextPreprocessor()
        train = pre.process_many(["I forgot my ATM PIN.", "I forgot my card PIN again.",
                                  "My debit card is not working.", "My credit card is not working today."])
        fb = FeatureBuilder().fit(train)
        X = fb.transform(train, "hybrid_nlp")
        self.assertEqual(X.shape[0], 4)
        self.assertGreater(X.shape[1], fb.transform(train, "hybrid").shape[1])
        clone = pickle.loads(pickle.dumps(fb))           # spaCy pipeline must not be pickled
        self.assertNotIn("_nlp", clone.__dict__)
        self.assertEqual(clone.transform(train[:1], "hybrid_nlp").shape[1], X.shape[1])


if __name__ == "__main__":
    unittest.main()
