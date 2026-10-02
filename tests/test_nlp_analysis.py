import unittest
from src.nlp_analysis import BankingNLPAnalyzer


class TestNLPAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = BankingNLPAnalyzer()

    def test_action_target_card(self):
        r = self.a.analyze("My debit card is not working.")
        self.assertEqual(r["action"], "not working"); self.assertEqual(r["target"], "debit card")

    def test_action_target_pin(self):
        r = self.a.analyze("I forgot my ATM PIN.")
        self.assertEqual(r["action"], "forgot"); self.assertEqual(r["target"], "atm pin")

    def test_entities(self):
        labels = {e["label"] for e in self.a.analyze("Someone charged $450 in Paris on my visa card!")["entities"]}
        self.assertTrue({"AMOUNT", "CARD_TYPE"} <= labels)

    def test_pos_present(self):
        self.assertTrue(any(p["pos"] == "VERB" for p in self.a.analyze("I forgot my ATM PIN.")["pos"]))


if __name__ == "__main__":
    unittest.main()
