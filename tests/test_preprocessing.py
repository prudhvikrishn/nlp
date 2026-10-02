import unittest
from src.preprocessing import BankingTextPreprocessor


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.p = BankingTextPreprocessor()

    def test_negation_kept(self):
        self.assertIn("not", self.p.process("My debit card is not working.")["tokens"])

    def test_contraction_and_slang(self):
        self.assertEqual(self.p.clean("I can't see my acct bal"), "i cannot see my account balance")
        self.assertIn("for", self.p.clean("how do i apply 4 a credit card").split())

    def test_currency_symbol_kept_as_word(self):
        self.assertIn("dollar", self.p.clean("charged $450"))

    def test_never_empty(self):
        self.assertTrue(self.p.process("how to")["tokens"])


if __name__ == "__main__":
    unittest.main()
