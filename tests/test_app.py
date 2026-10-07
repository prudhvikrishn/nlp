import unittest

from app.app import app


class TestCustomerApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_home_renders_form(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'name="query"', r.data)

    def test_submit_echoes_query_and_category(self):
        query = "I forgot my ATM PIN & can't withdraw"
        r = self.client.post("/submit", data={"customer_name": "Asha", "query": query})
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("I forgot my ATM PIN &amp; can&#39;t withdraw", html)   # shown back, HTML-escaped
        self.assertIn("Forgot PIN", html)
        self.assertIn("Closest matches", html)

    def test_missing_fields_keep_input(self):
        r = self.client.post("/submit", data={"customer_name": "", "query": "check my balance"})
        self.assertEqual(r.status_code, 400)
        self.assertIn(b"check my balance", r.data)

    def test_result_shows_routing_and_analysis(self):
        r = self.client.post("/submit", data={"customer_name": "Asha", "query": "what is my savings balance"})
        html = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("Balance Inquiry", html)
        self.assertIn("What we picked out of your message", html)

    def test_health(self):
        self.assertEqual(self.client.get("/health").json["status"], "ok")

if __name__ == "__main__":
    unittest.main()
