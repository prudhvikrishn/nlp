import os
import tempfile
import unittest

os.environ.setdefault("BSD_BANK_SQLITE_PATH", os.path.join(tempfile.mkdtemp(), "test.sqlite3"))

from app.app import app  # noqa: E402


class TestCustomerApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(WTF_CSRF_ENABLED=False, TESTING=True)
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


if __name__ == "__main__":
    unittest.main()
