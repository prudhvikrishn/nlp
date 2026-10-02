"""Local BSD Bank customer page. Run: streamlit run app/customer_streamlit.py --server.address 127.0.0.1 --server.port 8501"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from submissions import initialize_store, save_submission
from src.predictor import IntentPredictor

st.set_page_config(page_title="BSD Bank", layout="centered")

st.markdown("""
<style>
.stApp { background: #f4f7fb; }
.block-container { max-width: 780px; padding-top: 2.5rem; }
.bsd-header { background: #0b2d4d; color: white; padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1.2rem; }
.bsd-header h1 { margin: 0; color: white; }
.bsd-header p { margin: .35rem 0 0; color: #d7e6f3; }
</style>
""", unsafe_allow_html=True)

initialize_store()

@st.cache_resource
def load_predictor():
    return IntentPredictor()

st.markdown('<div class="bsd-header"><h1>BSD Bank</h1><p>Customer support</p></div>', unsafe_allow_html=True)
st.subheader("How can we help you?")
st.write("Enter your name and describe your banking question or issue.")

with st.form("customer_query_form", clear_on_submit=True):
    customer_name = st.text_input("Your name", max_chars=100, placeholder="Enter your name")
    query = st.text_area("Your query", max_chars=2000, height=140,
                         placeholder="For example: I forgot the PIN I use at the ATM.")
    submitted = st.form_submit_button("Submit query", type="primary", use_container_width=True)

if submitted:
    if not customer_name.strip():
        st.error("Please enter your name.")
    elif not query.strip():
        st.error("Please enter your query.")
    else:
        try:
            with st.spinner("Classifying your query..."):
                result = load_predictor().predict(query.strip(), with_analysis=False)
                submission_id = save_submission(
                    customer_name=customer_name.strip(),
                    query=query.strip(),
                    intent=result["intent"],
                    confidence=result["confidence"],
                )
            st.session_state["last_customer_result"] = {
                "submission_id": submission_id,
                "customer_name": customer_name.strip(),
                "intent": result["display"],
                "confidence": result["confidence"],
                "needs_review": result["needs_review"],
            }
        except Exception as exc:
            st.error(f"We could not process your query. Please try again. ({type(exc).__name__})")

result = st.session_state.get("last_customer_result")
if result:
    st.divider()
    st.subheader("Your query category")
    st.success(result["intent"])
    st.caption(f"Classification confidence: {result['confidence']:.1%}")
    if result["needs_review"]:
        st.info("Your wording may need a closer review. A support representative can help you further.")
    else:
        st.write("Your query has been recorded for the BSD Bank support team.")

