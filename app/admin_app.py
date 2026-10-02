"""BSD Bank local admin query inbox. Run on localhost port 8502."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from submissions import get_submissions, initialize_store

st.set_page_config(page_title="BSD Bank Admin", layout="wide")
st.title("BSD Bank | Admin")
st.caption("Customer names, original queries, and predicted intent categories")
initialize_store()

if st.button("Refresh inbox"):
    st.rerun()

records = get_submissions()
if not records:
    st.info("No customer queries have been submitted yet.")
else:
    frame = pd.DataFrame(records).rename(columns={
        "submitted_at": "Submitted",
        "customer_name": "Customer",
        "query": "Customer query",
        "intent": "Predicted category",
        "confidence": "Confidence",
    })
    frame["Confidence"] = frame["Confidence"].map(lambda value: f"{value:.1%}")
    st.dataframe(frame[["Submitted", "Customer", "Customer query", "Predicted category", "Confidence"]],
                 hide_index=True, use_container_width=True)
