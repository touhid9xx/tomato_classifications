

from pathlib import Path
import pandas as pd
import streamlit as st

CSV_PATH = Path(__file__).parent.parent / "knowledge_base.csv"


@st.cache_data(show_spinner=False)
def load_knowledge_base() -> pd.DataFrame:
    
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"knowledge_base.csv not found at {CSV_PATH}. "
            "Make sure the file is in the project root next to app.py."
        )
    df = pd.read_csv(CSV_PATH)
    # Normalize the expected_answer column in case trailing whitespace snuck in
    if "expected_answer" in df.columns:
        df["expected_answer"] = df["expected_answer"].astype(str).str.strip().str.lower()
    return df


def get_row_for_class(class_name: str) -> dict | None:
    """Return the knowledge base row for a class as a dict, or None."""
    df = load_knowledge_base()
    match = df[df["class_name"] == class_name]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def all_rows() -> list[dict]:
   
    df = load_knowledge_base()
    return df.to_dict(orient="records")