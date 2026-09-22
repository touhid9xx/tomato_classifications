

import streamlit as st

from utils.knowledge import load_knowledge_base, all_rows, get_row_for_class

st.set_page_config(page_title="Self-Diagnose — Tomato Assistant", page_icon="📋", layout="wide")
st.title("📋 Self-Diagnose from Symptoms")

st.markdown("""
Answer the questions below based on what you see on your tomato plant.
Check **only** the symptoms you can clearly confirm. The app will rank the
most likely diseases.
""")

rows = all_rows()

if not rows:
    st.error("Knowledge base is empty or could not be loaded.")
    st.stop()

# Build the checkbox list — one per disease's follow-up question
st.subheader("Questions")

responses = {}
with st.form("symptom_form"):
    for row in rows:
        class_name = row["class_name"]
        display = row["display_name"]
        question = row["follow_up_question"]
        responses[class_name] = st.checkbox(f"**{display}** — {question}")
    submitted = st.form_submit_button("See results")

if not submitted:
    st.info("Check the symptoms you observe, then click **See results**.")
    st.stop()

# --- Scoring ---
# Each disease gets 1 point for each confirmed symptom.
# We only have one question per disease, so the score is 0 or 1.
# Ties are broken by showing all matches.
matches = [name for name, confirmed in responses.items() if confirmed]

st.divider()

if not matches:
    st.warning(
        "You didn't confirm any symptoms. If your plant looks healthy, "
        "check the **Healthy** entry below. Otherwise, take a photo and "
        "use the **Diagnose** page for image-based prediction."
    )
    # Still show the healthy entry as a fallback
    healthy = get_row_for_class("Tomato___healthy")
    if healthy:
        st.subheader("Healthy")
        st.write(healthy.get("description", "—"))
    st.stop()

if len(matches) == 1:
    st.success(f"**Most likely: {get_row_for_class(matches[0])['display_name']}**") # type: ignore
else:
    st.warning(
        f"You confirmed symptoms for **{len(matches)} diseases**. "
        "Some symptoms overlap — check the details below to narrow it down."
    )

# Show all matching diseases with their details
for class_name in matches:
    row = get_row_for_class(class_name)
    with st.expander(f"**{row['display_name']}**", expanded=True): # type: ignore
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Description**")
            assert row is not None
            st.write(row.get("description", "—"))
        with c2:
            st.markdown("**Recommended treatment**")
            st.write(row.get("treatment", "—"))

st.divider()
st.caption(
    "This tool uses a simple symptom-match approach, not a trained model. "
    "For image-based prediction, use the **Diagnose** page."
)