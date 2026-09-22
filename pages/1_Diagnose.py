

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from utils.model import load_model, get_model_uri
from utils.preprocessing import preprocess_for_model, preprocess_for_display
from utils.knowledge import get_row_for_class

st.set_page_config(page_title="Diagnose — Tomato Assistant", page_icon="🔍", layout="wide")
st.title("🔍 Diagnose from Image")

st.caption(f"Model: `{get_model_uri()}`")

# Class order must match training. This is the alphabetical order ImageFolder produced.
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___healthy",
]

DISPLAY = {
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___healthy": "Healthy",
}

# --- Model (cached) ---
model = load_model()

# --- Upload ---
uploaded = st.file_uploader(
    "Upload a tomato leaf image (JPG or PNG, up to 5 MB)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info("Upload a leaf image to get a diagnosis.")
    st.stop()

if uploaded.size > 5 * 1024 * 1024:
    st.error("File is larger than 5 MB. Please compress or resize before uploading.")
    st.stop()

try:
    image = Image.open(uploaded)
    image.load()
except Exception as e:
    st.error(f"Could not read the image. Make sure it's a valid JPG or PNG. ({e})")
    st.stop()

# --- Layout: image on the left, predictions on the right ---
col_img, col_pred = st.columns([1, 1])

with col_img:
    st.subheader("Uploaded image")
    st.image(preprocess_for_display(image), use_container_width=True)
    st.caption(f"Original size: {image.size[0]}×{image.size[1]}")

with col_pred:
    st.subheader("Prediction")

    with st.spinner("Running inference..."):
        batch = preprocess_for_model(image)
        preds = model.predict(batch)
        preds = np.asarray(preds)

        # PyFunc may return class indices, one-hot, or probability vectors.
        if preds.ndim == 2 and preds.shape[1] == len(CLASS_NAMES):
            probs = preds[0]
            pred_idx = int(np.argmax(probs))
        elif preds.ndim == 2:
            pred_idx = int(np.argmax(preds[0]))
            probs = None
        else:
            pred_idx = int(preds[0])
            probs = None

predicted_class = CLASS_NAMES[pred_idx]
predicted_display = DISPLAY[predicted_class]

if probs is not None:
    # Show top-3 as a bar chart
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_df = pd.DataFrame({
        "Class": [DISPLAY[CLASS_NAMES[i]] for i in top3_idx],
        "Confidence": [float(probs[i]) for i in top3_idx],
    }).set_index("Class")
    st.bar_chart(top3_df, horizontal=True)
    st.metric("Top prediction", predicted_display, f"{probs[pred_idx]*100:.1f}% confidence")
else:
    st.metric("Predicted class", predicted_display)
    st.caption("Confidence scores not available — model returned class indices only.")

# --- Diagnosis card ---
st.divider()
st.subheader(f"About: {predicted_display}")

row = get_row_for_class(predicted_class)
if row is None:
    st.warning("No knowledge base entry found for this class.")
else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Description**")
        st.write(row.get("description", "—"))
    with c2:
        st.markdown("**Recommended treatment**")
        st.write(row.get("treatment", "—"))

    st.info(
        "Not sure this is correct? Try the **Self-Diagnose** page in the "
        "sidebar, or re-upload a clearer photo of the affected area."
    )