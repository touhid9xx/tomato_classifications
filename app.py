"""Tomato Disease Diagnostic Assistant — landing page.

Multipage app. See pages/ for the Diagnose, Self-Diagnose, and About views.
"""

import streamlit as st

st.set_page_config(
    page_title="Tomato Disease Diagnostic",
    page_icon="🍅",
    layout="wide",
)

st.title("🍅 Tomato Disease Diagnostic Assistant")

st.markdown("""
This app helps identify common diseases on tomato plants from leaf images.

### What it does

- **Diagnose from image** — upload a photo of a tomato leaf, and a
  MobileNetV2 model trained on the PlantVillage dataset predicts the most
  likely disease with confidence scores.

- **Self-diagnose from symptoms** — answer a short set of yes/no questions
  about what you're seeing on your plant. The app cross-references your
  answers with known symptom profiles for each disease.

### Model

Trained on 6,557 tomato leaf images across 6 classes:

| Class | Example symptoms |
|-------|------------------|
| Bacterial Spot | Water-soaked spots with yellow halos |
| Early Blight | Concentric rings (target pattern) |
| Late Blight | Large greasy lesions, white mold |
| Leaf Mold | Olive-green velvety mold on undersides |
| Septoria Leaf Spot | Gray centers with black dots |
| Healthy | Uniform green leaves |

**Test accuracy:** 91.4% across 990 held-out images.

### How to use

Use the sidebar to navigate:

1. **Diagnose** — image-based prediction
2. **Self-Diagnose** — symptom-based questionnaire
3. **About** — model card, limitations, dataset

---

⚠️ **Disclaimer:** This is a research and portfolio project. Predictions
are not a substitute for professional agricultural advice.
""")

st.sidebar.success("Select a page above to begin.")