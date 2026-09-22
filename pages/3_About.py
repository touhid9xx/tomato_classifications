"""Model card and project information page."""

import streamlit as st
from utils.model import get_model_uri

st.set_page_config(page_title="About — Tomato Assistant", page_icon="ℹ️", layout="wide")
st.title("ℹ️ About This Project")

st.markdown(f"""
## Model

**Architecture:** MobileNetV2 (ImageNet pretrained, backbone frozen)

**Head:** Linear(1280 → 6)

**Model URI:** `{get_model_uri()}`

**Training data:** [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)
tomato subset — 6,557 images, 6 classes.

| Split | Images |
|-------|--------|
| Train | 4,586 |
| Val | 981 |
| Test | 990 |

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | 91.4% |
| Macro F1 | 0.913 |
| Best val accuracy | 92.8% |

### Per-class F1

| Class | F1 |
|-------|-----|
| Bacterial Spot | 0.940 |
| Early Blight | 0.852 |
| Late Blight | 0.899 |
| Leaf Mold | 0.911 |
| Septoria Leaf Spot | 0.910 |
| Healthy | 0.970 |

## Known limitations

- **Early Blight is the hardest class** (F1 0.85). Its visual symptoms overlap
  with Late Blight and Septoria — a known confusion in the PlantVillage benchmark.

- **Lab-condition training data.** PlantVillage images were captured under
  controlled conditions. Real-world phone photos (varied lighting, angles,
  backgrounds) may see lower accuracy. This is called the *domain gap*.

- **Not a substitute for expert advice.** This is a portfolio project
  demonstrating an end-to-end ML pipeline, not a production diagnostic tool.

## Technical stack

- **Databricks Free Edition** — training, MLflow tracking, Unity Catalog model registry
- **MLflow** — experiment tracking and model versioning
- **Streamlit** — this UI
- **PyTorch / torchvision** — model training and inference

## Links

- [Source code on GitHub](#) *(add your repo URL)*
- [MLflow experiment](#) *(add link to your Databricks experiment)*
- [Model card in Unity Catalog](#) *(add link)*
""")

st.divider()
st.caption(
    "Built as a portfolio project demonstrating Databricks + MLflow + Streamlit. "
    "Feedback welcome."
)