# Tomato Disease Diagnostic

A Streamlit app that identifies common tomato plant diseases from leaf images, backed by a MobileNetV2 classifier trained on Databricks and registered in MLflow.

**Live demo:** [tomato-diagnostic.streamlit.app](<your-app-url>)

## Objective

Build an end-to-end ML pipeline — data ingestion, training, experiment tracking, model registry, and a user-facing app — and demonstrate that the deployed app pulls the model directly from the MLflow registry rather than a hardcoded file.

## Results

- **Test accuracy:** 91.4% on 990 held-out images
- **Classes:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Healthy
- **Training data:** 6,557 images from the PlantVillage tomato subset

## Tech stack

| Layer | Tool |
|-------|------|
| Compute & storage | Databricks Free Edition, Unity Catalog Volumes |
| Training | PyTorch, torchvision (MobileNetV2, frozen backbone) |
| Tracking & registry | MLflow, Unity Catalog model registry (`@champion` alias) |
| UI | Streamlit |
| Metrics | scikit-learn |

## How it works

1. Images uploaded to a Unity Catalog volume
2. Training notebook (MobileNetV2, transfer learning) logs params, metrics, and model to MLflow
3. Best run promoted to `@champion` alias in Unity Catalog
4. Streamlit app loads `models:/...@champion` at startup and runs inference

## How to run

**Prerequisites:** Python 3.12, a Databricks workspace with the model registered, and a Personal Access Token with `Can view` permission on the model.

```bash
git clone https://github.com/touhid9xx/tomato_classifications.git
cd tomato_classifications/tomato-streamlit-app

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt


Create .streamlit/secrets.toml:
DATABRICKS_HOST = "https://<your-workspace>.cloud.databricks.com"
DATABRICKS_TOKEN = "dapi..."
MODEL_URI = "models:/workspace.default.tomato_disease_classifier@champion"

Run:
streamlit run app.py

```
## Dataset source

This project uses the **Processed Tomato Leaf Disease Image Dataset** on Mendeley Data, a preprocessed and pre-split version of the PlantVillage tomato subset.

- **Mendeley Data:** [Processed Tomato Leaf Disease Image Dataset](https://data.mendeley.com/datasets/3zwdw6y4pn/1)
- **Original PlantVillage paper:** [Hughes & Salathé, 2015](https://arxiv.org/abs/1511.08060)
- **PlantVillage on Kaggle:** [emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

The Mendeley version was used because it already has the six tomato classes cleaned, deduplicated, and split into train/val/test, which saved the preprocessing work.

