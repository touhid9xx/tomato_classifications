import os
import streamlit as st
import mlflow
import mlflow.pyfunc


def _configure_mlflow() -> str:
    """Set Databricks credentials from Streamlit secrets and return the model URI."""
    required = ["DATABRICKS_HOST", "DATABRICKS_TOKEN", "MODEL_URI"]
    missing = [k for k in required if k not in st.secrets]
    if missing:
        raise RuntimeError(
            f"Missing keys in .streamlit/secrets.toml: {missing}. "
            "Add DATABRICKS_HOST, DATABRICKS_TOKEN, and MODEL_URI."
        )

    os.environ["DATABRICKS_HOST"] = st.secrets["DATABRICKS_HOST"]
    os.environ["DATABRICKS_TOKEN"] = st.secrets["DATABRICKS_TOKEN"]

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")  # ← ADD THIS LINE
    return st.secrets["MODEL_URI"]


@st.cache_resource(show_spinner="Loading model from MLflow registry...")
def load_model():
   
    model_uri = _configure_mlflow()
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        st.error(
            "Failed to load the model from MLflow. Common causes:\n"
            "- Invalid or expired DATABRICKS_TOKEN\n"
            "- Token lacks 'Can view' permission on the model\n"
            "- MODEL_URI points to a version without a `champion` alias\n\n"
            f"Underlying error: {e}"
        )
        st.stop()


def get_model_uri() -> str:
    
    return st.secrets.get("MODEL_URI", "models:/<unknown>@champion")