"""Load trained model artifacts for inference."""

import os

import joblib
import streamlit as st


@st.cache_resource
def load_artifacts(model_name: str = "random_forest.pkl"):
    """Load model, scaler and feature names once for the session."""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    model_path = os.path.join(base_dir, model_name)
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    feature_path = os.path.join(base_dir, "feature_columns.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_path)

    return model, scaler, feature_columns
