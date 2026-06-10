"""Minimal Streamlit entry point for the Explainable Credit Risk Assessment app."""

import pandas as pd
import streamlit as st

from explainability.shap_analysis import build_textual_explanation
from frontend.layouts.sidebar import render_sidebar
from frontend.pages.prediction import render_prediction_panel
from model_loading.loader import load_artifacts
from preprocessing.preprocess import build_input_frame, prepare_features_for_model


st.set_page_config(page_title="Credit Risk Assessment", page_icon="", layout="wide")


def main() -> None:
    st.title("Explainable Credit Risk Assessment")
    st.caption("Minimal MVP: input → preprocess → predict → explainability-ready dashboard")

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    model, scaler, feature_columns = load_artifacts("random_forest.pkl")

    st.sidebar.title("Banking AI Workspace")
    st.sidebar.caption("Current backend uses saved .pkl artifacts from the training pipeline.")

    user_input = render_sidebar()

    if user_input["submitted"]:
        try:
            input_frame = build_input_frame(user_input, feature_columns)
            model_input = prepare_features_for_model(input_frame, scaler, feature_columns)

            probability = float(model.predict_proba(model_input)[0, 1])
            prediction_label = "Bad Credit" if probability >= 0.5 else "Good Credit"
            explanation = build_textual_explanation(model, model_input, feature_columns)

            st.session_state.last_prediction = {
                "probability": probability,
                "prediction_label": prediction_label,
                "user_input": user_input,
                "explanation": explanation,
                "model": model,
                "model_input": model_input,
                "feature_columns": feature_columns,
            }

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.caption("Check the saved model artifact alignment and input schema before deployment.")

    last_prediction = st.session_state.get("last_prediction")
    if last_prediction is not None:
        render_prediction_panel(
            last_prediction["probability"],
            last_prediction["prediction_label"],
            last_prediction["user_input"],
            last_prediction["explanation"],
            last_prediction["model"],
            last_prediction["model_input"],
            last_prediction["feature_columns"],
        )
        st.info("The explainability section now refreshes from the latest submitted applicant input.")


if __name__ == "__main__":
    main()
