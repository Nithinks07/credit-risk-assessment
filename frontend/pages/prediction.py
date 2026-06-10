"""Prediction and result rendering for the Streamlit app."""

import streamlit as st

from explainability.shap_analysis import build_waterfall_plot
from utilities.helpers import format_probability, risk_category, risk_color


def render_prediction_panel(probability: float, prediction_label: str, user_input: dict,
                            explanation: dict | None = None,
                            model=None,
                            model_input=None,
                            feature_columns=None) -> None:
    """Render a simple, professional prediction display."""
    category = risk_category(probability)
    color = risk_color(probability)

    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Probability", format_probability(probability))
    with col2:
        st.metric("Risk Category", category)
    with col3:
        st.metric("Decision", prediction_label)

    st.progress(min(probability, 1.0))

    if color == "danger":
        st.error("High-risk signal detected. The applicant should be reviewed carefully.")
    elif color == "warning":
        st.warning("Moderate risk. Manual review is recommended before final approval.")
    else:
        st.success("Low-risk signal. The applicant looks financially stable based on the current model.")

    st.markdown("---")
    st.subheader("Why This Prediction Was Made")
    st.caption("This section updates from the latest submitted applicant values and regenerates the SHAP-based explanation for that run.")

    if explanation:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Top Risk Drivers**")
            if explanation.get("top_increasing"):
                for label, value in explanation["top_increasing"]:
                    st.write(f"• {label} — contributes to a higher risk signal (impact: {value:.3f})")
            else:
                st.caption("No strong risk-increasing factors were detected for this applicant.")

        with col_right:
            st.markdown("**Features Reducing Risk**")
            if explanation.get("top_reducing"):
                for label, value in explanation["top_reducing"]:
                    st.write(f"• {label} — supports a lower risk signal (impact: {value:.3f})")
            else:
                st.caption("No strong risk-reducing factors were detected for this applicant.")

        st.markdown("**Business Interpretation**")
        st.info(explanation.get("business_summary", "The explanation is based on the current model output and applicant profile."))

        st.caption(
            f"Risk drivers identified: {explanation.get('risk_driver_count', 0)} | "
            f"Risk-reducing signals identified: {explanation.get('risk_reducer_count', 0)}"
        )

        with st.expander("Advanced Explainability", expanded=False):
            st.caption("This section shows the current applicant-level SHAP view for a more detailed decision explanation.")
            if model is not None and model_input is not None and feature_columns is not None:
                fig = build_waterfall_plot(model, model_input, feature_columns)
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.warning("Waterfall plot could not be generated for this input. The textual explanation is still available.")
            else:
                st.info("SHAP waterfall plot will appear when the model and applicant input are available.")
    else:
        st.info("Explainability is unavailable for this session. Please retry the prediction.")
