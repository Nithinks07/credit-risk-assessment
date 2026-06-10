"""SHAP explainability helpers for the Streamlit frontend."""

import numpy as np

import shap
import streamlit as st


FEATURE_LABELS = {
    "credit_per_month": "Monthly credit burden",
    "checking_status": "Checking account status",
    "credit_amount": "Credit amount",
    "installment_burden": "Installment burden",
    "age": "Age",
    "duration": "Loan duration",
    "credit_history": "Credit history",
    "purpose": "Loan purpose",
    "employment": "Employment status",
    "savings_status": "Savings status",
    "property_magnitude": "Property value",
    "residence_since": "Years at current residence",
    "personal_status": "Personal status",
    "age_group": "Age group",
    "installment_commitment": "Installment commitment",
    "existing_credits": "Existing credits",
    "foreign_worker": "Foreign worker status",
    "housing": "Housing type",
    "job": "Job category",
    "num_dependents": "Dependents",
    "other_parties": "Other parties",
    "other_payment_plans": "Payment plans",
    "high_credit": "High credit flag",
    "long_duration": "Long duration flag",
}


@st.cache_resource
def load_shap_explainer(_model):
    """Create and cache the SHAP explainer for the loaded model."""
    return shap.TreeExplainer(_model)


def _normalize_values(raw_values):
    """Convert SHAP outputs to a simple 2D array for a single sample."""
    if isinstance(raw_values, list):
        raw_values = raw_values[1] if len(raw_values) == 2 else raw_values[-1]

    values = np.asarray(raw_values, dtype=float)

    if values.ndim == 3:
        values = values[:, :, 1]

    if values.ndim == 2 and values.shape[0] == 1:
        return values[0]

    return values


def build_textual_explanation(model, model_input, feature_names):
    """Create a business-friendly explanation summary for the frontend."""
    explainer = load_shap_explainer(model)
    shap_values = explainer.shap_values(model_input)
    values = _normalize_values(shap_values)

    top_increasing = []
    top_reducing = []

    for feature, contribution in zip(feature_names, values):
        if contribution > 0.03:
            top_increasing.append((feature, contribution))
        elif contribution < -0.03:
            top_reducing.append((feature, contribution))

    top_increasing = sorted(top_increasing, key=lambda item: item[1], reverse=True)[:4]
    top_reducing = sorted(top_reducing, key=lambda item: item[1])[:4]

    def friendly_label(name):
        return FEATURE_LABELS.get(name, name.replace('_', ' ').title())

    top_increasing_labels = [friendly_label(name) for name, _ in top_increasing[:3]]
    top_reducing_labels = [friendly_label(name) for name, _ in top_reducing[:3]]

    if top_increasing_labels:
        driver_text = ", ".join(top_increasing_labels)
        business_summary = (
            f"The model gives the strongest attention to {driver_text}. "
            "These signals are consistent with higher repayment pressure and elevated default likelihood."
        )
    else:
        business_summary = "The current decision is driven mainly by stable profile characteristics rather than severe repayment pressure."

    if top_reducing_labels:
        business_summary += f" The profile also benefits from {', '.join(top_reducing_labels)} supporting a lower risk view."

    business_summary += " This explanation is intended for decision support and should be reviewed alongside standard banking judgment."

    return {
        "top_increasing": [(friendly_label(name), round(float(value), 3)) for name, value in top_increasing],
        "top_reducing": [(friendly_label(name), round(float(value), 3)) for name, value in top_reducing],
        "business_summary": business_summary,
        "risk_driver_count": len(top_increasing),
        "risk_reducer_count": len(top_reducing),
    }


def build_waterfall_plot(model, model_input, feature_names):
    """Create a SHAP waterfall plot for the current applicant."""
    try:
        import warnings

        import matplotlib.pyplot as plt

        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

        explainer = load_shap_explainer(model)
        shap_values = _normalize_values(explainer.shap_values(model_input))

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            base_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            base_value = expected_value

        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=model_input.iloc[0].to_numpy(),
            feature_names=list(feature_names),
        )

        shap.plots.waterfall(explanation, max_display=8)
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    except Exception:
        return None
