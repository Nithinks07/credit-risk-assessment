"""Sidebar form for the credit-risk app."""

import streamlit as st


def render_sidebar() -> dict:
    """Render the first-pass applicant form for the MVP."""
    st.sidebar.header("Applicant Profile")
    st.sidebar.caption("Minimal MVP inputs for prediction flow.")

    with st.sidebar.form("applicant_form"):
        age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
        duration = st.number_input("Loan Duration (months)", min_value=3, max_value=72, value=18, step=1)
        credit_amount = st.number_input("Credit Amount", min_value=500, max_value=20000, value=3000, step=100)

        housing = st.selectbox("Housing", ["own", "rent", "free"])
        checking_status = st.selectbox("Checking Account", ["little", "moderate", "quite rich", "rich"])
        savings_status = st.selectbox("Savings Status", ["little", "moderate", "quite rich", "rich"])
        purpose = st.selectbox("Purpose", ["car", "radio/TV", "education", "furniture/equipment", "business", "repairs"])

        submitted = st.form_submit_button("Run Prediction")

    return {
        "age": age,
        "duration": duration,
        "credit_amount": credit_amount,
        "housing": housing,
        "checking_status": checking_status,
        "savings_status": savings_status,
        "purpose": purpose,
        "submitted": submitted,
    }
