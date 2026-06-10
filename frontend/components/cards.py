"""Reusable card-style UI helpers."""

import streamlit as st


def render_metric_card(title: str, value: str, subtitle: str = "") -> None:
    """Render a simple metric card in the main dashboard."""
    st.markdown(
        f"""
        <div style="background:#0f172a; padding:16px; border-radius:14px; border:1px solid #334155; margin-bottom:12px;">
            <div style="color:#cbd5e1; font-size:12px; text-transform:uppercase; letter-spacing:1.2px;">{title}</div>
            <div style="color:white; font-size:28px; font-weight:700;">{value}</div>
            <div style="color:#94a3b8; font-size:13px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
