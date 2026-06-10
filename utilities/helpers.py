"""Utility helpers for the Streamlit frontend."""


def risk_category(probability: float) -> str:
    """Return a business-friendly risk label."""
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def risk_color(probability: float) -> str:
    """Return a simple color token for UI styling."""
    if probability >= 0.70:
        return "danger"
    if probability >= 0.40:
        return "warning"
    return "success"


def format_probability(probability: float) -> str:
    """Format probability as a percentage string."""
    return f"{probability * 100:.1f}%"
