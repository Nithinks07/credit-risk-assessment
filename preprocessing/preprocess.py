"""Preprocessing helpers that mirror the notebook training pipeline for inference."""

import json
import os

import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _load_preprocessing_metadata():
    """Load the training-time preprocessing metadata used for inference."""
    metadata_path = os.path.join(MODELS_DIR, "preprocessing_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # Conservative fallback for older saved artifacts.
    return {
        "categorical_columns": [
            "checking_status", "credit_history", "purpose", "savings_status",
            "employment", "property_magnitude", "personal_status", "other_payment_plans",
            "housing", "job", "other_parties", "own_telephone", "foreign_worker"
        ],
        "numeric_columns": [
            "duration", "credit_amount", "installment_commitment", "residence_since",
            "existing_credits", "age", "num_dependents"
        ],
        "numeric_defaults": {
            "duration": 12.0,
            "credit_amount": 2000.0,
            "installment_commitment": 1.0,
            "residence_since": 1.0,
            "existing_credits": 1.0,
            "age": 30.0,
            "num_dependents": 1.0,
        },
        "categorical_defaults": {
            "checking_status": "little",
            "credit_history": "critical/other existing credit",
            "purpose": "car",
            "savings_status": "little",
            "employment": "1<= <4 years",
            "property_magnitude": "real estate",
            "personal_status": "male single",
            "other_payment_plans": "none",
            "housing": "own",
            "job": "skilled employee / official",
            "other_parties": "none",
            "own_telephone": "yes",
            "foreign_worker": "yes",
        },
    }


def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add the same engineered features used in the notebook training pipeline."""
    df_fe = df_in.copy()

    if 'credit_amount' in df_fe.columns and 'duration' in df_fe.columns:
        df_fe['credit_per_month'] = df_fe['credit_amount'] / (df_fe['duration'] + 1)

    if 'age' in df_fe.columns:
        df_fe['age_group'] = pd.cut(
            df_fe['age'],
            bins=[0, 25, 35, 50, 100],
            labels=[0, 1, 2, 3],
        ).astype(int)

    if 'credit_amount' in df_fe.columns:
        q75 = df_fe['credit_amount'].quantile(0.75)
        df_fe['high_credit'] = (df_fe['credit_amount'] > q75).astype(int)

    if 'duration' in df_fe.columns:
        med_dur = df_fe['duration'].median()
        df_fe['long_duration'] = (df_fe['duration'] > med_dur).astype(int)

    if 'installment_commitment' in df_fe.columns and 'duration' in df_fe.columns:
        df_fe['installment_burden'] = df_fe['installment_commitment'] * df_fe['duration']

    return df_fe


def build_input_frame(user_input: dict, expected_columns: list[str]) -> pd.DataFrame:
    """Build a model-aligned feature frame from UI input using the saved encoders."""
    metadata = _load_preprocessing_metadata()
    label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl')) if os.path.exists(os.path.join(MODELS_DIR, 'label_encoders.pkl')) else {}

    categorical_cols = metadata.get('categorical_columns', [])
    numeric_cols = metadata.get('numeric_columns', [])
    numeric_defaults = metadata.get('numeric_defaults', {})
    categorical_defaults = metadata.get('categorical_defaults', {})

    raw_values = {}
    for col in categorical_cols:
        raw_values[col] = user_input.get(col, categorical_defaults.get(col, ''))

    for col in numeric_cols:
        raw_values[col] = user_input.get(col, numeric_defaults.get(col, 0.0))

    # Add the UI fields the form actually provides.
    for key in ('age', 'duration', 'credit_amount', 'housing', 'checking_status', 'savings_status', 'purpose'):
        if key in user_input:
            raw_values[key] = user_input[key]

    raw_frame = pd.DataFrame([raw_values])

    # Convert numeric values to floats and fill missing values with training defaults.
    for col in numeric_cols:
        raw_frame[col] = pd.to_numeric(raw_frame[col], errors='coerce').fillna(float(numeric_defaults.get(col, 0.0)))

    # Apply the fitted label encoders used in training.
    for col in categorical_cols:
        if col not in label_encoders:
            continue

        le = label_encoders[col]
        values = raw_frame[col].astype(str)
        known_values = set(le.classes_.tolist())
        fallback_value = categorical_defaults.get(col, le.classes_[0] if len(le.classes_) else '')

        values = values.map(lambda value: value if value in known_values else fallback_value)
        raw_frame[col] = le.transform(values.astype(str)).astype(float)

    # Fill any remaining categorical fields that were not encoded.
    for col in categorical_cols:
        if col not in raw_frame.columns:
            raw_frame[col] = 0.0

    engineered = engineer_features(raw_frame)
    feature_frame = engineered.reindex(columns=expected_columns, fill_value=0.0)

    return feature_frame.astype(float)


def prepare_features_for_model(input_frame: pd.DataFrame, scaler, expected_columns: list[str]):
    """Scale the engineered feature frame using the saved scaler when possible."""
    feature_frame = input_frame.reindex(columns=expected_columns, fill_value=0.0)

    if scaler is None:
        return feature_frame

    try:
        scaled = scaler.transform(feature_frame)
        return pd.DataFrame(scaled, columns=expected_columns)
    except ValueError:
        return feature_frame
