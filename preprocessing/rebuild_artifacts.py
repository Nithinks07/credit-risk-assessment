"""Rebuild inference artifacts from the same training pipeline used in the notebook.

This script regenerates the saved model/scaler/feature metadata so that the
frontend inference pipeline uses the same encoding and engineered feature schema
as the training notebook.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"


def load_dataset():
    """Load the German Credit dataset in the same format as the notebook."""
    col_names = [
        'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings_status', 'employment', 'installment_commitment', 'personal_status',
        'other_parties', 'residence_since', 'property_magnitude', 'age',
        'other_payment_plans', 'housing', 'existing_credits', 'job', 'num_dependents',
        'own_telephone', 'foreign_worker', 'class'
    ]
    df = pd.read_csv(DATASET_URL, sep=' ', header=None, names=col_names)
    df.columns = [c.strip() for c in df.columns]
    return df


def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add the same engineered features used in the notebook."""
    df_fe = df_in.copy()

    if 'credit_amount' in df_fe.columns and 'duration' in df_fe.columns:
        df_fe['credit_per_month'] = df_fe['credit_amount'] / (df_fe['duration'] + 1)

    if 'age' in df_fe.columns:
        df_fe['age_group'] = pd.cut(
            df_fe['age'],
            bins=[0, 25, 35, 50, 100],
            labels=[0, 1, 2, 3]
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


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_dataset()
    target_col = 'class'

    # Basic target mapping used in training notebook.
    y = df[target_col].copy().map({1: 0, 2: 1})
    X = df.drop(columns=[target_col]).copy()

    # Fit per-column LabelEncoders exactly the way the notebook does.
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    X_encoded = X.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le

    # Compute reliable defaults from the training data for inference fallback.
    numeric_defaults = X.select_dtypes(include=['number']).median().to_dict()
    categorical_defaults = X.select_dtypes(include=['object']).mode().iloc[0].to_dict()

    # Feature engineering
    X_fe = engineer_features(X_encoded)

    # Remove low-variance / highly correlated features (same notebook logic)
    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X_fe)
    removed = [col for col, kept in zip(X_fe.columns, vt.get_support()) if not kept]
    if removed:
        X_fe = X_fe.drop(columns=removed)

    corr_mat = X_fe.corr().abs()
    upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    if to_drop_corr:
        X_fe = X_fe.drop(columns=to_drop_corr)

    # Feature importance ranking
    rf_quick = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_quick.fit(X_fe, y)
    importances = pd.Series(rf_quick.feature_importances_, index=X_fe.columns).sort_values(ascending=False)
    top_features = importances.head(15).index.tolist()

    X_selected = X_fe[top_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler_fe = StandardScaler()
    X_train_fe_sc = pd.DataFrame(scaler_fe.fit_transform(X_train), columns=top_features)
    X_test_fe_sc = pd.DataFrame(scaler_fe.transform(X_test), columns=top_features)

    models = {
        'Logistic Regression': None,
        'Decision Tree': None,
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    # Train the same models used in the notebook.
    for name, model in models.items():
        if model is None:
            continue
        cv_acc = cross_val_score(model, X_train_fe_sc, y_train, cv=cv, scoring='accuracy').mean()
        cv_auc = cross_val_score(model, X_train_fe_sc, y_train, cv=cv, scoring='roc_auc').mean()
        model.fit(X_train_fe_sc, y_train)
        y_pred = model.predict(X_test_fe_sc)
        acc = accuracy_score(y_test, y_pred)
        results.append({
            'Model': name,
            'CV Accuracy': round(cv_acc, 4),
            'CV ROC-AUC': round(cv_auc, 4),
            'Test Accuracy': round(acc, 4),
        })
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl"))

    # Save the corrected scaler and feature list.
    joblib.dump(scaler_fe, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(top_features, os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'label_encoders.pkl'))

    # Persist defaults for inference fallback.
    preprocessing_metadata = {
        'categorical_columns': categorical_cols,
        'numeric_columns': X.select_dtypes(include=['number']).columns.tolist(),
        'numeric_defaults': numeric_defaults,
        'categorical_defaults': categorical_defaults,
        'top_features': top_features,
        'engineered_features': [col for col in X_fe.columns if col not in X.columns],
    }
    with open(os.path.join(MODELS_DIR, 'preprocessing_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(preprocessing_metadata, f, indent=2)

    print('Artifacts rebuilt successfully.')
    print('Top features:', top_features)
    print('Scaler shape:', scaler_fe.mean_.shape)
    print('Saved model files:', [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')])
    print('Results:')
    print(pd.DataFrame(results))


if __name__ == '__main__':
    main()
