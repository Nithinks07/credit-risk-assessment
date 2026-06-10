import joblib
from preprocessing.preprocess import build_input_frame, prepare_features_for_model

model = joblib.load('models/random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

user_input = {
    'age': 45,
    'duration': 6,
    'credit_amount': 500,
    'housing': 'own',
    'checking_status': 'rich',
    'savings_status': 'rich',
    'purpose': 'car',
}

X = build_input_frame(user_input, feature_columns)
print(X.T)
X_model = prepare_features_for_model(X, scaler, feature_columns)

prob = model.predict_proba(X_model)[0, 1]
cls = int(model.predict(X_model)[0])

print('probability=', prob)
print('class=', cls)
print('label=', 'Bad Credit' if cls == 1 else 'Good Credit')