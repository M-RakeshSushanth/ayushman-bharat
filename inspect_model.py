import joblib
s = joblib.load('healthcare_fraud_scaler_v1_20260227_1924.pkl')
features = list(s.feature_names_in_)
print('Number of features:', len(features))
print('Features:', features)
