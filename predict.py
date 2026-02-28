import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "healthcare_fraud_model_v1_20260227_1924.pkl"
SCALER_PATH = "healthcare_fraud_scaler_v1_20260227_1924.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Get the feature names the scaler was trained on
EXPECTED_FEATURES = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else None
N_FEATURES = int(scaler.n_features_in_)


def run_prediction(csv_path):
    df = pd.read_csv(csv_path)

    if EXPECTED_FEATURES:
        # Build a feature matrix aligned exactly to what the scaler expects.
        # For each expected feature:
        #   - If it exists as a numeric column → use it
        #   - Otherwise → fill with 0
        feat_df = pd.DataFrame(index=df.index)
        for feat in EXPECTED_FEATURES:
            if feat in df.columns:
                feat_df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0)
            else:
                feat_df[feat] = 0.0
    else:
        # Fallback: use all numeric columns and pad/trim to match dimensionality
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        if numeric_df.shape[1] < N_FEATURES:
            # Pad with zeros
            for i in range(N_FEATURES - numeric_df.shape[1]):
                numeric_df[f'_pad_{i}'] = 0.0
        elif numeric_df.shape[1] > N_FEATURES:
            numeric_df = numeric_df.iloc[:, :N_FEATURES]
        feat_df = numeric_df

    scaled = scaler.transform(feat_df)

    df["anomaly"] = model.predict(scaled)
    df["anomaly_score"] = model.decision_function(scaled)

    # Compute risk_score normalised to 0–100
    raw = 1 - df["anomaly_score"]
    rmin, rmax = raw.min(), raw.max()
    if rmax == rmin:
        df["risk_score"] = 50.0
    else:
        df["risk_score"] = ((raw - rmin) / (rmax - rmin) * 100).round(2)

    def classify(score):
        if score >= 70:
            return "High"
        elif score >= 40:
            return "Medium"
        return "Low"

    df["Risk_Level"] = df["risk_score"].apply(classify)

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/processed_output.csv", index=False)
    return df