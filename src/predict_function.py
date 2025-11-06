import joblib
import pandas as pd
from typing import List, Dict, Any

MODEL_PATH = "models/model.pkl"

def load_model(path: str = MODEL_PATH):
    model = joblib.load(path)
    print(f"Loaded model from {path}")
    return model

def _prepare_input(record: Dict[str, Any], expected_features: List[str]) -> pd.DataFrame:
    df = pd.DataFrame([record])
    for col in expected_features:
        if col not in df.columns:
            df[col] = None
    df = df[expected_features]
    return df

def predict_single(model, record: Dict[str, Any]) -> Dict[str, Any]:
    expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else list(record.keys())
    X = _prepare_input(record, list(expected_features))
    proba = model.predict_proba(X)[:, 1][0]
    pred = model.predict(X)[0]
    return {"prediction": int(pred), "probability": float(proba)}

def predict_batch(model, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else list(records[0].keys())
    df = pd.DataFrame(records)
    for col in expected_features:
        if col not in df.columns:
            df[col] = None
    df = df[list(expected_features)]
    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)
    return [{"prediction": int(p), "probability": float(prob)} for p, prob in zip(preds, probs)]

if __name__ == "__main__":
    model = load_model()
    sample = {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "DSL",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 29.85,
        "total_charges": 500.5
    }
    result = predict_single(model, sample)
    print("Single prediction:", result)
    batch_result = predict_batch(model, [sample, sample])
    print("Batch prediction:", batch_result)
