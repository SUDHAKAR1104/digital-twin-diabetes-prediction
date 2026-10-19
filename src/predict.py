import joblib
import pandas as pd
import shap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "risk_model.pkl"

saved_data = joblib.load(MODEL_PATH)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "risk_model.pkl"
saved_data = joblib.load(MODEL_PATH)



model = saved_data["model"]

features = saved_data["features"]
explainer = shap.TreeExplainer(model)
def predict_risk(patient_data):
    input_df = pd.DataFrame([patient_data], columns=features)

    probability = float(model.predict_proba(input_df)[0][1])

    if probability < 0.30:
        risk_level = "Low"
    elif probability < 0.60:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    explanation = explainer(input_df, check_additivity=False)
    values = explanation.values

    if values.ndim == 3:
        contributions = values[0, :, 1]
    else:
        contributions = values[0]

    ranked_indices = sorted(
        range(len(features)),
        key=lambda i: abs(float(contributions[i])),
        reverse=True,
    )[:3]

    top_factors = []

    for i in ranked_indices:
        contribution = float(contributions[i])
        top_factors.append({
            "feature": features[i],
            "contribution": contribution,
            "direction": (
                "increases model output"
                if contribution > 0
                else "decreases model output"
            ),
        })

    return {
        "risk_prediction": int(probability >= 0.50),
        "risk_probability": probability,
        "risk_level": risk_level,
        "top_factors": top_factors,
    }