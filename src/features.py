"""
features.py
Turns raw patient data (EHR + wearable) into model inputs.

GOLDEN RULE: every feature at time t uses only readings from time t and EARLIER.
No future values are used, so the model can never "see the answer".

Raw columns expected:
    patient_id, timestamp, glucose, heart_rate, steps, sleep_hours,
    age, sex, bmi, hba1c, years_since_diagnosis   (+ spike_2h label when training)
Readings must be consecutive 5-minute samples.
"""
import numpy as np
import pandas as pd

STEP_MIN = 5
WINDOWS = {30: 6, 60: 12, 120: 24}   # minutes -> number of 5-minute readings
MIN_HISTORY = 24                      # readings needed before a prediction (2 hours)
LABEL_COL = "spike_2h"
SPIKE_THRESHOLD = 180                 # mg/dL

FEATURE_COLS = [
    # glucose (time-series)
    "glucose", "glucose_mean_30m", "glucose_mean_60m", "glucose_mean_120m",
    "glucose_std_30m", "glucose_std_60m", "glucose_std_120m",
    "glucose_max_60m", "glucose_max_120m", "glucose_min_60m",
    "glucose_change_30m", "glucose_change_60m",
    # wearables
    "steps_30m", "steps_60m", "steps_120m", "hr_mean_30m", "hr_mean_60m", "sleep_hours",
    # time of day
    "hour_sin", "hour_cos",
    # static EHR (data fusion)
    "age", "sex_male", "bmi", "hba1c", "years_since_diagnosis",
]

# Domain knowledge the model must respect (also keeps the what-if simulator sensible):
#   +1 = more of this never lowers risk, -1 = more of this never raises risk
MONOTONIC = {
    "glucose": 1, "hba1c": 1,
    "steps_30m": -1, "steps_60m": -1, "steps_120m": -1,
    "sleep_hours": -1,
}

# Human-readable names for the "Why this risk?" panel
FEATURE_LABELS = {
    "glucose": "Current glucose",
    "glucose_mean_30m": "Average glucose (last 30 min)",
    "glucose_mean_60m": "Average glucose (last 1 h)",
    "glucose_mean_120m": "Average glucose (last 2 h)",
    "glucose_std_30m": "Glucose variability (30 min)",
    "glucose_std_60m": "Glucose variability (1 h)",
    "glucose_std_120m": "Glucose variability (2 h)",
    "glucose_max_60m": "Highest glucose (last 1 h)",
    "glucose_max_120m": "Highest glucose (last 2 h)",
    "glucose_min_60m": "Lowest glucose (last 1 h)",
    "glucose_change_30m": "Glucose trend (30 min)",
    "glucose_change_60m": "Glucose trend (1 h)",
    "steps_30m": "Steps (last 30 min)",
    "steps_60m": "Steps (last 1 h)",
    "steps_120m": "Steps (last 2 h)",
    "hr_mean_30m": "Heart rate (last 30 min)",
    "hr_mean_60m": "Heart rate (last 1 h)",
    "sleep_hours": "Last night's sleep",
    "hour_sin": "Time of day",
    "hour_cos": "Time of day",
    "age": "Age",
    "sex_male": "Sex",
    "bmi": "BMI",
    "hba1c": "HbA1c",
    "years_since_diagnosis": "Years since diagnosis",
}


def _one_patient(g: pd.DataFrame) -> pd.DataFrame:
    """Features for ONE patient whose rows are sorted by time."""
    glu, steps, hr = g["glucose"], g["steps"], g["heart_rate"]
    f = pd.DataFrame(index=g.index)
    f["glucose"] = glu
    for m, w in WINDOWS.items():
        r = glu.rolling(w, min_periods=w)
        f[f"glucose_mean_{m}m"] = r.mean()
        f[f"glucose_std_{m}m"] = r.std()
        if m >= 60:
            f[f"glucose_max_{m}m"] = r.max()
    f["glucose_min_60m"] = glu.rolling(12, min_periods=12).min()
    f["glucose_change_30m"] = glu - glu.shift(6)
    f["glucose_change_60m"] = glu - glu.shift(12)
    for m, w in WINDOWS.items():
        f[f"steps_{m}m"] = steps.rolling(w, min_periods=w).sum()
    for m in (30, 60):
        f[f"hr_mean_{m}m"] = hr.rolling(WINDOWS[m], min_periods=WINDOWS[m]).mean()
    f["sleep_hours"] = g["sleep_hours"]
    hod = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hod / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hod / 24)
    f["age"] = g["age"]
    f["sex_male"] = (g["sex"] == "M").astype(int)
    f["bmi"] = g["bmi"]
    f["hba1c"] = g["hba1c"]
    f["years_since_diagnosis"] = g["years_since_diagnosis"]
    return f


def build_features(df: pd.DataFrame, keep_label: bool = True) -> pd.DataFrame:
    """Return one row of features per timestamp (rows without 2 h of history are dropped)."""
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    parts = []
    for pid, g in df.groupby("patient_id", sort=False):
        g = g.reset_index(drop=True)
        f = _one_patient(g)
        f.insert(0, "timestamp", g["timestamp"])
        f.insert(0, "patient_id", pid)
        if keep_label and LABEL_COL in g.columns:
            f[LABEL_COL] = g[LABEL_COL].values
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=FEATURE_COLS).reset_index(drop=True)
