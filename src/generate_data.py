"""
generate_data.py
Synthetic Type 2 Diabetes patient data generator (EHR + wearable).

Run from the repo root:
    python src/generate_data.py
    python src/generate_data.py --patients 80 --days 21 --seed 7

Output: data/synthetic_patients.csv

Columns (agreed team schema):
    patient_id, timestamp, glucose, heart_rate, steps, sleep_hours,
    age, sex, bmi, hba1c, years_since_diagnosis, spike_2h
"""
import argparse
import os

import numpy as np
import pandas as pd

STEP_MIN = 5                      # sampling interval (minutes)
PER_DAY = 24 * 60 // STEP_MIN     # 288 samples per day
HORIZON = 120 // STEP_MIN         # 2 hours = 24 samples
SPIKE_THRESHOLD = 180             # mg/dL
TAIL = 48                         # meal effect lasts up to 4 hours


def meal_shape(tp_min=50):
    """Glucose response curve after a meal (peaks at tp_min, decays after)."""
    t = np.arange(TAIL) * STEP_MIN
    return (t / tp_min) * np.exp(1 - t / tp_min)


MEAL = meal_shape()


def make_patient_profile(rng, pid):
    hba1c = float(np.clip(rng.normal(7.4, 0.9), 5.9, 10.0))
    return {
        "patient_id": f"P{pid:03d}",
        "age": int(np.clip(rng.normal(55, 10), 30, 80)),
        "sex": rng.choice(["M", "F"]),
        "bmi": round(float(np.clip(rng.normal(30, 4.5), 21, 45)), 1),
        "hba1c": round(hba1c, 1),
        "years_since_diagnosis": int(np.clip(rng.exponential(6), 0, 30)),
        "activity_level": float(np.clip(rng.normal(1.0, 0.35), 0.4, 1.8)),
        "sleep_mean": float(np.clip(rng.normal(6.8, 0.8), 5.0, 8.5)),
    }


def simulate_patient(rng, prof, days, start):
    n = days * PER_DAY
    h = prof["hba1c"]

    steps = np.zeros(n)
    sleep_per_day = np.zeros(days)
    meals = []  # (sample_index, carbs_g)

    for d in range(days):
        off = d * PER_DAY
        sleep_per_day[d] = np.clip(rng.normal(prof["sleep_mean"], 1.0), 3.5, 9.5)

        # --- steps: ambient movement 06:30-22:30 plus 1-3 walking bouts ---
        wake, bed = int(6.5 * 12), int(22.5 * 12)
        amb = rng.poisson(25 * prof["activity_level"], bed - wake)
        steps[off + wake: off + bed] = amb
        for _ in range(rng.integers(1, 4)):
            s = off + rng.integers(wake, bed - 8)
            length = rng.integers(2, 9)  # 10-40 min
            steps[s: s + length] += rng.normal(500, 80, min(length, n - s)).clip(200, 800)

        # --- meals: breakfast, lunch, dinner (+ occasional snack) ---
        for hour, carb_mu in [(7.5, 55), (13.0, 70), (19.5, 75)]:
            t = int((hour + rng.normal(0, 0.6)) * 12)
            carbs = float(np.clip(rng.normal(carb_mu, 20), 20, 140))
            meals.append((off + t, carbs))
        if rng.random() < 0.3:
            meals.append((off + int(rng.uniform(15, 17.5) * 12), float(rng.uniform(15, 40))))

    # --- glucose ---
    baseline = 80 + (h - 5.5) * 14           # fasting level rises with HbA1c
    day_idx = np.arange(n) // PER_DAY
    sleep_series = sleep_per_day[day_idx]
    base = baseline + np.clip(7 - sleep_series, 0, None) * 4   # poor sleep -> higher glucose

    effect = np.zeros(n + TAIL)
    sens = 0.6 + 0.2 * (h - 6)               # insulin resistance proxy
    for idx, carbs in meals:
        if idx >= n:
            continue
        post = steps[idx + 3: idx + 13].sum()            # walking 15-60 min after meal
        reduction = np.clip(post / 2500, 0, 1) * 0.4
        sleep_boost = 1 + 0.05 * max(0, 7 - sleep_per_day[min(idx // PER_DAY, days - 1)])
        amp = carbs * sens * sleep_boost * (1 - reduction)
        effect[idx: idx + TAIL] += amp * MEAL

    # direct glucose-lowering effect of recent activity
    recent_steps = np.convolve(steps, np.ones(6) / 6, mode="same")
    activity_drop = 0.02 * recent_steps

    noise = np.zeros(n)
    for i in range(1, n):                    # AR(1) sensor/physiology noise
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 2.5)

    glucose = np.clip(base + effect[:n] - activity_drop + noise, 50, 400)

    # --- heart rate ---
    hod = (np.arange(n) % PER_DAY) / 12.0
    resting = 62 + 0.15 * (prof["age"] - 50) + 0.2 * (prof["bmi"] - 28)
    night = (hod < 6.0) | (hod > 22.5)
    hr = resting - 7 * night + 0.06 * steps + rng.normal(0, 2.5, n)
    hr = np.clip(hr, 45, 170)

    ts = pd.date_range(start, periods=n, freq=f"{STEP_MIN}min")
    df = pd.DataFrame({
        "patient_id": prof["patient_id"],
        "timestamp": ts,
        "glucose": glucose.round(1),
        "heart_rate": hr.round(0).astype(int),
        "steps": steps.round(0).astype(int),
        "sleep_hours": sleep_series.round(1),
        "age": prof["age"],
        "sex": prof["sex"],
        "bmi": prof["bmi"],
        "hba1c": prof["hba1c"],
        "years_since_diagnosis": prof["years_since_diagnosis"],
    })

    # --- label: does glucose exceed threshold in the NEXT 2 hours? ---
    g = df["glucose"]
    fut_max = g[::-1].rolling(HORIZON, min_periods=HORIZON).max()[::-1].shift(-1)
    df["spike_2h"] = (fut_max > SPIKE_THRESHOLD).astype("float")
    df.loc[fut_max.isna(), "spike_2h"] = np.nan
    return df.dropna(subset=["spike_2h"]).astype({"spike_2h": int})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=50)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_patients.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    frames = []
    for pid in range(1, args.patients + 1):
        prof = make_patient_profile(rng, pid)
        frames.append(simulate_patient(rng, prof, args.days, "2026-01-01"))
    data = pd.concat(frames, ignore_index=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    data.to_csv(args.out, index=False)

    print(f"Saved {len(data):,} rows for {args.patients} patients -> {args.out}")
    print(f"Spike rate (spike_2h=1): {data['spike_2h'].mean():.1%}")
    print(f"Glucose mean/min/max: {data['glucose'].mean():.0f} / "
          f"{data['glucose'].min():.0f} / {data['glucose'].max():.0f} mg/dL")


if __name__ == "__main__":
    main()
