
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Healthcare Digital Twin",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Healthcare Digital Twin")
st.caption("Patient Monitoring • Glucose Trends • Lifestyle Insights")

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_patients.csv"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["glucose", "heart_rate", "steps", "sleep_hours",
                "age", "bmi", "hba1c", "years_since_diagnosis", "spike_2h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["patient_id", "timestamp", "glucose"])


try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Could not load patient data: {exc}")
    st.stop()

if df.empty:
    st.error("The dataset contains no usable patient readings.")
    st.stop()

patient_ids = sorted(df["patient_id"].astype(str).unique())
preferred = ["P006", "P026", "P043"]
default_patient = next((p for p in preferred if p in patient_ids), patient_ids[0])

with st.sidebar:
    st.header("Patient Selection")
    patient_id = st.selectbox(
        "Choose patient",
        patient_ids,
        index=patient_ids.index(default_patient),
    )
    st.caption("Dataset: synthetic patient records")
    st.caption("Use this dashboard for demonstration, not medical diagnosis.")

patient = df[df["patient_id"].astype(str) == patient_id].sort_values("timestamp")
latest = patient.iloc[-1]

st.subheader(f"Patient {patient_id}")
st.caption(
    f"Latest available reading: {latest['timestamp']:%d %b %Y, %H:%M}"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Glucose", f"{latest['glucose']:.1f}", help="Dataset glucose units")
if pd.notna(latest["heart_rate"]):
    c2.metric("Latest Heart Rate", f"{latest['heart_rate']:.0f} bpm")
else:
    c2.metric("Latest Heart Rate", "N/A")
if pd.notna(latest["steps"]):
    c3.metric("Latest Steps", f"{latest['steps']:,.0f}")
else:
    c3.metric("Latest Steps", "N/A")
if pd.notna(latest["sleep_hours"]):
    c4.metric("Sleep Duration", f"{latest['sleep_hours']:.1f} h")
else:
    c4.metric("Sleep Duration", "N/A")

st.divider()

left, right = st.columns([2, 1])
with left:
    st.subheader("Glucose Trend")
    chart = px.line(
        patient,
        x="timestamp",
        y="glucose",
        labels={"timestamp": "Time", "glucose": "Glucose"},
        title=f"Glucose readings — {patient_id}",
    )
    chart.update_layout(hovermode="x unified")
    st.plotly_chart(chart, use_container_width=True)

with right:
    st.subheader("Patient Profile")
    for label, column, suffix in [
        ("Age", "age", " years"),
        ("Sex", "sex", ""),
        ("BMI", "bmi", ""),
        ("HbA1c", "hba1c", ""),
        ("Years Since Diagnosis", "years_since_diagnosis", " years"),
    ]:
        value = latest[column] if column in patient.columns else None
        if pd.notna(value):
            formatted = f"{value:.1f}{suffix}" if isinstance(value, (int, float)) else str(value)
            st.write(f"**{label}:** {formatted}")
        else:
            st.write(f"**{label}:** N/A")

st.divider()
st.subheader("Spike Label Summary")

if "spike_2h" in patient.columns and patient["spike_2h"].notna().any():
    valid_labels = patient["spike_2h"].dropna()
    spike_rate = valid_labels.mean() * 100
    a, b = st.columns(2)
    a.metric("Readings Labelled as Spike", f"{spike_rate:.1f}%")
    b.metric("Labelled Readings", f"{int(valid_labels.sum()):,} / {len(valid_labels):,}")
    st.caption(
        "This percentage describes the dataset's spike_2h labels. "
        "It is not a diabetes diagnosis or a validated prediction."
    )
else:
    st.info("No spike labels are available for this patient.")

st.subheader("Recent Readings")
st.dataframe(
    patient[["timestamp", "glucose", "heart_rate", "steps", "sleep_hours", "spike_2h"]]
    .tail(20)
    .sort_values("timestamp", ascending=False),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Educational prototype using synthetic data. Not intended for clinical decisions."
)
