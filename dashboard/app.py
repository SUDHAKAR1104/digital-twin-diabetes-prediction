import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

# Make project root available to Python
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import predict_risk

st.set_page_config(
    page_title="Diabetes Digital Twin",
    page_icon="🩺",
    layout="wide",
)


st.title("🩺 Diabetes Digital Twin")

st.markdown(
    """
    ### AI-Powered Glucose Spike Risk Prediction

    A machine learning prototype that estimates the risk of a
    glucose spike within the next 2 hours using patient features.

    **TEAM TYST** · Machine Learning Hackathon Project
    """
)

with st.expander("🏆 Project Overview"):
    st.markdown(
        """
        **Objective:** Explore predicted glucose-spike risk
        using a machine learning model.

        **Key Features:**
        - Risk prediction with probability gauge
        - SHAP-based contributing factors
        - Glucose trend visualization
        - What-if activity and sleep simulator
        - Model evaluation and CSV report download

        **Dataset:** Synthetic patient data.

        **Disclaimer:** This prototype is not clinically
        validated and must not be used for medical decisions.
        """
    )

st.divider()

st.markdown(
    """
    ### AI-Powered Glucose Spike Risk Prediction
    Explore predicted risk, understand contributing factors,
    and compare hypothetical activity and sleep scenarios.
    """
)
st.caption("TEAM TYST | Machine Learning Prototype")
st.divider()

st.sidebar.title("🩺 Patient Profile")
st.sidebar.subheader("📋 Health Metrics")
st.sidebar.caption("Enter patient details to explore the model prediction.")
st.sidebar.divider()


glucose = st.sidebar.number_input("Current glucose (mg/dL)", 50, 400, 180)
heart_rate = st.sidebar.number_input("Heart rate (bpm)", 40, 200, 90)
steps = st.sidebar.number_input("Steps per 5 minutes", 0, 5000, 10)
sleep_hours = st.sidebar.number_input("Sleep hours", 0.0, 24.0, 6.0)
age = st.sidebar.number_input("Age", 1, 120, 25)
bmi = st.sidebar.number_input("BMI", 10.0, 70.0, 24.0)
hba1c = st.sidebar.number_input("HbA1c", 3.0, 20.0, 6.5)
years_since_diagnosis = st.sidebar.number_input(
    "Years since diagnosis", 0, 100, 2
)
st.sidebar.divider()
st.sidebar.caption("Prototype only — not for medical decisions.")


if "show_prediction" not in st.session_state:
    st.session_state["show_prediction"] = False

if st.button("Predict Glucose-Spike Risk", type="primary"):
    st.session_state["show_prediction"] = True

if st.session_state["show_prediction"]:
    patient = {
        "glucose": glucose,
        "heart_rate": heart_rate,
        "steps": steps,
        "sleep_hours": sleep_hours,
        "age": age,
        "bmi": bmi,
        "hba1c": hba1c,
        "years_since_diagnosis": years_since_diagnosis,
    }

    result = predict_risk(patient)
    probability = result["risk_probability"]
    
    st.divider()
    st.subheader("📈 Patient Glucose Trend")

    data_path = ROOT / "data" / "synthetic_patients.csv"
    df = pd.read_csv(data_path)

    patient_ids = sorted(df["patient_id"].unique())
    selected_patient = st.selectbox(
        "Select Patient",
        patient_ids,
    )

    patient_df = df[df["patient_id"] == selected_patient].copy()
    patient_df["timestamp"] = pd.to_datetime(patient_df["timestamp"])
    patient_df = patient_df.sort_values("timestamp").head(100)

    glucose_fig = go.Figure()

    glucose_fig.add_trace(
        go.Scatter(
            x=patient_df["timestamp"],
            y=patient_df["glucose"],
            mode="lines",
            name="Glucose",
        )
    )

    glucose_fig.add_hline(
        y=180,
        line_dash="dash",
        annotation_text="180 mg/dL reference",
    )

    glucose_fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Glucose (mg/dL)",
        height=400,
    )

    st.plotly_chart(glucose_fig, width="stretch")


    st.subheader("Why this risk?")

    for factor in result["top_factors"]:
        contribution = factor["contribution"]
        direction = factor["direction"]
        st.write(
            f"**{factor['feature']}** — {direction} "
            f"(SHAP contribution: {contribution:+.3f})"
        )

    st.subheader("👤 Patient Summary")

    summary1, summary2, summary3 = st.columns(3)

    summary1.metric("Glucose", f"{glucose} mg/dL")
    summary1.metric("Heart Rate", f"{heart_rate} bpm")

    summary2.metric("BMI", f"{bmi:.1f}")
    summary2.metric("HbA1c", f"{hba1c:.1f}")

    summary3.metric("Sleep", f"{sleep_hours:.1f} hours")
    summary3.metric("Steps / 5 min", f"{steps}")

    
    st.subheader("📊 Prediction Results")

    col1, col2 = st.columns(2)

    risk_level = result["risk_level"]

    with col1:
        st.markdown("**Predicted Risk Level**")
        if risk_level == "Low":
            st.success("🟢 LOW")
        elif risk_level == "Moderate":
            st.warning("🟡 MODERATE")
        else:
            st.error("🔴 HIGH")

    with col2:
        st.metric(
            "Estimated Spike Probability",
            f"{probability:.2%}",
        )


    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Estimated Spike Probability (%)"},
            gauge={"axis": {"range": [0, 100]}},
        )
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("🔮 What-if Simulator")
    st.write("Change activity and sleep values to compare model predictions.")

    whatif_steps = st.slider(
        "What-if steps per 5 minutes",
        min_value=0,
        max_value=5000,
        value=int(steps),
    )

    whatif_sleep = st.slider(
        "What-if sleep hours",
        min_value=0.0,
        max_value=24.0,
        value=float(sleep_hours),
        step=0.5,
    )

    whatif_patient = patient.copy()
    whatif_patient["steps"] = whatif_steps
    whatif_patient["sleep_hours"] = whatif_sleep

    whatif_result = predict_risk(whatif_patient)

    col1, col2 = st.columns(2)
    col1.metric("Original Probability", f"{probability:.2%}")
    col2.metric(
        "What-if Probability",
        f"{whatif_result['risk_probability']:.2%}",
        delta=f"{(whatif_result['risk_probability'] - probability):+.2%}",
        delta_color="inverse",
    )

    
    st.subheader("📊 Original vs What-if Comparison")
    st.caption("Compare predicted glucose-spike probability before and after hypothetical changes.")

    comparison_fig = go.Figure(
        data=[
            go.Bar(
                x=["Original", "What-if"],
                y=[
                    probability * 100,
                    whatif_result["risk_probability"] * 100,
                ],
                text=[
                    f"{probability:.1%}",
                    f"{whatif_result['risk_probability']:.1%}",
                ],
                textposition="outside",
                cliponaxis=False,
                width=0.5,
            )
        ]
    )

    comparison_fig.update_layout(
        yaxis_title="Estimated Spike Probability (%)",
        yaxis=dict(
            range=[0, 110],
            ticksuffix="%",
        ),
        xaxis_title="Scenario",
        showlegend=False,
        height=400,
        margin=dict(t=40, b=50, l=50, r=30),
        template="plotly_white",
    )

    comparison_fig.update_traces(
        marker_line_width=0,
        hovertemplate="%{x}<br>Probability: %{y:.1f}%<extra></extra>",
    )

    st.plotly_chart(comparison_fig, width="stretch")

   
    st.write(
        f"Original risk: {result['risk_level']}"
        f" | What-if risk: {whatif_result['risk_level']}"
    )

    st.subheader("🎯 Risk Level Comparison")

    risk_order = ["Low", "Moderate", "High"]

    original_risk = result["risk_level"]
    whatif_risk = whatif_result["risk_level"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Risk**")
        if original_risk == "Low":
            st.success("🟢 LOW")
        elif original_risk == "Moderate":
            st.warning("🟡 MODERATE")
        else:
            st.error("🔴 HIGH")

    with col2:
        st.markdown("**What-if Risk**")
        if whatif_risk == "Low":
            st.success("🟢 LOW")
        elif whatif_risk == "Moderate":
            st.warning("🟡 MODERATE")
        else:
            st.error("🔴 HIGH")

    original_index = risk_order.index(original_risk)
    whatif_index = risk_order.index(whatif_risk)

    if whatif_index < original_index:
        st.success("✅ The model predicts a lower risk category in this scenario.")
    elif whatif_index > original_index:
        st.error("⚠️ The model predicts a higher risk category in this scenario.")
    else:
        st.info("ℹ️ The predicted risk category remains the same.")

    

    st.info(
        "Prototype only. Risk categories and probabilities are not "
        "clinically validated and must not be used for medical decisions."
    )

    st.divider()
    st.subheader("📥 Download Prediction Report")

    report = pd.DataFrame([{
        "Glucose": glucose,
        "Heart Rate": heart_rate,
        "BMI": bmi,
        "HbA1c": hba1c,
        "Sleep Hours": sleep_hours,
        "Steps per 5 Minutes": steps,
        "Predicted Risk": result["risk_level"],
        "Spike Probability (%)": round(probability * 100, 2),
        "What-if Risk": whatif_result["risk_level"],
        "What-if Probability (%)": round(
            whatif_result["risk_probability"] * 100, 2
        ),
    }])

    st.download_button(
        label="⬇️ Download Report (CSV)",
        data=report.to_csv(index=False),
        file_name="diabetes_risk_report.csv",
        mime="text/csv",
    )

    
 

    # MODEL PERFORMANCE STARTS HERE
    st.divider()
    st.subheader("📈 Model Performance")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Accuracy", "89%")
    metric2.metric("Precision", "64%")
    metric3.metric("Recall", "71%")
    metric4.metric("F1-score", "68%")
    
    st.subheader("Confusion Matrix")

    confusion_fig = go.Figure(
        data=go.Heatmap(
            z=[
                [31299, 2493],
                [1806, 4482],
            ],
            x=["Predicted No Spike", "Predicted Spike"],
            y=["Actual Spike", "Actual No Spike"],
            text=[
                [31299, 2493],
                [1806, 4482],
            ],
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )

    confusion_fig.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        height=400,
    )

    st.plotly_chart(confusion_fig, width="stretch")

    st.caption(
        "Evaluation results from synthetic test data only. "
        "These metrics are not clinically validated."
    )

    st.divider()
    st.subheader("🧠 About the Model")

    with st.expander("Learn how this prediction works"):
        st.markdown(
            """
            **Model:** Random Forest Classifier

            **Dataset:** Synthetic patient data

            **Prediction Target:** Glucose spike within the next
            2 hours (`spike_2h`).

            **Input Features:**
            - Current glucose
            - Heart rate
            - Physical activity (steps)
            - Sleep duration
            - Age, BMI, HbA1c
            - Years since diagnosis

            **Explainability:** SHAP identifies features that
            contribute to the model's prediction.

            **Important Limitation:** This is a hackathon prototype
            trained and evaluated on synthetic data. It has not
            been clinically validated and must not be used for
            medical decisions.
            """
        )

