from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "yield_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "combined_crop_yield.csv"
METRICS_PATH = ROOT / "reports" / "model_metrics.csv"

st.set_page_config(
    page_title="AgriPredict AI",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 AgriPredict AI")
st.subheader("AI-Powered Crop Yield Prediction")
st.write(
    "Explore historical crop yields and generate a baseline estimate "
    "using the trained machine-learning model."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: py src\\train_model.py")
    st.stop()

if not DATA_PATH.exists():
    st.error("Processed crop dataset not found.")
    st.stop()

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

st.sidebar.header("Prediction Inputs")

available_crops = sorted(data["crop"].dropna().unique().tolist())
crop = st.sidebar.selectbox("Select crop", available_crops)

min_year = int(data["year"].min())
max_year = int(data["year"].max())

prediction_year = st.sidebar.number_input(
    "Prediction year",
    min_value=min_year,
    max_value=2100,
    value=max_year + 1,
    step=1,
)

st.sidebar.caption(
    "The model uses crop and year only. Rainfall and temperature "
    "are not included in this model."
)

if st.button("Predict crop yield", type="primary"):
    input_data = pd.DataFrame(
        {"year": [int(prediction_year)], "crop": [crop]}
    )
    prediction = float(model.predict(input_data)[0])

    st.subheader("Baseline Yield Estimate")

    col1, col2 = st.columns(2)
    col1.metric("Selected crop", crop)
    col2.metric(
        "Estimated yield",
        f"{prediction:.3f} tonnes/hectare",
    )

    if prediction_year > max_year:
        st.warning(
            "This is an extrapolation beyond the training data. "
            "It is experimental and should not be treated as a reliable "
            "real-world forecast."
        )

    st.info(
        "This baseline uses historical year and crop information only. "
        "It does not account for weather, soil, irrigation, pests, or "
        "farming practices. Do not use it alone for agricultural decisions."
    )

st.divider()
st.header("Historical Yield Data")

selected_history = data[data["crop"] == crop].sort_values("year")

fig = px.line(
    selected_history,
    x="year",
    y="yield_tonnes_per_hectare",
    markers=True,
    title=f"Historical {crop} yield in the dataset",
    labels={
        "year": "Year",
        "yield_tonnes_per_hectare": "Yield (tonnes/hectare)",
    },
)
st.plotly_chart(fig, width='stretch')

st.subheader("Model Evaluation")
if METRICS_PATH.exists():
    metrics = pd.read_csv(METRICS_PATH).iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Test MAE", f"{metrics['test_MAE']:.3f}")
    col2.metric("Test RMSE", f"{metrics['test_RMSE']:.3f}")
    col3.metric("Test R²", f"{metrics['test_R2']:.3f}")

    st.caption(
        f"Evaluation period: {int(metrics['test_start_year'])}–"
        f"{int(metrics['test_end_year'])}; "
        f"{int(metrics['test_rows'])} test records."
    )

st.caption(
    "AgriPredict AI | Experimental historical trend baseline"
)
