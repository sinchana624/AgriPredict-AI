
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AgriPredict AI",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 AgriPredict AI")
st.subheader(
    "AI-Powered Crop Yield Prediction "
    "and Agricultural Decision Support"
)

st.write(
    "Explore agricultural conditions and prepare "
    "inputs for crop yield prediction."
)

st.divider()

# Sidebar inputs
st.sidebar.header("Agricultural Inputs")

crop = st.sidebar.selectbox(
    "Select Crop",
    ["Rice", "Wheat", "Maize", "Ragi"]
)

rainfall = st.sidebar.number_input(
    "Annual Rainfall (mm)",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

temperature = st.sidebar.number_input(
    "Average Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=27.0
)

st.header("Selected Conditions")

col1, col2, col3 = st.columns(3)

col1.metric("Selected Crop", crop)
col2.metric("Rainfall", f"{rainfall:.1f} mm")
col3.metric("Temperature", f"{temperature:.1f} °C")

st.info(
    "This is the initial dashboard interface. "
    "A reliable yield prediction will be added after "
    "the model is trained and evaluated using verified data."
)

# Illustrative chart data, not actual agricultural records
chart_data = pd.DataFrame({
    "Condition": ["Rainfall", "Temperature"],
    "Input Value": [rainfall, temperature]
})

fig = px.bar(
    chart_data,
    x="Condition",
    y="Input Value",
    title="Your Selected Agricultural Conditions"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "AgriPredict AI | Initial dashboard prototype"
)
