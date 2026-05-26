import streamlit as st
import joblib
import numpy as np

# Cache model loading
@st.cache_resource
def load_model():
    return joblib.load("models/house_price_model.pkl")

# Load model
model = load_model()

# Page settings
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 House Price Prediction System")

st.write("Enter house details below to predict house price")

# Inputs
bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1.0,
    max_value=10.0,
    value=2.0
)

sqft_living = st.number_input(
    "Living Area (sqft)",
    min_value=500,
    max_value=10000,
    value=2000
)

sqft_lot = st.number_input(
    "Lot Area (sqft)",
    min_value=500,
    max_value=50000,
    value=5000
)

floors = st.number_input(
    "Floors",
    min_value=1,
    max_value=5,
    value=2
)

waterfront = st.selectbox(
    "Waterfront",
    [0, 1]
)

view = st.slider(
    "View Rating",
    0,
    4,
    2
)

condition = st.slider(
    "Condition",
    1,
    5,
    3
)

sqft_above = st.number_input(
    "Sqft Above",
    min_value=500,
    max_value=10000,
    value=1500
)

sqft_basement = st.number_input(
    "Basement Sqft",
    min_value=0,
    max_value=5000,
    value=500
)

yr_built = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2025,
    value=2010
)

# Prediction button
if st.button("Predict Price"):

    features = np.array([[
        bedrooms,
        bathrooms,
        sqft_living,
        sqft_lot,
        floors,
        waterfront,
        view,
        condition,
        sqft_above,
        sqft_basement,
        yr_built
    ]])

    prediction = model.predict(features)

    st.success(
        f"Predicted House Price: ${prediction[0]:,.2f}"
    )