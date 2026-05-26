import streamlit as st
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return joblib.load(
        "house-price-prediction/models/house_price_model.pkl"
    )

model = load_model()

# Page config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

# Title
st.title("🏠 House Price Prediction")

st.write("Enter house details below")

# Inputs
bedrooms = st.number_input("Bedrooms", 1, 10, 3)

bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.0)

sqft_living = st.number_input(
    "Living Area (sqft)",
    500,
    10000,
    2000
)

sqft_lot = st.number_input(
    "Lot Area (sqft)",
    500,
    50000,
    5000
)

floors = st.number_input("Floors", 1, 5, 2)

waterfront = st.selectbox(
    "Waterfront",
    [0, 1]
)

view = st.slider("View Rating", 0, 4, 2)

condition = st.slider("Condition", 1, 5, 3)

sqft_above = st.number_input(
    "Sqft Above",
    500,
    10000,
    1500
)

sqft_basement = st.number_input(
    "Basement Sqft",
    0,
    5000,
    500
)

yr_built = st.number_input(
    "Year Built",
    1900,
    2025,
    2010
)

# Prediction
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
        f"Predicted Price: ${prediction[0]:,.2f}"
    )