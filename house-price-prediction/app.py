import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load(
    "house-price-prediction/models/house_price_model.pkl"
)

# Page settings
st.set_page_config(page_title="House Price Prediction")

# Title
st.title("🏠 House Price Prediction System")

st.write("Enter house details below to predict price")

# User inputs
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1.0, 10.0)
sqft_living = st.number_input("Living Area (sqft)", 500, 10000)
sqft_lot = st.number_input("Lot Area (sqft)", 500, 50000)
floors = st.number_input("Floors", 1, 5)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.slider("View Rating", 0, 4)
condition = st.slider("Condition", 1, 5)
sqft_above = st.number_input("Sqft Above", 500, 10000)
sqft_basement = st.number_input("Basement Sqft", 0, 5000)
yr_built = st.number_input("Year Built", 1900, 2025)

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