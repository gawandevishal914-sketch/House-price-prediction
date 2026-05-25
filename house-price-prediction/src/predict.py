import joblib
import numpy as np

# Load trained model
model = joblib.load("models/house_price_model.pkl")

# User input
bedrooms = float(input("Enter number of bedrooms: "))
bathrooms = float(input("Enter number of bathrooms: "))
sqft_living = float(input("Enter square feet living area: "))
floors = float(input("Enter number of floors: "))

# Create input array
data = np.array([[bedrooms, bathrooms, sqft_living, floors]])

# Predict price
prediction = model.predict(data)

# Show result
print("Predicted House Price: $", round(prediction[0], 2))