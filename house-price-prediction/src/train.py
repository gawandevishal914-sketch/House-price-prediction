import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("data/housing.csv")

# Features
X = df[
    [
        'bedrooms',
        'bathrooms',
        'sqft_living',
        'sqft_lot',
        'floors',
        'waterfront',
        'view',
        'condition',
        'sqft_above',
        'sqft_basement',
        'yr_built'
    ]
]

# Target
y = df['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Sample predictions
print("\nSample Predictions:\n")

for i in range(5):
    print("Actual:", y_test.iloc[i])
    print("Predicted:", y_pred[i])
    print("----------------")

# Save model
joblib.dump(model, "models/house_price_model.pkl")

print("\nModel saved successfully!")