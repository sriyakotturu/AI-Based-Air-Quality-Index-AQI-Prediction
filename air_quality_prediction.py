

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Create a sample dataset (simulated data)
data = {
    'PM2.5': [35, 50, 70, 90, 120, 150, 180, 200, 220, 250],
    'PM10': [50, 80, 100, 150, 180, 210, 250, 280, 300, 350],
    'NO2': [20, 30, 40, 50, 60, 70, 85, 95, 105, 120],
    'SO2': [10, 12, 15, 18, 20, 25, 30, 35, 40, 45],
    'CO': [0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
    'AQI': [50, 70, 90, 110, 130, 160, 190, 220, 250, 280]
}

df = pd.DataFrame(data)

# Step 2: Split the data
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Mean Absolute Error:", mae)
print("📈 R² Score:", r2)

# Step 5: Test custom input
sample_input = [[100, 150, 60, 25, 1.0]]
predicted_aqi = model.predict(sample_input)[0]
print(f"🌍 Predicted AQI for input {sample_input} = {predicted_aqi:.2f}")

# Step 6: Save model
import joblib
joblib.dump(model, "aqi_prediction_model.pkl")
print("✅ Model saved as aqi_prediction_model.pkl")
