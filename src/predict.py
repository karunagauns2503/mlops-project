import joblib
import pandas as pd

MODEL_PATH = "models/crop_yield_model.pkl"

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

new_farm = pd.DataFrame({
    "Crop": ["Maize"],
    "Region": ["Region_C"],
    "Soil_Type": ["Sandy"],
    "Soil_pH": [6.5],
    "Rainfall_mm": [800],
    "Temperature_C": [25],
    "Humidity_pct": [65],
    "Fertilizer_Used_kg": [100],
    "Irrigation": ["Yes"],
    "Pesticides_Used_kg": [20],
    "Planting_Density": [15],
    "Previous_Crop": ["Rice"]
})

prediction = model.predict(new_farm)

print("\nCrop Yield Prediction")
print("-----------------------")
print(f"Predicted Yield: {prediction[0]:.2f} tons/hectare")