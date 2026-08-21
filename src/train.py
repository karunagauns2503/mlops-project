import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/crop_yield_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_yield_model.pkl")


df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")


TARGET = "Yield_ton_per_ha"

X = df.drop(columns=[TARGET])
y = df[TARGET]


categorical_features = [
    "Crop",
    "Region",
    "Soil_Type",
    "Irrigation",
    "Previous_Crop"
]

numeric_features = [
    "Soil_pH",
    "Rainfall_mm",
    "Temperature_C",
    "Humidity_pct",
    "Fertilizer_Used_kg",
    "Pesticides_Used_kg",
    "Planting_Density"
]


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}


results = []

best_model_name = None
best_pipeline = None
best_r2 = float("-inf")


print("\nModel Comparison")
print("=" * 70)

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_pipeline = pipeline


results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))


os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(best_pipeline, MODEL_PATH)

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Selected Model : {best_model_name}")
print(f"Best R²        : {best_r2:.4f}")
print(f"Model saved to : {MODEL_PATH}")
